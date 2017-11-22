import os
import time
from collections import deque
import pickle
import math
from baselines.ddpg.ddpg import DDPG
# from baselines.ddpg.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import baselines.common.tf_util as U

from baselines import logger
import numpy as np
import tensorflow as tf

# from mpi4py import MPI
mpi_mean = np.mean
mpi_sum = np.sum
mpi_std = np.std


def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
          normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
          popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory,
          tau=0.01, eval_env=None, param_noise_adaption_interval=50, outdir=None, no_hyp=1):
    rank = 0

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high

    summary_dir = os.path.join(outdir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    summary_writer = tf.summary.FileWriter(summary_dir)

    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
                 gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                 normalize_observations=normalize_observations,
                 batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
                 actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                 reward_scale=reward_scale, summary_writer=summary_writer, no_hyp=no_hyp)

    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.

    step = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    eval_dir = os.path.join(outdir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    with tf.Session(config=config) as sess:
        agent.initialize(sess)

        if rank == 0:
            saver = tf.train.Saver(actor.trainable_vars + critic.trainable_vars)
        else:
            saver = None

        # sess.graph.finalize()
        # Prepare everything.

        agent.reset()
        obs = env.reset()
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0

        for epoch in range(nb_epochs):
            epoch_start_time = time.time()
            for cycle in range(nb_epoch_cycles):
                # print("epoch:{}, cycle:{}".format(epoch, cycle))
                # Perform rollouts.
                for t_rollout in range(nb_rollout_steps):
                    # print("rollout:{}".format(t_rollout))
                    # Predict next action.
                    action, q, all_actions, sample = agent.pi(obs, apply_noise=True, compute_Q=True)

                    # print('action: {}'.format(action))
                    assert action.shape == env.action_space.shape

                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    assert max_action.shape == action.shape
                    # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    new_obs, r, done, info = env.step(max_action * action)
                    t += 1
                    if rank == 0 and render:
                        env.render()
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done, sample=sample)
                    obs = new_obs

                    if done:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        agent.reset()
                        obs = env.reset()

                # Train.
                epoch_actor_losses = []
                epoch_critic_losses = []
                epoch_adaptive_distances = []
                for t_train in range(nb_train_steps):
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al = agent.train(
                        train_step=epoch * nb_epoch_cycles * nb_train_steps + cycle * nb_train_steps + t_train)
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

            # Evaluate.

            if epoch > -1 and epoch % 20 == 0 and eval_env is not None:
                evaluate_model(eval_env, agent, epoch, eval_dir, render_eval, summary_writer)

            # Log stats.
            epoch_train_duration = time.time() - epoch_start_time
            logger.info("Epoch:{}, time taken:{}".format(epoch, epoch_train_duration))
            duration = time.time() - start_time
            stats = agent.get_stats()
            combined_stats = {}
            for key in sorted(stats.keys()):
                combined_stats[key] = mpi_mean(stats[key])

            # Rollout statistics.
            combined_stats['rollout/return'] = mpi_mean(epoch_episode_rewards)
            combined_stats['rollout/return_history'] = mpi_mean(np.mean(episode_rewards_history))
            combined_stats['rollout/episode_steps'] = mpi_mean(epoch_episode_steps)
            combined_stats['rollout/episodes'] = mpi_sum(epoch_episodes)
            combined_stats['rollout/actions_mean'] = mpi_mean(epoch_actions)
            combined_stats['rollout/actions_std'] = mpi_std(epoch_actions)
            combined_stats['rollout/Q_mean'] = mpi_mean(epoch_qs)

            # Train statistics.
            combined_stats['train/loss_actor'] = mpi_mean(epoch_actor_losses)
            combined_stats['train/loss_critic'] = mpi_mean(epoch_critic_losses)
            combined_stats['train/param_noise_distance'] = mpi_mean(epoch_adaptive_distances)

            # Total statistics.
            combined_stats['total/duration'] = mpi_mean(duration)
            combined_stats['total/steps_per_second'] = mpi_mean(float(t) / float(duration))
            combined_stats['total/episodes'] = mpi_mean(episodes)
            combined_stats['total/epochs'] = epoch + 1
            combined_stats['total/steps'] = t

            for key in sorted(combined_stats.keys()):
                logger.record_tabular(key, combined_stats[key])
            logger.dump_tabular()
            logger.info('')
            logdir = logger.get_dir()
            if rank == 0 and logdir:
                if hasattr(env, 'get_state'):
                    with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                        pickle.dump(env.get_state(), f)
                if eval_env and hasattr(eval_env, 'get_state'):
                    with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                        pickle.dump(eval_env.get_state(), f)
            if epoch > 0 and epoch % 50 == 0 and outdir:
                weight_file = os.path.join(outdir, 'model.ckpt')
                logger.info('Saving weights to :{}, epoch:{}'.format(weight_file, epoch))
                saver.save(sess, weight_file, global_step=epoch)


def evaluate_model(eval_env, agent, epoch, outdir, render_eval=False, summary_writer=None):
    logger.info("Starting evaluation at epoch:{}".format(epoch))
    eval_episode_rewards = []
    eval_qs = []
    eval_episode_reward = 0.
    total_eval_steps = []
    eval_done = False
    max_eval_steps = 2000
    max_action = eval_env.action_space.high
    eval_obs = eval_env.reset()
    all_eval_steps = 0
    for t_rollout in range(10):  # evaluate for 10 episodes
        eval_steps = 0
        ep_std = []
        ep_mean = []
        ep_selected = []
        ep_all_actions = []
        ep_reward = []
        ep_all_obs = []
        while not eval_done and eval_steps < max_eval_steps:
            eval_action, eval_q, all_actions, sample = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
            ep_all_obs.append(eval_obs)
            ep_std.append(all_actions.std())
            ep_mean.append(all_actions.mean())
            ep_selected.append(eval_action)
            ep_all_actions.append(all_actions)
            ep_all_obs.append(eval_obs)
            eval_obs, eval_r, eval_done, eval_info = eval_env.step(
                max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
            if render_eval:
                eval_env.render()
            ep_reward.append(eval_r)
            eval_episode_reward += eval_r
            eval_qs.append(eval_q)
            eval_steps += 1
            all_eval_steps += 1

        np.savez(os.path.join(outdir, '{}_{}'.format(epoch, t_rollout)), np.vstack(ep_all_actions),
                 np.asarray(ep_reward), np.vstack(ep_all_obs))

        if eval_done:
            eval_obs = eval_env.reset()
            eval_done = False

        eval_episode_rewards.append(eval_episode_reward)
        total_eval_steps.append(eval_steps)
        eval_episode_reward = 0.

    if summary_writer:
        add_scalar_summary(summary_writer, 'eval/return', mpi_mean(eval_episode_rewards), epoch)
        add_scalar_summary(summary_writer, 'eval/avg_steps', mpi_mean(total_eval_steps), epoch)
        add_scalar_summary(summary_writer, 'eval/Q', mpi_mean(eval_qs), epoch)
        add_scalar_summary(summary_writer, 'eval/episodes', len(eval_episode_rewards), epoch)
        summary_writer.flush()

    # summary_writer.flush()

    logger.info("Eval steps:{}, episode rewards:{}".format(total_eval_steps, eval_episode_rewards))
    eval_stats = {}
    eval_stats['eval/return'] = mpi_mean(eval_episode_rewards)
    eval_stats['eval/Q'] = mpi_mean(eval_qs)
    eval_stats['eval/episodes'] = mpi_mean(len(eval_episode_rewards))
    eval_stats['eval/avg_steps'] = mpi_mean(total_eval_steps)
    logger.info("Epoch:{}, evaluation stats:".format(epoch))
    for key in sorted(eval_stats.keys()):
        logger.record_tabular(key, eval_stats[key])
    logger.dump_tabular()
    logger.info('')


def add_scalar_summary(summary_writer, key, value, step):
    summary_writer.add_summary(as_summary(key, value), step)


def as_summary(tag, value):
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=tag, simple_value=value),
    ])
    return summary
