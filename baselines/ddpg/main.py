import sys

if hasattr(sys, '__plen'):
    new = sys.path[sys.__plen:]
    del sys.path[sys.__plen:]
    p = getattr(sys, '__egginsert', 0)
    sys.path[p:p] = new
    sys.__egginsert = p + len(new)
from baselines.ddpg.ddpg import DDPG
from baselines.envs.maze2d import Maze2D
import argparse
import datetime
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.ddpg.training as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *

import gym
import tensorflow as tf
from mpi4py import MPI
import baselines.common.tf_util as U


def make_env(env_id):
    if env_id == 'Maze2D-v0':
        # obst = [(2, 30, 30, 5, 10), (2, 30, 60, 5, 10), (2, 65, 65, 35, 5),
        #         (1, 30, 75, 5), (1, 5, 45, 5), (2, 65, 20, 5, 20)]
        obst = [(2, 22, 6, 2, 6), (2, 10, 10, 2, 3), (2, 18, 20, 10, 2),
                (2, 2, 14, 2, 2), (2, 5, 28, 1, 4)]
        # env = Maze2D(maze_shape=(32, 32), obstacles=obst, targets=[(1, 30, 30, 2)])
        env = Maze2D(maze_shape=(32, 32), obstacles=obst, targets=[(1, 30, 30, 2)], obs_area=8, obs_complete=False)
        return env
    else:
        return gym.make(env_id)


def run(env_id, seed, noise_type, layer_norm, evaluation, outdir, no_hyp, **kwargs):
    # Configure things.
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0: logger.set_level(logger.DISABLED)

    # Create envs.
    env = make_env(env_id)
    outdir = os.path.join(outdir, env_id, '{}_{}'.format(no_hyp, kwargs['nb_epochs']))
    logger.configure(outdir)
    os.makedirs(outdir, exist_ok=True)

    env = bench.Monitor(env, os.path.join(outdir, "%i.monitor.json" % rank))
    gym.logger.setLevel(logging.WARN)
    logger.info('Output directory:{}, env:{}, no_hyp:{}'.format(outdir, env_id, no_hyp))

    if evaluation and rank == 0:
        eval_env = make_env(env_id)
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'), allow_early_resets=True)
        # env = bench.Monitor(env, None)
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                        sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm, no_hyp=no_hyp)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    # set_global_seeds(seed)
    # env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()
    weight_file = kwargs.pop('weight_file')
    if weight_file:
        evaluate(env, nb_episodes=kwargs.get('nb_epochs', 100), reward_scale=kwargs.get('reward_scale'),
                 render=kwargs.get('render'), param_noise=None, action_noise=None, actor=actor,
                 critic=critic,
                 critic_l2_reg=kwargs.get('critic_l2_reg'),
                 memory=memory, weight_file=weight_file, )
    else:
        training.train(env=env, eval_env=eval_env, param_noise=param_noise,
                       action_noise=action_noise, actor=actor, critic=critic, memory=memory, outdir=outdir, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def evaluate(env, nb_episodes, reward_scale, render, param_noise, action_noise, actor, critic, memory,
             critic_l2_reg, normalize_returns=False, normalize_observations=True, weight_file=None):
    rank = MPI.COMM_WORLD.Get_rank()

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))
    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
                 normalize_returns=normalize_returns,
                 normalize_observations=normalize_observations,
                 action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
                 reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    with U.single_threaded_session() as sess:
        agent.initialize(sess)
        if weight_file:
            saver = tf.train.Saver(actor.trainable_vars + critic.trainable_vars)
            saver.restore(sess, weight_file)
            # agent.actor_optimizer.sync()
            # agent.critic_optimizer.sync()
            pass
        sess.graph.finalize()

        agent.reset()
        obs = env.reset()
        total_reward = 0.0
        max_steps = 2000
        for ep in range(nb_episodes):
            i = 0
            done = False
            episode_reward = 0.0
            while not done and i < max_steps:
                action, q = agent.pi(obs, apply_noise=False, compute_Q=True)
                assert action.shape == env.action_space.shape

                assert max_action.shape == action.shape
                obs, r, done, info = env.step(max_action * action)
                episode_reward += r
                env.render()
                # print('Action:{}, reward:{}'.format(action, r))
                # time.sleep(0.1)
                i += 1
            total_reward += episode_reward
            logger.info("Episode:{}, reward:{}, steps:{}".format(ep, episode_reward, i))
            if done:
                obs = env.reset()

        logger.info("Average reward:{}, total reward:{}, episodes:{}".format((total_reward / nb_episodes), total_reward,
                                                                             nb_episodes))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='HalfCheetah-v1')
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-epochs', type=int, default=2000)  # with default settings, perform 1M steps total
    parser.add_argument('--nb-epoch-cycles', type=int, default=20)
    parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str,
                        default='adaptive-param_0.2')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    parser.add_argument('--no-hyp', type=int, default=1)
    parser.add_argument('--outdir', default='/data/out/')
    parser.add_argument('--weight-file', default=None)

    boolean_flag(parser, 'evaluation', default=True)
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    # Run actual script.
    run(**args)
