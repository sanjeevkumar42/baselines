import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True, no_hyp=1):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.no_hyp = no_hyp

    def __call__(self, obs, reuse=False, action_idx=[[0]]):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            batch_size = tf.shape(obs)[0]

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.nb_actions * self.no_hyp,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.reshape(x, [batch_size, self.no_hyp, self.nb_actions])
            x = tf.nn.tanh(x)
            self.all_actions = x
            self.action_idx = tf.squeeze(action_idx, axis=1)
            idx = tf.range(0, batch_size) * self.no_hyp + tf.squeeze(action_idx, axis=1)
            self.idx = idx
            x = tf.gather(tf.reshape(x, (-1, self.nb_actions)), idx)
        return x


class Critic(Model):
    def __init__(self, name='critic', layer_norm=True):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars


class ConvCritic(Model):
    def __init__(self, name='critic', layer_norm=True, training=True):
        super(ConvCritic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.training = training

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = tf.layers.conv2d(obs, 32, (8, 8), strides=(4, 4), activation=tf.nn.relu)
            # x = tf.layers.batch_normalization(x, training=self.training)
            x = tf.layers.conv2d(x, 64, (4, 4), strides=(2, 2), activation=tf.nn.relu)
            # x = tf.layers.batch_normalization(x, training=self.training)
            x = tf.layers.conv2d(x, 64, (3, 3), strides=(1, 1), activation=tf.nn.relu)
            # x = tf.layers.batch_normalization(x, training=self.training)

            x = flatten(x)

            x = tf.concat([x, action], axis=-1)

            x = tf.layers.dense(x, 200, activation=tf.nn.relu)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.layers.dense(x, 200, activation=tf.nn.relu)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars


class ConvActor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True, no_hyp=1, training=True):
        super(ConvActor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm
        self.no_hyp = no_hyp
        self.training = training

    def __call__(self, obs, reuse=False, action_idx=[[0]]):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            batch_size = tf.shape(obs)[0]
            x = obs
            x = tf.layers.conv2d(x, 32, (8, 8), strides=(4, 4), activation=tf.nn.relu)
            # x = tf.layers.batch_normalization(x, training=self.training)
            x = tf.layers.conv2d(x, 64, (4, 4), strides=(2, 2), activation=tf.nn.relu)
            # x = tf.layers.batch_normalization(x, training=self.training)
            x = tf.layers.conv2d(x, 64, (3, 3), strides=(1, 1), activation=tf.nn.relu)
            # x = tf.layers.batch_normalization(x, training=self.training)

            x = flatten(x)
            x = tf.layers.dense(x, 200, activation=tf.nn.relu)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.layers.dense(x, 200, activation=tf.nn.relu)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)

            x = tf.layers.dense(x, self.nb_actions * self.no_hyp,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.reshape(x, [batch_size, self.no_hyp, self.nb_actions])
            x = tf.nn.tanh(x)

            idx = tf.range(0, batch_size) * self.no_hyp + tf.squeeze(action_idx, axis=1)
            x = tf.gather(tf.reshape(x, (-1, self.nb_actions)), idx)

        return x


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
