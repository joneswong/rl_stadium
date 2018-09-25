from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym.spaces import Box
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import model as ModelCatalog


A_SCOPE = "a_func"
P_SCOPE = "p_func"
P_TARGET_SCOPE = "target_p_func"
Q_SCOPE = "q_func"
Q_TARGET_SCOPE = "target_q_func"


class PNetwork(object):
    """Maps an observations (i.e., state) to an action where each entry takes
    value from (0, 1) due to the sigmoid function."""

    def __init__(self, model, dim_actions, hiddens=[64, 64],
                 activation="relu"):
        action_out = model.last_layer
        activation = tf.nn.__dict__[activation]
        for hidden in hiddens:
            action_out = layers.fully_connected(
                action_out, num_outputs=hidden, activation_fn=activation)
        # Use sigmoid layer to bound values within (0, 1)
        # shape of action_scores is [batch_size, dim_actions]
        self.action_scores = layers.fully_connected(
            action_out, num_outputs=dim_actions, activation_fn=tf.nn.sigmoid)


class ActionNetwork(object):
    """Acts as a stochastic policy for inference, but a deterministic policy
    for training, thus ignoring the batch_size issue when constructing a
    stochastic action."""

    def __init__(self,
                 p_values,
                 low_action,
                 high_action,
                 stochastic,
                 eps,
                 theta=0.15,
                 sigma=0.2):

        # shape is [None, dim_action]
        deterministic_actions = (
            (high_action - low_action) * p_values + low_action)

        exploration_sample = tf.get_variable(
            name="ornstein_uhlenbeck",
            dtype=tf.float32,
            initializer=low_action.size * [.0],
            trainable=False)
        normal_sample = tf.random_normal(
            shape=[low_action.size], mean=0.0, stddev=1.0)
        exploration_value = tf.assign_add(
            exploration_sample,
            theta * (.0 - exploration_sample) + sigma * normal_sample)
        stochastic_actions = deterministic_actions + eps * (
            high_action - low_action) * exploration_value

        self.actions = tf.cond(stochastic, lambda: stochastic_actions,
                               lambda: deterministic_actions)


class QNetwork(object):
    def __init__(self,
                 model,
                 action_inputs,
                 hiddens=[64, 64],
                 activation="relu"):
        q_out = tf.concat([model.last_layer, action_inputs], axis=1)
        activation = tf.nn.__dict__[activation]
        for hidden in hiddens:
            q_out = layers.fully_connected(
                q_out, num_outputs=hidden, activation_fn=activation)
        self.value = layers.fully_connected(
            q_out, num_outputs=1, activation_fn=None)


class ActorCriticLoss(object):
    def __init__(self,
                 q_t,
                 q_tp1,
                 q_tp0,
                 importance_weights,
                 rewards,
                 done_mask,
                 gamma=0.99,
                 n_step=1,
                 use_huber=False,
                 huber_threshold=1.0):

        q_t_selected = tf.squeeze(q_t, axis=len(q_t.shape) - 1)

        q_tp1_best = tf.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
        q_tp1_best_masked = (1.0 - done_mask) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = rewards + gamma**n_step * q_tp1_best_masked

        # compute the error (potentially clipped)
        self.td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        if use_huber:
            errors = _huber_loss(self.td_error, huber_threshold)
        else:
            errors = 0.5 * tf.square(self.td_error)

        self.critic_loss = tf.reduce_mean(importance_weights * errors)

        # for policy gradient
        self.actor_loss = -1.0 * tf.reduce_mean(q_tp0)
        self.total_loss = self.actor_loss + self.critic_loss


class DDPGPolicyGraph(object):
    def __init__(self, observation_space, action_space, config):
        if not isinstance(action_space, Box):
            raise TypeError(
                "Action space {} is not supported for DDPG.".format(
                    action_space))

        self.config = config
        self.dim_actions = action_space.shape[0]
        self.low_action = action_space.low
        self.high_action = action_space.high
        self.actor_optimizer = tf.train.AdamOptimizer(
            learning_rate=config["actor_lr"])
        self.critic_optimizer = tf.train.AdamOptimizer(
            learning_rate=config["critic_lr"])
        
        # Action inputs
        self.stochastic = tf.placeholder(tf.bool, (), name="stochastic")
        self.eps = tf.placeholder(tf.float32, (), name="eps")
        self.cur_observations = tf.placeholder(
            tf.float32, shape=(None, ) + observation_space.shape)

        # Actor: P (policy) network
        with tf.variable_scope(P_SCOPE) as scope:
            p_values = self._build_p_network(self.cur_observations)
            self.p_func_vars = _scope_vars(scope.name)

        # Action outputs
        with tf.variable_scope(A_SCOPE) as scope:
            self.output_actions = self._build_action_network(
                p_values, self.stochastic, self.eps)
            self.a_func_vars = _scope_vars(scope.name)

        with tf.variable_scope(A_SCOPE, reuse=True):
            exploration_sample = tf.get_variable(name="ornstein_uhlenbeck")
            self.reset_noise_op = tf.assign(exploration_sample,
                                            self.dim_actions * [.0])

        # Replay inputs
        self.obs_t = tf.placeholder(
            tf.float32,
            shape=(None, ) + observation_space.shape,
            name="observation")
        self.act_t = tf.placeholder(
            tf.float32, shape=(None, ) + action_space.shape, name="action")
        self.rew_t = tf.placeholder(tf.float32, [None], name="reward")
        self.obs_tp1 = tf.placeholder(
            tf.float32, shape=(None, ) + observation_space.shape)
        self.done_mask = tf.placeholder(tf.float32, [None], name="done")
        self.importance_weights = tf.placeholder(
            tf.float32, [None], name="weight")

        # p network evaluation
        with tf.variable_scope(P_SCOPE, reuse=True) as scope:
            self.p_t = self._build_p_network(self.obs_t)

        # target p network evaluation
        with tf.variable_scope(P_TARGET_SCOPE) as scope:
            p_tp1 = self._build_p_network(self.obs_tp1)
            self.target_p_func_vars = _scope_vars(scope.name)

        # Action outputs
        with tf.variable_scope(A_SCOPE, reuse=True):
            deterministic_flag = tf.constant(value=False, dtype=tf.bool)
            zero_eps = tf.constant(value=.0, dtype=tf.float32)
            output_actions = self._build_action_network(
                self.p_t, deterministic_flag, zero_eps)

            output_actions_estimated = self._build_action_network(
                p_tp1, deterministic_flag, zero_eps)

        # q network evaluation
        with tf.variable_scope(Q_SCOPE) as scope:
            q_t = self._build_q_network(self.obs_t, self.act_t)
            self.q_func_vars = _scope_vars(scope.name)
        with tf.variable_scope(Q_SCOPE, reuse=True):
            q_tp0 = self._build_q_network(self.obs_t, output_actions)

        # target q network evalution
        with tf.variable_scope(Q_TARGET_SCOPE) as scope:
            q_tp1 = self._build_q_network(self.obs_tp1,
                                          output_actions_estimated)
            self.target_q_func_vars = _scope_vars(scope.name)

        self.loss = self._build_actor_critic_loss(q_t, q_tp1, q_tp0)
        if config["l2_reg"] is not None:
            for var in self.p_func_vars:
                if "bias" not in var.name:
                    self.loss.actor_loss += (
                        config["l2_reg"] * 0.5 * tf.nn.l2_loss(var))
            for var in self.q_func_vars:
                if "bias" not in var.name:
                    self.loss.critic_loss += (
                        config["l2_reg"] * 0.5 * tf.nn.l2_loss(var))
        #self.opt_op = [self.actor_optimizer.minimize(self.loss.actor_loss),
        #               self.critic_optimizer.minimize(self.loss.critic_loss)]
        #self.opt_op = tf.group(*(self.opt_op))
        actor_grads_and_vars = _minimize_and_clip(
                self.actor_optimizer,
                self.loss.actor_loss,
                var_list=self.p_func_vars,
                clip_val=self.config["grad_norm_clipping"])
        critic_grads_and_vars = _minimize_and_clip(
            self.critic_optimizer,
            self.loss.critic_loss,
            var_list=self.q_func_vars,
            clip_val=self.config["grad_norm_clipping"])
        actor_grads_and_vars = [(g, v) for (g, v) in actor_grads_and_vars
                                if g is not None]
        critic_grads_and_vars = [(g, v) for (g, v) in critic_grads_and_vars
                                 if g is not None]
        self.opt_op = [self.actor_optimizer.apply_gradients(grads_and_vars=actor_grads_and_vars),
                       self.critic_optimizer.apply_gradients(grads_and_vars=critic_grads_and_vars)]
        self.opt_op = tf.group(*(self.opt_op))

        slots_variables = [
            self.actor_optimizer._slots[slot][key]
            for slot in sorted(self.actor_optimizer._slots)
            for key in sorted(self.actor_optimizer._slots[slot])
        ]
        slots_variables += list(self.actor_optimizer._get_beta_accumulators())
        slots_variables += [
            self.critic_optimizer._slots[slot][key]
            for slot in sorted(self.critic_optimizer._slots)
            for key in sorted(self.critic_optimizer._slots[slot])
        ]
        slots_variables += list(self.critic_optimizer._get_beta_accumulators())
        self.slot_vars = slots_variables

        # update_target_fn will be called periodically to copy Q network to
        # target Q network
        self.tau_value = config.get("tau")
        update_target_expr = []
        for var, var_target in zip(
                sorted(self.q_func_vars, key=lambda v: v.name),
                sorted(self.target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(
                var_target.assign(self.tau_value * var +
                                  (1.0 - self.tau_value) * var_target))
        for var, var_target in zip(
                sorted(self.p_func_vars, key=lambda v: v.name),
                sorted(self.target_p_func_vars, key=lambda v: v.name)):
            update_target_expr.append(
                var_target.assign(self.tau_value * var +
                                  (1.0 - self.tau_value) * var_target))
        self.update_target_expr = tf.group(*update_target_expr)

    def _build_q_network(self, obs, actions):
        return QNetwork(
            ModelCatalog.get_model(obs, 1, self.config["model"]), actions,
            self.config["critic_hiddens"],
            self.config["critic_hidden_activation"]).value

    def _build_p_network(self, obs):
        return PNetwork(
            ModelCatalog.get_model(obs, 1, self.config["model"]),
            self.dim_actions, self.config["actor_hiddens"],
            self.config["actor_hidden_activation"]).action_scores

    def _build_action_network(self, p_values, stochastic, eps):
        return ActionNetwork(p_values, self.low_action, self.high_action,
                             stochastic, eps, self.config["exploration_theta"],
                             self.config["exploration_sigma"]).actions

    def _build_actor_critic_loss(self, q_t, q_tp1, q_tp0):
        return ActorCriticLoss(
            q_t, q_tp1, q_tp0, self.importance_weights, self.rew_t,
            self.done_mask, self.config["gamma"], self.config["n_step"],
            self.config["use_huber"], self.config["huber_threshold"])

    def postprocess_trajectory(self, obs, actions, new_obs, rewards, dones):
        return _postprocess_dqn(self, obs, actions, new_obs, rewards, dones)


def _postprocess_dqn(policy_graph, obs, actions, new_obs, rewards, dones):
    # N-step Q adjustments
    if policy_graph.config["n_step"] > 1:
        adjust_nstep(policy_graph.config["n_step"],
                     policy_graph.config["gamma"], obs, actions, rewards,
                     new_obs, dones)

    return obs, actions, rewards, new_obs, dones

def adjust_nstep(n_step, gamma, obs, actions, rewards, new_obs, dones):
    """Rewrites the given trajectory fragments to encode n-step rewards.

    reward[i] = (
        reward[i] * gamma**0 +
        reward[i+1] * gamma**1 +
        ... +
        reward[i+n_step-1] * gamma**(n_step-1))

    The ith new_obs is also adjusted to point to the (i+n_step-1)'th new obs.

    If the episode finishes, the reward will be truncated. After this rewrite,
    all the arrays will be shortened by (n_step - 1).
    """
    traj_length = len(rewards)
    for i in range(traj_length):
        if dones[i]:
            continue  # episode end
        for j in range(1, min(n_step, traj_length-i)):
            new_obs[i] = new_obs[i + j]
            rewards[i] += gamma**j * rewards[i + j]
            if dones[i + j]:
                dones[i] = dones[i + j]
                break  # episode end
    # truncate ends of the trajectory
    #new_len = len(obs) - n_step + 1
    #for arr in [obs, actions, rewards, new_obs, dones]:
    #    del arr[new_len:]


def _huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5, delta * (tf.abs(x) - 0.5 * delta))


def _minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return gradients


def _scope_vars(scope, trainable_only=False):
    """
    Get variables inside a scope
    The scope can be specified as a string

    Parameters
    ----------
    scope: str or VariableScope
      scope in which the variables reside.
    trainable_only: bool
      whether or not to return only the variables that were marked as
      trainable.

    Returns
    -------
    vars: [tf.Variable]
      list of variables in `scope`.
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES
        if trainable_only else tf.GraphKeys.VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name)
