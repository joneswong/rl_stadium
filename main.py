from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import six.moves as cpt
import json
import numpy as np
import tensorflow as tf
import gym
from osim.env import ProstheticsEnv

from env import wrap_opensim
from ddpg_agent import DDPGPolicyGraph
from replay_buffer import PrioritizedReplayBuffer


FLAGS = tf.flags.FLAGS
# pai tf used
tf.flags.DEFINE_string("ps_hosts", "", "ps_hosts")
tf.flags.DEFINE_string("worker_hosts", "", "worker_hosts")
tf.flags.DEFINE_string("job_name", "", "job_name")
tf.flags.DEFINE_integer("task_index", "-1", "task_index")
tf.flags.DEFINE_string("tables", "", "tables names")
tf.flags.DEFINE_string("outputs", "", "output tables names")
tf.flags.DEFINE_string("checkpointDir", "", "oss buckets for saving checkpoint")
tf.flags.DEFINE_string("buckets", "", "oss buckets")
# user defined
tf.flags.DEFINE_string("config", "", "path of config file")


# number of threads for producing
PRODUCER_REPLICA=8
# number of threads for replaying
REPLAY_REPLICA=4
# trajectory length
TRAJ_LEN = 32


AGENT_CONFIG=dict()


def hack_port(hosts):
    # the docker provided by xuhu
    # requires this trick to avoid port conflicts
    tps = [v.split(':') for v in hosts]
    return [tp[0]+':'+str(int(tp[1])+1) for tp in tps]


def main(_):
    # distributed pai tf
    ps_hosts = hack_port(FLAGS.ps_hosts.split(','))
    worker_hosts = hack_port(FLAGS.worker_hosts.split(','))
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    shared_job_device = "/job:ps/task:0"
    local_job_device = "/job:" + FLAGS.job_name + ("/task:%d" % FLAGS.task_index)
    global_variable_device = shared_job_device + "/cpu"

    # configurations
    with open(FLAGS.config, 'r') as ips:
        specified_configs = json.load(ips)
    AGENT_CONFIG.update(specified_configs)

    # create environment
    if AGENT_CONFIG["env"] == "pendulum":
        env = gym.make("Pendulum-v0")
    elif AGENT_CONFIG["env"] == "prosthetics":
        env = ProstheticsEnv(False)
        env = wrap_opensim(env)
    
    # create agent (actor and learner)
    is_learner = (FLAGS.job_name=="ps")#(FLAGS.task_index == 0)
    per_producer_delta_eps = AGENT_CONFIG["noise_scale"] / (PRODUCER_REPLICA * len(worker_hosts))
    
    # build graph
    g = tf.get_default_graph()
    with g.as_default():

        # create learner queue
        with tf.device(global_variable_device):
            with tf.variable_scope("learner") as ps_scope:
                learner = DDPGPolicyGraph(
                    env.observation_space, env.action_space, AGENT_CONFIG)
                tf.add_to_collection(tf.GraphKeys.INIT_OP, tf.group(*([v.initializer for v in learner.p_func_vars+learner.a_func_vars+learner.target_p_func_vars+learner.q_func_vars+learner.target_q_func_vars]+[v.initializer for v in learner.slot_vars])))
                dtypes = 5 * [tf.float32]
                shapes = [
                    tf.TensorShape((TRAJ_LEN,)+env.observation_space.shape), tf.TensorShape((TRAJ_LEN,)+env.action_space.shape),
                    tf.TensorShape((TRAJ_LEN,)+env.observation_space.shape), tf.TensorShape((TRAJ_LEN,)), tf.TensorShape((TRAJ_LEN,))]
                queue = tf.FIFOQueue(8, dtypes, shapes, shared_name="buffer")
                dequeue_op = queue.dequeue()

        with tf.device(local_job_device+"/cpu"):
            if not is_learner:
                actor = DDPGPolicyGraph(
                    env.observation_space, env.action_space, AGENT_CONFIG)
                tf.add_to_collection(tf.GraphKeys.LOCAL_INIT_OP, tf.group(*([v.initializer for v in actor.p_func_vars+actor.a_func_vars+actor.target_p_func_vars+actor.q_func_vars+actor.target_q_func_vars]+[v.initializer for v in actor.slot_vars])))

                # sync with learner
                sync_ops = list()
                for tgt, sc in zip(actor.p_func_vars, learner.p_func_vars):
                    sync_ops.append(tf.assign(tgt, sc))
                sync_op = tf.group(*(sync_ops))

                states = tf.placeholder(
                    tf.float32,
                    shape=(TRAJ_LEN,)+env.observation_space.shape)
                actions = tf.placeholder(
                    tf.float32,
                    shape=(TRAJ_LEN,)+env.action_space.shape)
                next_states = tf.placeholder(
                    tf.float32,
                    shape=(TRAJ_LEN,)+env.observation_space.shape)
                rewards = tf.placeholder(
                    tf.float32, shape=(TRAJ_LEN,))
                terminals = tf.placeholder(
                    tf.float32, shape=(TRAJ_LEN,))
                enqueue_op = queue.enqueue([states, actions, next_states, rewards, terminals])
            tf.add_to_collection(tf.GraphKeys.READY_FOR_LOCAL_INIT_OP, tf.report_uninitialized_variables(learner.p_func_vars+learner.a_func_vars+learner.target_p_func_vars+learner.q_func_vars+learner.target_q_func_vars+learner.slot_vars))

    # create session and run
    with tf.train.MonitoredTrainingSession(
        server.target,
        is_chief=is_learner,
        checkpoint_dir=FLAGS.checkpointDir,
        save_checkpoint_secs=600,
        save_summaries_secs=30,
        log_step_count_steps=50000,
        config=tf.ConfigProto(allow_soft_placement=True)) as session:

        if is_learner:

            print("*************************learner started*************************")

            replay_buffers = list()
            for idx in range(REPLAY_REPLICA):
                replay_actor = ReplayThread(
                    session, dequeue_op,
                    AGENT_CONFIG["buffer_size"] // REPLAY_REPLICA,
                    AGENT_CONFIG["prioritized_replay_alpha"],
                    AGENT_CONFIG["prioritized_replay_beta"],
                    AGENT_CONFIG["prioritized_replay_eps"],
                    AGENT_CONFIG["learning_starts"] // REPLAY_REPLICA,
                    AGENT_CONFIG["train_batch_size"], idx)
                replay_actor.start()
                replay_buffers.append(replay_actor)

            session.run(learner.update_target_expr)
            last_target_update_iter = 0
            iter_cnt = 0
            losses = list()
            while True:
                for rb in replay_buffers:
                    if not rb.outqueue.empty():
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_indexes) = rb.outqueue.get()
                        _, critic_loss, td_error = session.run([learner.opt_op, learner.loss.critic_loss, learner.loss.td_error], feed_dict={
                            learner.obs_t: obses_t,
                            learner.act_t: actions,
                            learner.rew_t: rewards,
                            learner.obs_tp1: obses_tp1,
                            learner.done_mask: dones,
                            learner.importance_weights: weights})
                        losses.append(critic_loss)
                        if len(losses) % 64 == 0:
                            print("mean_100_critic_loss=%.3f" % np.mean(losses[-100:]))
                        rb.inqueue.put((batch_indexes, td_error))
                        iter_cnt += 1
                        if iter_cnt-last_target_update_iter >= AGENT_CONFIG["target_network_update_freq"]:
                            session.run(learner.update_target_expr)
                            last_target_update_iter = iter_cnt
        else:

            print("*************************actor started*************************")

            stat_buf = cpt.queue.Queue(maxsize=2*PRODUCER_REPLICA)
            session.run(sync_op)
            producer_eps = FLAGS.task_index * (per_producer_delta_eps * PRODUCER_REPLICA)
            producers = list()
            for idx in range(PRODUCER_REPLICA):
                producer = ProducerThread(
                    AGENT_CONFIG["env"], actor, session,
                    (states, actions, next_states, rewards, terminals), enqueue_op,
                    producer_eps, stat_buf, idx)
                producers.append(producer)
                producer.start()
                producer_eps += per_producer_delta_eps

            # collect stats to report and coordinate
            collected_rwds = list()
            collected_lens = list()
            while True:
                (ep_rwd, ep_len) = stat_buf.get()
                collected_rwds.append(ep_rwd)
                collected_lens.append(ep_len)
                session.run(sync_op)
                if len(collected_lens) % 32 == 0:
                    print("mean_50_reward=%.3f\tmean_50_len=%d" % (np.mean(collected_rwds[-50:]), np.mean(collected_lens[-50:])))

    print("done.")


class ProducerThread(threading.Thread):
    def __init__(self, env_name, actor, sess,
                 enqueue_input, enqueue_op, eps,
                 stat_buf, producer_th):
        threading.Thread.__init__(self)
        self.daemon = True

        if env_name == "pendulum":
            self.env = gym.make("Pendulum-v0")
        elif env_name == "prosthetics":
            env = ProstheticsEnv(False)
            self.env = wrap_opensim(env)

        self.actor = actor
        self.sess = sess
        self.input_nodes = enqueue_input
        self.op = enqueue_op
        self.eps = eps

        self.cur_ob = self.env.reset()
        self.episode_rwd = .0
        self.episode_len = 0
        self.traj_obs = list()
        self.traj_act = list()
        self.traj_next_obs = list()
        self.traj_rwd = list()
        self.traj_done = list()

        self.stat_buf = stat_buf

        self.identifier = producer_th

    def run(self):
        while True:
            self.step()

    def step(self):
        # carefully handling the batch_size dimension
        act = self.sess.run(self.actor.output_actions, feed_dict={
            self.actor.cur_observations: [self.cur_ob], self.actor.eps: self.eps,
            self.actor.stochastic: True})[0]
        next_ob, rwd, done, _ = self.env.step(act)

        # collect and return trajectories
        self.traj_obs.append(self.cur_ob)
        self.traj_act.append(act)
        self.traj_next_obs.append(next_ob)
        self.traj_rwd.append(rwd)
        self.traj_done.append(done)
        if len(self.traj_rwd) == TRAJ_LEN:
            traj_obs, traj_act, traj_rwd, traj_next_obs, traj_done = self.actor.postprocess_trajectory(
                self.traj_obs, self.traj_act, self.traj_next_obs,
                self.traj_rwd, self.traj_done)
            #print("%d-th producer sampled %d steps" % (self.identifier, TRAJ_LEN))
            self.sess.run(self.op, feed_dict={
                self.input_nodes[0]: self.traj_obs, self.input_nodes[1]: self.traj_act,
                self.input_nodes[2]: self.traj_next_obs,
                self.input_nodes[3]: self.traj_rwd, self.input_nodes[4]: self.traj_done})
            #print("%d-th producer inserted %d steps" % (self.identifier, TRAJ_LEN))
            del self.traj_obs[:]
            del self.traj_act[:]
            del self.traj_next_obs[:]
            del self.traj_rwd[:]
            del self.traj_done[:]

        # natural epsidodic truncation and performance stats
        self.episode_rwd += rwd
        self.episode_len += 1
        if done:
            self.cur_ob = self.env.reset()
            self.sess.run(self.actor.reset_noise_op)
            self.stat_buf.put((self.episode_rwd, self.episode_len))
            self.episode_rwd = .0
            self.episode_len = 0
        else:
            self.cur_ob = next_ob


class ReplayThread(threading.Thread):
    def __init__(self, sess, dequeue_op,
                 buffer_size, alpha, beta, eps,
                 replay_start, train_batch_size,
                 replay_th):
        threading.Thread.__init__(self)
        self.daemon = True

        self.sess = sess
        self.op = dequeue_op
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=alpha)
        self.beta = beta
        self.eps = eps
        self.inqueue = cpt.queue.Queue(maxsize=8)
        self.outqueue = cpt.queue.Queue(maxsize=8)
        self.replay_start = replay_start
        self.train_batch_size = train_batch_size

        self.num_sample_steps = 0
        #self.dummy_weights = np.ones(TRAJ_LEN).astype("float32")

        self.identifier = replay_th

    def run(self):
        while True:
            self.step()

    def step(self):
        # update priority
        if not self.inqueue.empty():
            # (batch_indexes, td_errors)
            (batch_indexes, td_errors) = self.inqueue.get()
            new_priorities = (np.abs(td_errors) + self.eps)
            self.replay_buffer.update_priorities(batch_indexes, new_priorities)

        # sample a batch for learner
        if len(self.replay_buffer) > self.replay_start:
            # (obses_t, actions, rewards, obses_tp1, dones, weights, batch_indexes)
            batch_data = self.replay_buffer.sample(self.train_batch_size, beta=self.beta)
            self.outqueue.put(batch_data)

        # add trajectory
        traj = self.sess.run(self.op)
        for i in range(TRAJ_LEN):
            self.replay_buffer.add(
                traj[0][i], traj[1][i], traj[3][i], traj[2][i], traj[4][i], 1.0)
        self.num_sample_steps += TRAJ_LEN
        #print("%d-th replay thread added %d steps" % (self.identifier, self.num_sample_steps))


if __name__ == "__main__":
    tf.app.run()
