from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, os
import threading
from multiprocessing import Process, Queue
import six.moves as cpt
import json
import numpy as np
import tensorflow as tf
import gym
import opensim as osim
from osim.env import ProstheticsEnv

from env import wrap_opensim
from osim.http.client import Client
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
# no oss now, use default local dir tmp
tf.flags.DEFINE_string("checkpointDir", "ckpt", "oss buckets for saving checkpoint")
tf.flags.DEFINE_string("buckets", "", "oss buckets")
# user defined
tf.flags.DEFINE_string("config", "", "path of config file")


SAMPLE_QUEUE_DEPTH = 2
REPLAY_QUEUE_DEPTH = 4
# number of threads for replaying
REPLAY_REPLICA = 4
# trajectory length
TRAJ_LEN = 50


AGENT_CONFIG=dict()


def hack_port(hosts):
    # the docker provided by xuhu
    # requires this trick to avoid port conflicts
    tps = [v.split(':') for v in hosts]
    return [tp[0]+':'+str(int(tp[1])+1) for tp in tps]


def main(_):
    # distributed pai tf
    #ps_hosts = hack_port(FLAGS.ps_hosts.split(','))
    #worker_hosts = hack_port(FLAGS.worker_hosts.split(','))
    #cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    #server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    worker_hosts = [0, 1]

    shared_job_device = ""#"/job:ps/task:0"
    local_job_device = ""#"/job:" + FLAGS.job_name + ("/task:%d" % FLAGS.task_index)
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
    is_learner = (FLAGS.job_name == "ps" or FLAGS.task_index == -1)
    num_actors = len(worker_hosts)
    # do NOT use a single worker
    per_worker_eps = AGENT_CONFIG["noise_scale"] * (0.4**(1 + FLAGS.task_index / float(num_actors-1) * 7))
    print("actor eps: %.3f" % per_worker_eps)
    
    # build graph
    g = tf.get_default_graph()
    with g.as_default():

        # create learner queue
        with tf.device(global_variable_device):
            global_step = tf.train.get_or_create_global_step()
            with tf.variable_scope("learner") as ps_scope:
                learner = DDPGPolicyGraph(
                    env.observation_space, env.action_space, AGENT_CONFIG, global_step)
                tf.add_to_collection(tf.GraphKeys.INIT_OP, tf.group(*([v.initializer for v in learner.p_func_vars+learner.a_func_vars+learner.target_p_func_vars+learner.q_func_vars+learner.target_q_func_vars]+[v.initializer for v in learner.slot_vars]+[global_step.initializer])))
                dtypes = 5 * [tf.float32]
                shapes = [
                    tf.TensorShape((TRAJ_LEN,)+env.observation_space.shape), tf.TensorShape((TRAJ_LEN,)+env.action_space.shape),
                    tf.TensorShape((TRAJ_LEN,)+env.observation_space.shape), tf.TensorShape((TRAJ_LEN,)), tf.TensorShape((TRAJ_LEN,))]
                queues = [tf.FIFOQueue(32, dtypes, shapes, shared_name="buffer%d"%i) for i in range(REPLAY_REPLICA)]
                dequeue_ops = [q.dequeue() for q in queues]
                metrics_queue = tf.FIFOQueue(num_actors, dtypes=[tf.float32], shapes=[()], shared_name="metrics_queue")
                collect_metrics = metrics_queue.dequeue()

        with tf.device(local_job_device+"/cpu"):
            if not is_learner:
                actor = DDPGPolicyGraph(
                    env.observation_space, env.action_space, AGENT_CONFIG, global_step)
                tf.add_to_collection(tf.GraphKeys.LOCAL_INIT_OP, tf.group(*([v.initializer for v in actor.p_func_vars+actor.a_func_vars+actor.target_p_func_vars+actor.q_func_vars+actor.target_q_func_vars]+[v.initializer for v in actor.slot_vars])))

                # sync with learner
                sync_ops = list()
                for tgt, sc in zip(actor.p_func_vars, learner.p_func_vars):
                    sync_ops.append(tf.assign(tgt, sc, use_locking=True))
                sync_op = tf.group(*(sync_ops))

                # enqueue
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
                enqueue_ops = [q.enqueue([states, actions, next_states, rewards, terminals]) for q in queues]
                
                # contribute metrics
                lt_return = tf.placeholder(
                    tf.float32, shape=())
                contribute_metrics = metrics_queue.enqueue([lt_return])

            tf.add_to_collection(tf.GraphKeys.READY_FOR_LOCAL_INIT_OP, tf.report_uninitialized_variables(learner.p_func_vars+learner.a_func_vars+learner.target_p_func_vars+learner.q_func_vars+learner.target_q_func_vars+learner.slot_vars))

    # create session and run
    # hack for ckpt
    if is_learner:
        saver = tf.train.Saver()
    
    with tf.train.MonitoredTrainingSession(
        #server.target,
        is_chief=is_learner,
        checkpoint_dir='tmp',
        save_checkpoint_secs=180,
        save_summaries_secs=180,
        log_step_count_steps=2000,
        config=tf.ConfigProto(allow_soft_placement=True)) as session:

        if is_learner:
            print("*************************learner started*************************")
            saver.restore(session, "tmp/model.ckpt-66329486")

            # CrowdAI related
            remote_base = "http://grader.crowdai.org:1729"
            crowdai_token = "9956ed7abd0712f9c429966ee7dddbfd"

            client = Client(remote_base)
            obs = client.env_create(crowdai_token, env_id='ProstheticsEnv')
            obs = env._relative_dict_to_list(obs)
            while True:
                act = session.run(learner.output_actions, feed_dict={
                    learner.cur_observations: [obs],
                    learner.stochastic: False,
                    learner.eps: .0})[0]
                [obs, reward, done, info] = client.env_step(act.tolist(), True)
                obs = env._relative_dict_to_list(obs)
                if done:
                    obs = client.env_reset()
                    if not obs:
                        break
                    obs = env._relative_dict_to_list(obs)
            client.submit()

            # repeat actions
            client = Client(remote_base)
            obs = client.env_create(crowdai_token, env_id='ProstheticsEnv')
            obs = env._relative_dict_to_list(obs)
            repeat_cnt = 0
            act = None

            while True:
                if repeat_cnt == 0:
                    act = session.run(learner.output_actions, feed_dict={
                        learner.cur_observations: [obs],
                        learner.stochastic: False,
                        learner.eps: .0})[0]
                [obs, reward, done, info] = client.env_step(act.tolist(), True)
                repeat_cnt = (repeat_cnt + 1) % 3
                obs = env._relative_dict_to_list(obs)
                print(obs)
                if done:
                    obs = client.env_reset()
                    repeat_cnt = 0
                    if not obs:
                        break
                    obs = env._relative_dict_to_list(obs)
            client.submit()

            # local prediction
            #done = False
            #episode_rwd = .0
            #obs = env.reset()

            #while not done:
            #    act = session.run(learner.output_actions, feed_dict={
            #        learner.cur_observations: [obs],
            #        learner.stochastic: False,
            #        learner.eps: .0})[0]
            #    obs, rwd, done, _ = env.step(act)
            #    episode_rwd += rwd

            #print(episode_rwd)


    print("done.")


def f(data_in, data_out, priority_in):
    # like Ray ReplayActor
    replay_buffer = PrioritizedReplayBuffer(
        AGENT_CONFIG["buffer_size"], alpha=AGENT_CONFIG["prioritized_replay_alpha"])
    eps = AGENT_CONFIG["prioritized_replay_eps"]
    replay_start = AGENT_CONFIG["learning_starts"] // REPLAY_REPLICA
    train_batch_size = AGENT_CONFIG["train_batch_size"]
    beta = AGENT_CONFIG["prioritized_replay_beta"]

    sample_step_cnt = .0
    train_step_cnt = .0

    while True:
        # add trajectory
        for _ in range(SAMPLE_QUEUE_DEPTH):
            if not data_in.empty():
                traj = data_in.get()
                for i in range(TRAJ_LEN):
                    replay_buffer.add(
                        traj[0][i], traj[1][i], traj[3][i], traj[2][i], traj[4][i], None)
                sample_step_cnt += TRAJ_LEN

        # sample a batch for learner
        if len(replay_buffer) > replay_start:# and 64*sample_step_cnt > train_step_cnt:
            for _ in range(REPLAY_QUEUE_DEPTH):
                if not data_out.full():
                    # (obses_t, actions, rewards, obses_tp1, dones, weights, batch_indexes)
                    batch_data = replay_buffer.sample(train_batch_size, beta=beta)
                    data_out.put(batch_data)
                    train_step_cnt += train_batch_size

        # update priority
        while not priority_in.empty():
            (batch_indexes, td_errors) = priority_in.get()
            new_priorities = (np.abs(td_errors) + eps)
            replay_buffer.update_priorities(batch_indexes, new_priorities)

        if train_step_cnt > 1048576:
            print("sample_time_steps={}\ttrain_time_steps={}".format(sample_step_cnt, train_step_cnt))
            train_step_cnt = 0
            sample_step_cnt = 0


class DequeueThread(threading.Thread):
    def __init__(self, sess, dequeue_op, recv, signal):
        threading.Thread.__init__(self)
        self.daemon = True

        self.sess = sess
        self.dequeue_op = dequeue_op
        self.recv = recv

        self.signal = signal

    def run(self):
        while not self.signal.is_set():
            traj = self.sess.run(self.dequeue_op)
            self.recv.put(traj)


"""
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
"""


if __name__ == "__main__":
    tf.app.run()
