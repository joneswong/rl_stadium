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
# no oss now, use default local dir tmp
tf.flags.DEFINE_string("checkpointDir", "tmp", "oss buckets for saving checkpoint")
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
    is_learner = (FLAGS.job_name=="ps")
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
        os.system("mkdir tmp")
        os.system('osscmd --host=oss-cn-hangzhou-zmf.aliyuncs.com --id=LTAI7wU9Qj3OQo0t --key=JHIACB8W1vu6ZFFF6V6k1ZrqrG4I8k mkdir oss://142534/nips18/ckpt')
        stop_criteria = AGENT_CONFIG["stop_criteria"]
    
    with tf.train.MonitoredTrainingSession(
        server.target,
        is_chief=is_learner,
        checkpoint_dir='tmp',
        save_checkpoint_secs=36000,
        save_summaries_secs=18000,
        log_step_count_steps=500000,
        config=tf.ConfigProto(allow_soft_placement=True)) as session:

        if is_learner:
            print("*************************learner started*************************")
            
            replay_buffers = list()
            data_ins = list()
            data_outs = list()
            priority_ins = list()
            for idx in range(REPLAY_REPLICA):
                data_in = Queue(8)#cpt.queue.Queue(maxsize=8)
                data_out = Queue(8)#cpt.queue.Queue(maxsize=8)
                priority_in = Queue(8)#cpt.queue.Queue(maxsize=8)

                replay_actor = Process(target=f, args=(data_in, data_out, priority_in,))
                replay_actor.start()
                replay_buffers.append(replay_actor)
                data_ins.append(data_in)
                data_outs.append(data_out)
                priority_ins.append(priority_in)

            # multi-thread for dequeue operations
            completed = threading.Event()
            op_runners = list()
            for idx in range(REPLAY_REPLICA):
                trd = DequeueThread(session, dequeue_ops[idx], data_ins[idx], completed)
                trd.start()
                op_runners.append(trd)

            metrics = list()
            metrics_channel = Queue()
            metrics_collecter = DequeueThread(session, collect_metrics, metrics_channel, completed)
            metrics_collecter.start()

            session.run(learner.update_target_expr)
            training_batch_cnt = 0
            last_target_update_iter = 0
            num_target_update = 0
            losses = list()
            start_time = time.time()

            while True:
                for i in range(REPLAY_REPLICA):

                    if not data_outs[i].empty():
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_indexes) = data_outs[i].get()

                        _, critic_loss, td_error = session.run([learner.opt_op, learner.loss.critic_loss, learner.loss.td_error], feed_dict={
                            learner.obs_t: obses_t,
                            learner.act_t: actions,
                            learner.rew_t: rewards,
                            learner.obs_tp1: obses_tp1,
                            learner.done_mask: dones,
                            learner.importance_weights: weights})
                        priority_ins[i].put((batch_indexes, td_error))

                        training_batch_cnt += 1
                        losses.append(critic_loss)
                        if training_batch_cnt % 64 == 0:
                            cur_time = time.time()
                            print("%.3f timesteps/sec\tmean_100_critic_loss=%.3f >>> %d training_batches" % (
                                64.0*AGENT_CONFIG["train_batch_size"]/(cur_time-start_time),
                                np.mean(losses[-100:]),
                                training_batch_cnt))
                            if len(losses) >= 1000:
                                del losses[:500]
                            start_time = cur_time
                        
                        if training_batch_cnt-last_target_update_iter >= AGENT_CONFIG["target_network_update_freq"]:
                            session.run(learner.update_target_expr)
                            last_target_update_iter = training_batch_cnt
                            num_target_update += 1
                            print("sync with target nets {} times".format(num_target_update))

                while not metrics_channel.empty():
                    metrics.append(metrics_channel.get())
                if len(metrics) >= num_actors:
                    perf = np.mean(metrics[-num_actors:])
                    print(">>>>>>>>>>>>mean_episodes_reward={}".format(perf))
                    if perf >= stop_criteria:
                        completed.set()
                        for p in replay_buffers:
                            p.terminate()
                        time.sleep(3)
                        for p in replay_buffers:
                            p.join()
                        break
                    del metrics[:num_actors]

        else:
            print("*************************actor started*************************")

            horizon = AGENT_CONFIG["horizon"] or float('inf')

            start_time = time.time()
            session.run(sync_op)
            sync_consumed = time.time() - start_time

            episode_rwds = list()
            episode_lens = list()
            last_episode_cnt = 0
            cur_ob = env.reset()
            episode_rwd = .0
            episode_len = 0
            traj_cnt = 0

            enqueue_consumed = .0
            
            while True:
                traj_obs = list()
                traj_acts = list()
                traj_next_obs = list()
                traj_rwds = list()
                traj_done_masks = list()

                for i in range(TRAJ_LEN):
                    # carefully handling the batch_size dimension
                    act = session.run(actor.output_actions, feed_dict={
                        actor.cur_observations: [cur_ob], actor.eps: per_worker_eps,
                        actor.stochastic: True})[0]

                    next_ob, rwd, done, _ = env.step(act)

                    traj_obs.append(cur_ob)
                    traj_acts.append(act)
                    traj_next_obs.append(next_ob)
                    traj_rwds.append(rwd)
                    traj_done_masks.append(done)
                        
                    # natural epsidodic truncation and performance stats
                    episode_rwd += rwd
                    episode_len += 1
                    if episode_len >= horizon:
                        done = True
                    if done:
                        cur_ob = env.reset()
                        session.run(actor.reset_noise_op)
                        episode_rwds.append(episode_rwd)
                        episode_lens.append(episode_len)
                        if FLAGS.task_index >= 2*num_actors//3:
                            session.run(contribute_metrics, feed_dict={lt_return: episode_rwd})
                        episode_rwd = .0
                        episode_len = 0
                    else:
                        cur_ob = next_ob
            
                traj_obs, traj_acts, traj_rwds, traj_next_obs, traj_done_masks = actor.postprocess_trajectory(
                    traj_obs, traj_acts, traj_next_obs,
                    traj_rwds, traj_done_masks)

                enqueue_start = time.time()
                session.run(enqueue_ops[FLAGS.task_index%REPLAY_REPLICA], feed_dict={
                    states: traj_obs, actions: traj_acts,
                    next_states: traj_next_obs,
                    rewards: traj_rwds, terminals: traj_done_masks})
                enqueue_consumed += (time.time() - enqueue_start)

                del traj_obs[:]
                del traj_acts[:]
                del traj_next_obs[:]
                del traj_rwds[:]
                del traj_done_masks[:]
                traj_cnt += 1

                if traj_cnt % 8:
                    consumed_time = time.time() - start_time
                    print("%.3f timesteps/sec >>> %d timesteps" % (float(10*TRAJ_LEN)/consumed_time, traj_cnt*TRAJ_LEN))
                    print("enqueue ops cost {}".format(enqueue_consumed/consumed_time, '%'))
                    print("sync param ops cost {}".format(sync_consumed/consumed_time, '%'))

                    start_time = time.time()
                    session.run(sync_op)
                    sync_consumed = time.time() - start_time
                    enqueue_consumed = .0

                if len(episode_lens) - last_episode_cnt >= 10:
                    print("mean_50_reward=%.3f\tmean_50_len=%d >>> %d episodes" % (np.mean(episode_rwds[-50:]), np.mean(episode_lens[-50:]), len(episode_lens)))
                    last_episode_cnt = len(episode_lens)

    # upload util the session is closed
    if is_learner:
        print("there are {} files to upload.".format(len([name for name in os.listdir('tmp') if os.path.isfile(name)])))
        os.system('osscmd --host=oss-cn-hangzhou-zmf.aliyuncs.com --id=LTAI7wU9Qj3OQo0t --key=JHIACB8W1vu6ZFFF6V6k1ZrqrG4I8k uploadfromdir tmp oss://142534/nips18/ckpt')

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

        self.sampled_batch_cnt = 0
        self.signal = signal

    def run(self):
        start_time = time.time()

        while not self.signal.is_set():
            traj = self.sess.run(self.dequeue_op)
            self.recv.put(traj)
            self.sampled_batch_cnt += 1

        duration = time.time() - start_time
        sampled_timesteps = self.sampled_batch_cnt * TRAJ_LEN
        print("Sampled {} timesteps with throughput {} timesteps/sec".format(sampled_timesteps, sampled_timesteps/duration))


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
