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
from search_ranking_gym_env import GoodStuffEpisodicEnv
from env import CustomizedProstheticsEnv, wrap_opensim, wrap_round2_opensim
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
REPLAY_QUEUE_DEPTH = 2#4
# number of threads for replaying
REPLAY_REPLICA = 4

AGENT_CONFIG=dict()


def get_port(hosts):
    """
    avoid port conflicts when used for pai job
    """
    tps = [v.split(':') for v in hosts]
    return [tp[0]+':'+str(int(tp[1])+1) for tp in tps]


def get_env(env_name):
    """
    create environment
    """
    if env_name == "pendulum":
        return gym.make("Pendulum-v0")
    elif env_name == "prosthetics":
        np.random.seed(FLAGS.task_index)
        env = ProstheticsEnv(False, seed=FLAGS.task_index)
        return wrap_opensim(env)
    elif env_name == "round2":
        env = CustomizedProstheticsEnv(
            False, difficulty=1, seed=FLAGS.task_index,
            random_start=AGENT_CONFIG.get("random_start", 0))
        return wrap_round2_opensim(
            env, skip=AGENT_CONFIG.get("skip", 3),
            horizon=AGENT_CONFIG.get("horizon", float('inf')),
            clean=AGENT_CONFIG.get("clean", False))
    elif env_name == "sr":
        return GoodStuffEpisodicEnv({
            "input_path": "/gruntdata/app_data/jones.wz/rl/search_ranking/A3gent/search_ranking/episodic_data.tsv",
            "shuffle": True, "task_index": FLAGS.task_index, "num_partitions": len(FLAGS.worker_hosts.split(','))})


def adjust_noise_stddev(distances, cur_stddev):
    if distances:
        num_samples = 0
        d = .0
        idx = len(distances) - 1
        while idx >= 0:
            num_samples += distances[idx][0]
            d += distances[idx][0] * distances[idx][1]
            if num_samples >= 128:
                break
            idx -= 1
        if len(distances) >= 128:
            del distances[:len(distances)//2]
        d = np.sqrt(d / num_samples)
        if d > AGENT_CONFIG["exploration_sigma"]:
            return 0.9 * cur_stddev
        else:
            return cur_stddev / 0.9
    return cur_stddev


def parse_personal_info(ips):
    access_id, access_key = None, None
    for line in ips:
        line = line.strip()
        if line and line[0] != '#':
            kv = line.split('=')
            if kv[0] == "access_id":
                access_id = kv[1]
            if kv[0] == "access_key":
                access_key = kv[1]
    return access_id, access_key


def main(_):
    # configurations
    with open(FLAGS.config, 'r') as ips:
        specified_configs = json.load(ips)
    AGENT_CONFIG.update(specified_configs)
    with open("odps_config.ini" if os.path.isfile("odps_config.ini") else "rl_stadium/odps_config.ini" , 'r') as ips:
        access_id, access_key = parse_personal_info(ips)
        assert access_id is not None and access_key is not None, "no personal information given"

    # distributed pai tf
    ps_hosts = get_port(FLAGS.ps_hosts.split(','))
    worker_hosts = get_port(FLAGS.worker_hosts.split(','))
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    shared_job_device = "/job:ps/task:0"
    local_job_device = "/job:" + FLAGS.job_name + ("/task:%d" % FLAGS.task_index)
    global_variable_device = shared_job_device + "/cpu"

    is_learner = (FLAGS.job_name=="ps")
    num_actors = len(worker_hosts)
    # PLEASE use more than one worker
    per_worker_eps = AGENT_CONFIG["noise_scale"] * (0.4**(1 + FLAGS.task_index / float(num_actors-1) * 7))
    print("actor specific epsilon={}".format(per_worker_eps))

    # create environment
    env = get_env(AGENT_CONFIG["env"])
    
    # build graph
    g = tf.get_default_graph()
    with g.as_default():
        with tf.device(global_variable_device):
            # global_step is necessary for Monitoring
            global_step = tf.train.get_or_create_global_step()
            # create learner and queues
            with tf.variable_scope("learner") as ps_scope:
                learner = DDPGPolicyGraph(
                    env.observation_space, env.action_space, AGENT_CONFIG, global_step)
                tf.add_to_collection(tf.GraphKeys.INIT_OP, tf.group(*([v.initializer for v in learner.p_func_vars+learner.a_func_vars+learner.target_p_func_vars+learner.q_func_vars+learner.target_q_func_vars]+[v.initializer for v in learner.slot_vars]+[global_step.initializer])))

                dtypes = 5 * [tf.float32]
                shapes = [
                    tf.TensorShape((AGENT_CONFIG["sample_batch_size"],)+env.observation_space.shape), tf.TensorShape((AGENT_CONFIG["sample_batch_size"],)+env.action_space.shape),
                    tf.TensorShape((AGENT_CONFIG["sample_batch_size"],)+env.observation_space.shape), tf.TensorShape((AGENT_CONFIG["sample_batch_size"],)), tf.TensorShape((AGENT_CONFIG["sample_batch_size"],))]
                queues = [tf.FIFOQueue(32, dtypes, shapes, shared_name="buffer{}".format(i)) for i in range(REPLAY_REPLICA)]
                dequeue_ops = [q.dequeue() for q in queues]

                metrics_queue = tf.FIFOQueue(
                    num_actors, dtypes=[tf.float32], shapes=[()],
                    shared_name="metrics_queue")
                collect_metrics = metrics_queue.dequeue()

        with tf.device(local_job_device+"/cpu"):
            # create actor and enqueue ops
            if not is_learner:
                actor = DDPGPolicyGraph(
                    env.observation_space, env.action_space, AGENT_CONFIG, global_step)
                #tf.add_to_collection(tf.GraphKeys.LOCAL_INIT_OP, tf.group(*([v.initializer for v in actor.p_func_vars+actor.a_func_vars+actor.target_p_func_vars+actor.q_func_vars+actor.target_q_func_vars]+[v.initializer for v in actor.slot_vars])))

                # sync with learner and add parameter space noise for policy net
                param_noise_stddev = tf.placeholder(tf.float32, shape=())
                sync_ops = list()
                cached_noise, gen_noise_ops, add_noise_ops, subtract_noise_ops = list(), list(), list(), list()
                label_idx = 0
                for tgt, sc in zip(actor.p_func_vars, learner.p_func_vars):
                    sync_ops.append(tf.assign(tgt, sc, use_locking=True))
                    if "fc" in tgt.name or "fully_connected" in tgt.name:
                        noise = tf.get_variable(name='noise{}'.format(label_idx), dtype=tf.float32, shape=tgt.shape)
                        label_idx += 1
                        cached_noise.append(noise)
                        gen_noise_ops.append(tf.assign(noise, tf.random_normal(shape=tgt.shape, stddev=param_noise_stddev, seed=FLAGS.task_index)))
                        add_noise_ops.append(tf.assign_add(tgt, noise))
                        subtract_noise_ops.append(tf.assign_add(tgt, -noise))
                sync_op = tf.group(*(sync_ops))
                gen_noise_ops = tf.group(*(gen_noise_ops))
                add_noise_ops = tf.group(*(add_noise_ops))
                subtract_noise_ops = tf.group(*(subtract_noise_ops))
                tf.add_to_collection(tf.GraphKeys.LOCAL_INIT_OP, tf.group(*([v.initializer for v in actor.p_func_vars+actor.a_func_vars+actor.target_p_func_vars+actor.q_func_vars+actor.target_q_func_vars]+[v.initializer for v in actor.slot_vars]+[v.initializer for v in cached_noise])))

                states = tf.placeholder(
                    tf.float32,
                    shape=(AGENT_CONFIG["sample_batch_size"],)+env.observation_space.shape)
                actions = tf.placeholder(
                    tf.float32,
                    shape=(AGENT_CONFIG["sample_batch_size"],)+env.action_space.shape)
                next_states = tf.placeholder(
                    tf.float32,
                    shape=(AGENT_CONFIG["sample_batch_size"],)+env.observation_space.shape)
                rewards = tf.placeholder(
                    tf.float32, shape=(AGENT_CONFIG["sample_batch_size"],))
                terminals = tf.placeholder(
                    tf.float32, shape=(AGENT_CONFIG["sample_batch_size"],))
                enqueue_ops = [q.enqueue([states, actions, next_states, rewards, terminals]) for q in queues]
                
                lt_return = tf.placeholder(
                    tf.float32, shape=())
                contribute_metrics = metrics_queue.enqueue([lt_return])

            tf.add_to_collection(tf.GraphKeys.READY_FOR_LOCAL_INIT_OP, tf.report_uninitialized_variables(learner.p_func_vars+learner.a_func_vars+learner.target_p_func_vars+learner.q_func_vars+learner.target_q_func_vars+learner.slot_vars))

    if is_learner:
        # save ckpt to a local folder during the training procedure
        os.system("mkdir tmp")
        destination_folder = "oss://142534/nips18/ckpt_" + time.asctime(time.localtime(time.time())).replace(' ', '_')
        os.system("osscmd --host=oss-cn-hangzhou-zmf.aliyuncs.com --id=" + access_id + " --key=" + access_key + " mkdir " + destination_folder)
    
    # create session
    with tf.train.MonitoredTrainingSession(
        server.target,
        is_chief=is_learner,
        checkpoint_dir='tmp',
        save_checkpoint_secs=9000 if AGENT_CONFIG["env"] in ["prosthetics", "round2"] else 600,
        save_summaries_secs=9000 if AGENT_CONFIG["env"] == ["prosthetics", "round2"] else 120,
        log_step_count_steps=250000 if AGENT_CONFIG["env"] == ["prosthetics", "round2"] else 1000,
        config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True)) as session:

        if is_learner:
            print("*************************learner started*************************")
            # spawn subprocesses function as Ray replay actor
            replay_buffers = list()
            data_ins = list()
            data_outs = list()
            priority_ins = list()
            for idx in range(REPLAY_REPLICA):
                data_in = Queue(8)
                data_out = Queue(8)
                priority_in = Queue(8)

                replay_actor = Process(target=f, args=(data_in, data_out, priority_in,))
                replay_actor.start()
                replay_buffers.append(replay_actor)
                data_ins.append(data_in)
                data_outs.append(data_out)
                priority_ins.append(priority_in)

            # multi-thread for running dequeue operations
            completed = threading.Event()
            op_runners = list()
            for idx in range(REPLAY_REPLICA):
                trd = DequeueThread(session, dequeue_ops[idx], data_ins[idx], completed)
                trd.start()
                op_runners.append(trd)

            # multi-thread for checking metrics
            metrics = list()
            metrics_channel = Queue()
            metrics_collecter = DequeueThread(session, collect_metrics, metrics_channel, completed)
            metrics_collecter.start()
            if AGENT_CONFIG["env"] == "pendulum":
                least_considered = 16
            elif AGENT_CONFIG["env"] in ["prosthetics", "round2"]:
                least_considered = num_actors
            elif AGENT_CONFIG["env"] == "sr":
                least_considered = 2048
            stop_criteria = AGENT_CONFIG["stop_criteria"]

            # begin training
            session.run(learner.update_target_expr)
            training_batch_cnt = 0
            train_batch_size = AGENT_CONFIG["train_batch_size"]
            sample_batch_size = AGENT_CONFIG["sample_batch_size"]
            last_target_update_iter = 0
            num_target_update = 0
            use_lr_decay = AGENT_CONFIG.get("lr_decay", False)
            init_actor_lr = AGENT_CONFIG["actor_lr"]
            init_critic_lr = AGENT_CONFIG["critic_lr"]
            num_sampled_timestep = 0
            losses = list()
            start_time = time.time()

            while True:
                if not use_lr_decay:
                    cur_actor_lr = init_actor_lr
                    cur_critic_lr = init_critic_lr
                else:
                    cur_actor_lr = 5e-5 + max(.0, 2e7-num_sampled_timestep)/(2e7) * (init_actor_lr - 5e-5)
                    cur_critic_lr = 5e-5 + max(.0, 2e7-num_sampled_timestep)/(2e7) * (init_critic_lr - 5e-5)

                for i in range(REPLAY_REPLICA):
                    if not data_outs[i].empty():
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_indexes) = data_outs[i].get()
                        _, critic_loss, td_error = session.run(
                            [learner.opt_op, learner.loss.critic_loss, learner.loss.td_error],
                            feed_dict={
                                learner.obs_t: obses_t,
                                learner.act_t: actions,
                                learner.rew_t: rewards,
                                learner.obs_tp1: obses_tp1,
                                learner.done_mask: dones,
                                learner.importance_weights: weights,
                                learner.cur_actor_lr: cur_actor_lr,
                                learner.cur_critic_lr: cur_critic_lr})
                        priority_ins[i].put((batch_indexes, td_error))

                        training_batch_cnt += 1
                        losses.append(critic_loss)
                        if training_batch_cnt % 64 == 0:
                            print("mean_critic_loss={}".format(np.mean(losses)))
                            del losses[:]
                        if training_batch_cnt-last_target_update_iter >= AGENT_CONFIG["target_network_update_freq"]:
                            session.run(learner.update_target_expr)
                            last_target_update_iter = training_batch_cnt
                            num_target_update += 1

                while not metrics_channel.empty():
                    metrics.append(metrics_channel.get())
                if len(metrics) >= least_considered:
                    perf = np.mean(metrics[-least_considered:])
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    print("mean_episodes_reward={}".format(perf))
                    num_sampled_timestep = np.sum([t.sampled_batch_cnt for t in op_runners]) * sample_batch_size
                    print("num_sampled_timestep={}".format(num_sampled_timestep))
                    print("num_train_timestep={}".format(training_batch_cnt * train_batch_size))
                    print("num_target_sync={}".format(num_target_update))
                    print("current_actor_lr={}".format(cur_actor_lr))
                    print("current_critic_lr={}".format(cur_critic_lr))
                    print("time_since_start={}".format(time.time()-start_time))
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    if perf >= stop_criteria:
                        completed.set()
                        for p in replay_buffers:
                            p.terminate()
                        time.sleep(3)
                        for p in replay_buffers:
                            p.join()
                        break
                    del metrics[:len(metrics)-least_considered//2]
        else:
            time.sleep(0.1*FLAGS.task_index)
            print("*************************actor started*************************")
            # frequently used arguments
            traj_len = AGENT_CONFIG["sample_batch_size"]
            max_policy_lag = AGENT_CONFIG["max_weight_sync_delay"]

            start_time = time.time()
            session.run(sync_op)
            sync_consumed = time.time() - start_time
            enqueue_consumed = .0
            report_consumed = .0

            cur_ob = env.reset()
            episode_rwd = .0
            episode_len = 0
            episode_rwds = list()
            episode_lens = list()
            episode_cnt = 0

            # TO DO: specify the ratio (now 1:0 or 1:1)
            use_action_noise = True
            use_param_noise = AGENT_CONFIG.get("param_noise", False)
            cur_param_noise_stddev = .1
            action_distance = list()

            last_sync_ts, timestep_cnt, traj_cnt = 0, 0, 0
            traj_obs, traj_acts, traj_next_obs, traj_rwds, traj_done_masks = list(), list(), list(), list(), list()

            # begin sampling
            while True:
                act = session.run(actor.output_actions, feed_dict={
                    actor.cur_observations: [cur_ob], actor.eps: per_worker_eps,
                    actor.stochastic: use_action_noise})[0]
                next_ob, rwd, done, _ = env.step(np.clip(act, .0, 1.0, out=act))

                episode_rwd += rwd
                episode_len += 1
                traj_obs.append(cur_ob)
                traj_acts.append(act)
                traj_next_obs.append(next_ob)
                traj_rwds.append(rwd)
                traj_done_masks.append(done)
                timestep_cnt += 1

                if done:
                    episode_rwds.append(episode_rwd)
                    episode_lens.append(episode_len)
                    episode_cnt += 1
                    if FLAGS.task_index >= 2*num_actors//3:
                        report_start_mnt = time.time()
                        session.run(contribute_metrics, feed_dict={lt_return: episode_rwd})
                        report_consumed += time.time() - report_start_mnt
                    if len(episode_lens) >= 8:
                        print("mean_reward={}\tmean_length={}".format(
                            np.mean(episode_rwds), np.mean(episode_lens)))
                        del episode_lens[:]
                        del episode_rwds[:]

                    # adjust action/parameter space noise
                    if use_param_noise:
                        # use parameter space noise in the coming episode
                        if use_action_noise:
                            use_action_noise = False
                            sync_start_mnt = time.time()
                            session.run(sync_op)
                            last_sync_ts = timestep_cnt
                            sync_consumed = time.time() - sync_start_mnt
                            # adjust noise stddev
                            cur_param_noise_stddev = adjust_noise_stddev(action_distance, cur_param_noise_stddev)
                            session.run(
                                gen_noise_ops,
                                feed_dict={param_noise_stddev: cur_param_noise_stddev})
                            session.run(add_noise_ops)
                        else:
                            use_action_noise = True
                            # calculate action distances
                            session.run(subtract_noise_ops)
                            act = session.run(actor.output_actions, feed_dict={
                                actor.cur_observations: traj_obs,
                                actor.eps: per_worker_eps, actor.stochastic: use_action_noise})
                            action_distance.append((len(traj_rwds), np.mean((act-traj_acts)**2)))
                    else:
                        session.run(actor.reset_noise_op)
                    episode_rwd = .0
                    episode_len = 0
                    cur_ob = env.reset()
                else:
                    cur_ob = next_ob

                # sync parameters
                if use_action_noise and timestep_cnt - last_sync_ts >= max_policy_lag:
                    sync_start_mnt = time.time()
                    session.run(sync_op)
                    last_sync_ts = timestep_cnt
                    sync_consumed = time.time() - sync_start_mnt

                # reach the sample_batch_size
                if len(traj_rwds) == traj_len:
                    traj_cnt += 1
                    traj_obs, traj_acts, traj_rwds, traj_next_obs, traj_done_masks = actor.postprocess_trajectory(
                        traj_obs, traj_acts, traj_next_obs,
                        traj_rwds, traj_done_masks)
                    enqueue_start_mnt = time.time()
                    session.run(enqueue_ops[FLAGS.task_index%REPLAY_REPLICA], feed_dict={
                        states: traj_obs, actions: traj_acts,
                        next_states: traj_next_obs,
                        rewards: traj_rwds, terminals: traj_done_masks})
                    enqueue_consumed += (time.time() - enqueue_start_mnt)

                    # calculate action distances
                    if use_param_noise and not use_action_noise:
                        closest_done_idx = max(0, traj_len - (timestep_cnt-last_sync_ts))
                        # parts of the trajectory use param noise
                        if closest_done_idx < traj_len:
                            session.run(subtract_noise_ops)
                            act = session.run(actor.output_actions, feed_dict={
                                actor.cur_observations: traj_obs[closest_done_idx:],
                                actor.eps: .0, actor.stochastic: use_action_noise})
                            action_distance.append((traj_len-closest_done_idx, np.mean((act-traj_acts[closest_done_idx:])**2)))
                            session.run(add_noise_ops)

                    del traj_obs[:]
                    del traj_acts[:]
                    del traj_next_obs[:]
                    del traj_rwds[:]
                    del traj_done_masks[:]

                    if traj_cnt % 8 == 0:
                        consumed_time = time.time() - start_time
                        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                        print("num_sampled_episode={}".format(episode_cnt))
                        print("num_sampled_timestep={}".format(timestep_cnt))
                        print("throughput={:.3}".format(float(8*traj_len)/consumed_time))
                        print("enqueue_ratio={:.2%}".format(enqueue_consumed/consumed_time))
                        print("sync_ratio={:.2%}".format(sync_consumed/consumed_time))
                        print("report_ratio={:.2%}".format(report_consumed/consumed_time))
                        print("param_noise_stddev={:.5}".format(cur_param_noise_stddev))
                        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                        start_time = time.time()
                        enqueue_consumed, sync_consumed, report_consumed = .0, .0, .0

    # upload util the session is closed
    if is_learner:
        print("upload {} files to {}".format(len([name for name in os.listdir("tmp") if os.path.isfile("tmp/"+name)]), destination_folder))
        os.system("osscmd --host=oss-cn-hangzhou-zmf.aliyuncs.com --id=" + access_id + " --key=" + access_key + " uploadfromdir tmp " + destination_folder)

    print("done.")


def f(data_in, data_out, priority_in):
    # like Ray ReplayActor
    # TO DO: coordinate training speed and sampling speed
    replay_buffer = PrioritizedReplayBuffer(
        AGENT_CONFIG["buffer_size"] // REPLAY_REPLICA,
        alpha=AGENT_CONFIG["prioritized_replay_alpha"])
    eps = AGENT_CONFIG["prioritized_replay_eps"]
    replay_start = AGENT_CONFIG["learning_starts"] // REPLAY_REPLICA
    train_batch_size = AGENT_CONFIG["train_batch_size"]
    beta = AGENT_CONFIG["prioritized_replay_beta"]

    while True:
        # add trajectory
        for _ in range(SAMPLE_QUEUE_DEPTH):
            if not data_in.empty():
                traj = data_in.get()
                for i in range(len(traj[0])):
                    replay_buffer.add(
                        traj[0][i], traj[1][i], traj[3][i], traj[2][i], traj[4][i], None)

        # sample a batch for learner
        if len(replay_buffer) > replay_start:
            for _ in range(REPLAY_QUEUE_DEPTH):
                if not data_out.full():
                    # (obses_t, actions, rewards, obses_tp1, dones, weights, batch_indexes)
                    batch_data = replay_buffer.sample(train_batch_size, beta=beta)
                    data_out.put(batch_data)

        # update priority
        while not priority_in.empty():
            (batch_indexes, td_errors) = priority_in.get()
            new_priorities = (np.abs(td_errors) + eps)
            replay_buffer.update_priorities(batch_indexes, new_priorities)


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
        while not self.signal.is_set():
            traj = self.sess.run(self.dequeue_op)
            self.recv.put(traj)
            self.sampled_batch_cnt += 1


if __name__ == "__main__":
    tf.app.run()
