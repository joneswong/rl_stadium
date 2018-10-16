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
from search_ranking_gym_env import GoodStuffEpisodicEnv
from env import wrap_opensim
from osim.http.client import Client
from ddpg_agent import DDPGPolicyGraph
from replay_buffer import PrioritizedReplayBuffer


FLAGS = tf.flags.FLAGS
# no oss now, use default local dir tmp
tf.flags.DEFINE_string("checkpointDir", "ckpt", "oss buckets for saving checkpoint")
# user defined
tf.flags.DEFINE_string("config", "", "path of config file")
tf.flags.DEFINE_integer("repeat", 3, "number of action repeats")


SAMPLE_QUEUE_DEPTH = 2
REPLAY_QUEUE_DEPTH = 4
# number of threads for replaying
REPLAY_REPLICA = 4


AGENT_CONFIG=dict()


def main(_):
    shared_job_device = ""
    local_job_device = ""
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
        env = wrap_opensim(env, clean=True, repeat=FLAGS.repeat)
    elif AGENT_CONFIG["env"] == "sr":
        env = GoodStuffEpisodicEnv({
            "input_path": "/gruntdata/app_data/jones.wz/rl/search_ranking/A3gent/search_ranking/episodic_data.tsv",
            "shuffle": False, "task_index": -1, "num_partitions": 1, "num_epochs": 1})

                    
    # create agent (actor and learner)
    is_learner = True
    num_actors = 1
    
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
                    tf.TensorShape((AGENT_CONFIG["sample_batch_size"],)+env.observation_space.shape), tf.TensorShape((AGENT_CONFIG["sample_batch_size"],)+env.action_space.shape),
                    tf.TensorShape((AGENT_CONFIG["sample_batch_size"],)+env.observation_space.shape), tf.TensorShape((AGENT_CONFIG["sample_batch_size"],)), tf.TensorShape((AGENT_CONFIG["sample_batch_size"],))]
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
                
                # contribute metrics
                lt_return = tf.placeholder(
                    tf.float32, shape=())
                contribute_metrics = metrics_queue.enqueue([lt_return])

            tf.add_to_collection(tf.GraphKeys.READY_FOR_LOCAL_INIT_OP, tf.report_uninitialized_variables(learner.p_func_vars+learner.a_func_vars+learner.target_p_func_vars+learner.q_func_vars+learner.target_q_func_vars+learner.slot_vars))

    # create session and run
    # hack for ckpt
    if is_learner:
        pass
        #saver = tf.train.Saver()
    
    with tf.train.MonitoredTrainingSession(
        #server.target,
        is_chief=is_learner,
        checkpoint_dir=FLAGS.checkpointDir,
        save_checkpoint_secs=180,
        save_summaries_secs=180,
        log_step_count_steps=2000,
        config=tf.ConfigProto(allow_soft_placement=True)) as session:

        if is_learner:
            print("*************************learner started*************************")
            #saver.restore(session, "tmp/model.ckpt-21498")
            
            done = False
            episode_rwd = .0
            obs = env.reset()

            while not done:
                act = session.run(learner.output_actions, feed_dict={
                    learner.cur_observations: [obs],
                    learner.stochastic: False,
                    learner.eps: .0})[0]
                #q = session.run(learner.q_value_tensor, feed_dict={
                #    learner.obs_t: [obs],
                #    learner.act_t: [act]})
                obs, rwd, done, _ = env.step(act)
                episode_rwd += rwd
                print(episode_rwd)

            print(episode_rwd)

    print("done.")


if __name__ == "__main__":
    tf.app.run()
