from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import json
import operator
import random, math
from collections import deque
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Pipe
from osim.env import ProstheticsEnv
from env import CustomizedProstheticsEnv, wrap_opensim, wrap_round2_opensim
from ddpg_agent import DDPGPolicyGraph


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("seed", None, "seed for np.random and random")


def f(conn,
      agent_config,
      observation_space,
      action_space,
      ckpt_dir
    ):
    """
    create graph and session for a specific model
    """
    # build graph
    g = tf.Graph()
    with g.as_default():
        # create learner queue
        with tf.device("/cpu"):
            global_step = tf.train.get_or_create_global_step()
            with tf.variable_scope("learner") as ps_scope:
                learner = DDPGPolicyGraph(
                    observation_space, action_space, agent_config, global_step)

        # create session and run
        with tf.train.MonitoredTrainingSession(
            is_chief=True,
            checkpoint_dir=ckpt_dir,
            save_checkpoint_secs=180,
            save_summaries_secs=180,
            log_step_count_steps=2000,
            config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)) as session:

            while True:
                state = conn.recv()
                if state is None:
                    break
                act = session.run(learner.output_actions, feed_dict={
                    learner.cur_observations: [state],
                    learner.stochastic: False,
                    learner.eps: .0})[0]
                conn.send(act)
                actions = conn.recv()
                q_vals = session.run(learner.q_value_tensor, feed_dict={
                    learner.obs_t: len(actions)*[state],
                    learner.act_t: actions})
                conn.send(q_vals)


def main():
    if FLAGS.seed is not None:
        seed = FLAGS.seed
    else:
        seed = int(time.time())
    np.random.seed(seed)
    env = ProstheticsEnv(False, difficulty=1, seed=seed+997)

    configs = ["examples/r2_v4dot1b.json", "examples/drl1.json", "examples/contd_r2_v4dot1b.json", "examples/contd_r2_v4dot1b.json", "examples/drl2.json"]
    ckpt_dirs = ["agents/v4dot1b/", "agents/drl1/", "agents/contd_v4dot1b/", "agents/tune_v4dot1b/", "agents/drl2/"]
    wrappers = list()
    ps = list()
    pps = list()
    
    for i in range(len(configs)):
        parent_conn, child_conn = Pipe()
        pps.append(parent_conn)

        agent_config = dict()
        # configurations
        with open(configs[i], 'r') as ips:
            specified_configs = json.load(ips)
        agent_config.update(specified_configs)
        wrapper = wrap_round2_opensim(
            env,
            skip=agent_config.get("skip", 3),
            start_index=0,
            clean=True)
        wrappers.append(wrapper)

        p = Process(target=f, args=(child_conn,
                                    agent_config,
                                    wrapper.observation_space,
                                    wrapper.action_space,
                                    ckpt_dirs[i]))
        p.start()
        ps.append(p)

    frames = deque([], maxlen=3)
    frames.append(np.zeros(223, dtype="float32"))
    frames.append(np.zeros(223, dtype="float32"))
    timestep_feature = 0
    obs = wrappers[0]._relative_dict_to_list(env.reset(project=False))
    frames.append(obs)
    state = obs + np.mean(list(frames), axis=0).tolist() + [timestep_feature/333.0]
    rwd = .0
    done = False

    while not done:
        for conn in pps:
            conn.send(state)
        actions = list()
        for conn in pps:
            actions.append(conn.recv())
        for conn in pps:
            conn.send(actions)
        qs = list()
        for conn in pps:
            qs.append(conn.recv())

        voted = np.zeros(len(actions))
        for i in range(len(qs)):
            qvals = [(idx, qv) for idx, qv in enumerate(qs[i])]
            qvals.sort(key=lambda elem: elem[1])
            qvals.reverse()
            for j in range(len(qvals)):
                voted[qvals[j][0]] += (1.0 / np.log2(1.0+j+1.0))
        choice = np.argmax(voted)
        # determined by argmax mean Q
        #qs = np.mean(np.array(qs), axis=1)
        #choice = np.argmax(qs)
        
        step_rwd = .0
        for _ in range(3):
            obs, reward, done, info = env.step(actions[choice], False)
            timestep_feature += 1
            step_rwd += reward
            obs = wrappers[0]._relative_dict_to_list(obs)
            frames.append(obs)
            if done:
                break
        state = obs + np.mean(list(frames), axis=0).tolist() + [timestep_feature/333.0]
        rwd += step_rwd
        print("step-{}\t+{}={}\t(controled by agent {})".format(
            timestep_feature, step_rwd, rwd, choice))

    print("{}\t{}".format(rwd, timestep_feature))
    for conn in pps:
        conn.send(None)
    for p in ps:
        p.join()
    print("done.")


if __name__=="__main__":
    main()
