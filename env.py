from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
import random, math
from collections import deque
import numpy as np
import gym
from gym import spaces
#import cv2
#cv2.ocl.setUseOpenCL(False)

from osim.env import ProstheticsEnv, rect


class CustomizedProstheticsEnv(ProstheticsEnv):
    def __init__(self, visualize=True, integrator_accuracy=5e-5, difficulty=0, seed=0, random_start=0):
        super(CustomizedProstheticsEnv, self).__init__(visualize=visualize, integrator_accuracy=integrator_accuracy, difficulty=difficulty, seed=seed)
        self._random_start = random_start
        np.random.seed(int(seed))

    def generate_new_targets(self, poisson_lambda = 300):
        nsteps = self.time_limit + 1
        rg = np.array(range(nsteps))
        velocity = np.zeros(nsteps)
        heading = np.zeros(nsteps)

        if self._random_start > 0:
            ch = np.random.randint(0, self._random_start)
            if ch == 0:
                velocity[0] = 1.25
                heading[0] = 0
            else:
                velocity[0] = random.uniform(1.0, 1.5)
                heading[0] = random.uniform(-math.pi/8, math.pi/8)
            self._random_start -= 1
        else:
            velocity[0] = 1.25
            heading[0] = 0

        change = np.cumsum(np.random.poisson(poisson_lambda, 10))

        for i in range(1,nsteps):
            velocity[i] = velocity[i-1]
            heading[i] = heading[i-1]

            if i in change:
                velocity[i] += random.choice([-1,1]) * random.uniform(-0.5,0.5)
                heading[i] += random.choice([-1,1]) * random.uniform(-math.pi/8,math.pi/8)

        trajectory_polar = np.vstack((velocity,heading)).transpose()
        self.targets = np.apply_along_axis(rect, 1, trajectory_polar)

    def reward_round2(self):
        """
        Override to increase the impact of velocity residual
        """
        state_desc = self.get_state_desc()
        prev_state_desc = self.get_prev_state_desc()
        # Small penalty for too much activation (cost of transport)
        penalty = np.sum(np.array(self.osim_model.get_activations())**2) * 0.001

        # Big penalty for not matching the vector on the X,Z projection.
        # No penalty for the vertical axis
        penalty += 3.0*(state_desc["body_vel"]["pelvis"][0] - state_desc["target_vel"][0])**2
        penalty += 3.0*(state_desc["body_vel"]["pelvis"][2] - state_desc["target_vel"][2])**2
        
        # Reward for not falling
        reward = 10.0
        
        return reward - penalty


class WalkingEnv(gym.Wrapper):
    def __init__(self, env, skip=3):
        """
        add 1 to original reward for each timestep except for the terminal one
        repeat an action for 4 timesteps
        """
        gym.Wrapper.__init__(self, env)
        self.observation_space.shape = (223,)
        self._skip = skip

    def _penalty(self, observation):
        x_head_pelvis = observation['body_pos']['head'][0]-observation['body_pos']['pelvis'][0]

        # height from pelvis to head is around 0.62
        # consider 0.62 * cos(60) first
        # consider 0.62 / sqrt(2) later
        accept_x1 = -0.31
        accept_x2 = 0.31
        if x_head_pelvis < accept_x1:
            pe = .667
            #pe = 5.0
            #done = True
        elif x_head_pelvis < accept_x2:
            pe = 0.0
            #done = False
        else:
            pe = 0.667
            #done = False

        z_head_pelvis = observation['body_pos']['head'][2]-observation['body_pos']['pelvis'][2]
        accept_z1 = -0.31
        accept_z2 = 0.31
        if z_head_pelvis < accept_z1:
            pe += 0.667
            #pe += 5.0
            #done = True
        elif z_head_pelvis < accept_z2:
            pass
        else:
            pe += 0.667
            #pe += 5.0
            #done = True

        pelvis_pos_y = observation["body_pos_rot"]["pelvis"][1]
        accept_y1 = -0.52
        accept_y2 = 0.52
        if pelvis_pos_y < accept_y1:
            pe += 2.0*(accept_y1-pelvis_pos_y)
        elif pelvis_pos_y < accept_y2:
            pass
        else:
            pe += 2.0*(pelvis_pos_y-accept_y2)

        # distance between left and right foot
        distance = np.abs(observation["body_pos"]["pros_foot_r"][0] - observation["body_pos"]["calcn_l"][0])
        if distance > 0.5:
            pe += 4.0*(np.abs(0.5-distance)**2)

        # cross leg
        theta = observation["body_pos_rot"]["pelvis"][1]
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        pelvis_pos_x, pelvis_pos_z = observation['body_pos']['pelvis'][0], observation['body_pos']['pelvis'][2]
        r_foot_x, r_foot_z =  observation['body_pos']['pros_foot_r'][0]-pelvis_pos_x, observation['body_pos']['pros_foot_r'][2]-pelvis_pos_z
        ip_r = r_foot_x * sin_theta + r_foot_z * cos_theta
        cross_leg_pe_r = max(.0-ip_r, .0)
        l_foot_x, l_foot_z =  observation['body_pos']['calcn_l'][0]-pelvis_pos_x, observation['body_pos']['calcn_l'][2]-pelvis_pos_z
        ip_l = l_foot_x * sin_theta + l_foot_z * cos_theta
        cross_leg_pe_l = max(ip_l-.0, .0)
        pe += 8 * (cross_leg_pe_r + cross_leg_pe_l)
        #print("{}\t{}".format(cross_leg_pe_r, cross_leg_pe_l))

        # stay at the starting position
        if observation['body_pos']['calcn_l'][0] <=0 and observation['body_pos']['pros_foot_r'][0] <= 0:
            pe += 1.0

        done = observation['body_pos']['pelvis'][1] <= 0.65

        return pe, done

    def _bonus(self, observation):
        pelvis_v = observation['body_vel']['pelvis'][0]
        lv = observation['body_vel']['femur_l'][0]
        rv = observation['body_vel']['femur_r'][0]

        if lv >= rv and 0.1 > observation["body_pos"]["calcn_l"][1]:
            return min(1.0, 5.0*(0.1 - observation["body_pos"]["calcn_l"][1])*max(lv-pelvis_v, .0))
        elif rv > lv and 0.1 > observation["body_pos"]["pros_foot_r"][1]:
            return min(1.0, 5.0*(0.1 - observation["body_pos"]["pros_foot_r"][1])*max(rv-pelvis_v, .0))
        else:
            return .0

    def _relative_dict_to_list(self, observation):
        res = []

        pelvs = {
            'body_pos': observation['body_pos']['pelvis'],
            'body_vel': observation['body_vel']['pelvis'],
            #'body_acc': list(map(lambda v: v/100.0, observation['body_acc']['pelvis']))
        }

        res += pelvs['body_pos']
        res += pelvs['body_vel']
        #res += pelvs['body_acc']

        # Body Observations
        for info_type in ['body_pos', 'body_vel']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += list(map(operator.sub, observation[info_type][body_part], pelvs[info_type]))

        #for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
        #                  'femur_l', 'femur_r', 'head',
        #                  'torso', 'pros_foot_r', 'pros_tibia_r']:
        #    res += list(map(lambda a,b: a/100.0-b, observation['body_acc'][body_part], pelvs['body_acc']))

        for info_type in ['body_pos_rot', 'body_vel_rot']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head', 'pelvis',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += observation[info_type][body_part]
                #if body_part == "pelvis":
                #    print(observation[info_type][body_part])
        #print("***********************************************************************************")
        #for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
        #                  'femur_l', 'femur_r', 'head', 'pelvis',
        #                  'torso', 'pros_foot_r', 'pros_tibia_r']:
        #    res += list(map(lambda v: v/1000.0, observation['body_acc_rot'][body_part]))

        # ground_pelvis
        res += list(map(operator.sub, observation['joint_pos']['ground_pelvis'][0:3], pelvs['body_pos']))
        res += observation['joint_pos']['ground_pelvis'][3:6]
        res += list(map(operator.sub, observation['joint_vel']['ground_pelvis'][0:3], pelvs['body_vel']))
        res += observation['joint_vel']['ground_pelvis'][3:6]
        #res += list(map(lambda a,b: a/100.0-b, observation['joint_acc']['ground_pelvis'][0:3], pelvs['body_acc']))
        #res += list(map(lambda v: v/1000.0, observation['joint_acc']['ground_pelvis'][3:6]))

        # joint
        for info_type in ['joint_pos', 'joint_vel']:
            for joint in ['ankle_l', 'ankle_r', 'back',
                          'hip_l', 'hip_r', 'knee_l', 'knee_r']:
                res += observation[info_type][joint]

        #for joint in ['ankle_l', 'ankle_r', 'back',
        #              'hip_l', 'hip_r', 'knee_l', 'knee_r']:
        #    res += list(map(lambda v: v/1000.0, observation['joint_acc'][joint]))

        # Muscle Observations
        for muscle in ['abd_l', 'abd_r', 'add_l', 'add_r',
                       'bifemsh_l', 'bifemsh_r', 'gastroc_l',
                       'glut_max_l', 'glut_max_r',
                       'hamstrings_l', 'hamstrings_r',
                       'iliopsoas_l', 'iliopsoas_r', 'rect_fem_l', 'rect_fem_r',
                       'soleus_l', 'tib_ant_l', 'vasti_l', 'vasti_r']:
            res.append(observation['muscles'][muscle]['activation'])
            #res.append(observation['muscles'][muscle]['fiber_force']/5000.0)
            res.append(observation['muscles'][muscle]['fiber_length'])
            res.append(observation['muscles'][muscle]['fiber_velocity'])

        return res
    
    def step(self, ac):
        total_reward = .0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(ac, False)
            penalty, strong_done = self._penalty(obs)
            b = self._bonus(obs)
            done = done if done else strong_done
            total_reward += (reward if done else reward+1.0) - penalty + b
            if done:
                break

        return self._relative_dict_to_list(obs), total_reward, done, info

    def reset(self, **kwargs):
        return self._relative_dict_to_list(self.env.reset(project=False, **kwargs))


class RunningEnv(gym.Wrapper):
    def __init__(self, env, skip=3):
        """
        add 1 to original reward for each timestep except for the terminal one
        repeat an action for 4 timesteps
        """
        gym.Wrapper.__init__(self, env)
        self.observation_space.shape = (223,)
        self._skip = skip

    def _penalty(self, observation):
        x_head_pelvis = observation['body_pos']['head'][0]-observation['body_pos']['pelvis'][0]

        # height from pelvis to head is around 0.62
        # consider 0.62 * cos(60) first
        # consider 0.62 / sqrt(2) later
        accept_x1 = -0.31
        accept_x2 = 0.31
        if x_head_pelvis < accept_x1:
            pe = .667
            #pe = 5.0
            #done = True
        elif x_head_pelvis < accept_x2:
            pe = 0.0
            #done = False
        else:
            pe = 0.667
            #done = False

        z_head_pelvis = observation['body_pos']['head'][2]-observation['body_pos']['pelvis'][2]
        accept_z1 = -0.31
        accept_z2 = 0.31
        if z_head_pelvis < accept_z1:
            pe += 0.667
            #pe += 5.0
            #done = True
        elif z_head_pelvis < accept_z2:
            pass
        else:
            pe += 0.667
            #pe += 5.0
            #done = True

        pelvis_pos_y = observation["body_pos_rot"]["pelvis"][1]
        accept_y1 = -0.52
        accept_y2 = 0.52
        if pelvis_pos_y < accept_y1:
            pe += 2.0*(accept_y1-pelvis_pos_y)
        elif pelvis_pos_y < accept_y2:
            pass
        else:
            pe += 2.0*(pelvis_pos_y-accept_y2)

        # distance between left and right foot
        distance = np.abs(observation["body_pos"]["pros_foot_r"][0] - observation["body_pos"]["calcn_l"][0])
        if distance > 0.5:
            pe += 4.0*(np.abs(0.5-distance)**2)

        # cross leg
        theta = observation["body_pos_rot"]["pelvis"][1]
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        pelvis_pos_x, pelvis_pos_z = observation['body_pos']['pelvis'][0], observation['body_pos']['pelvis'][2]
        r_foot_x, r_foot_z =  observation['body_pos']['pros_foot_r'][0]-pelvis_pos_x, observation['body_pos']['pros_foot_r'][2]-pelvis_pos_z
        ip_r = r_foot_x * sin_theta + r_foot_z * cos_theta
        cross_leg_pe_r = max(.0-ip_r, .0)
        l_foot_x, l_foot_z =  observation['body_pos']['calcn_l'][0]-pelvis_pos_x, observation['body_pos']['calcn_l'][2]-pelvis_pos_z
        ip_l = l_foot_x * sin_theta + l_foot_z * cos_theta
        cross_leg_pe_l = max(ip_l-.0, .0)
        pe += 8 * (cross_leg_pe_r + cross_leg_pe_l)
        #print("{}\t{}".format(cross_leg_pe_r, cross_leg_pe_l))

        # stay at the starting position
        #if observation['body_pos']['calcn_l'][0] <=0 and observation['body_pos']['pros_foot_r'][0] <= 0:
        #    pe += 1.0

        done = observation['body_pos']['pelvis'][1] <= 0.65

        return pe, done

    def _bonus(self, observation):
        pelvis_v = observation['body_vel']['pelvis'][0]
        lv = observation['body_vel']['femur_l'][0]
        rv = observation['body_vel']['femur_r'][0]

        if lv >= rv and 0.1 > observation["body_pos"]["calcn_l"][1]:
            return min(1.0, 5.0*(0.1 - observation["body_pos"]["calcn_l"][1])*max(lv-pelvis_v, .0))
        elif rv > lv and 0.1 > observation["body_pos"]["pros_foot_r"][1]:
            return min(1.0, 5.0*(0.1 - observation["body_pos"]["pros_foot_r"][1])*max(rv-pelvis_v, .0))
        else:
            return .0

    def _relative_dict_to_list(self, observation):
        res = []

        pelvs = {
            'body_pos': observation['body_pos']['pelvis'],
            'body_vel': observation['body_vel']['pelvis'],
            #'body_acc': list(map(lambda v: v/100.0, observation['body_acc']['pelvis']))
        }

        res += pelvs['body_pos']
        res += pelvs['body_vel']
        #res += pelvs['body_acc']

        # Body Observations
        for info_type in ['body_pos', 'body_vel']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += list(map(operator.sub, observation[info_type][body_part], pelvs[info_type]))

        #for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
        #                  'femur_l', 'femur_r', 'head',
        #                  'torso', 'pros_foot_r', 'pros_tibia_r']:
        #    res += list(map(lambda a,b: a/100.0-b, observation['body_acc'][body_part], pelvs['body_acc']))

        for info_type in ['body_pos_rot', 'body_vel_rot']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head', 'pelvis',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += observation[info_type][body_part]
                #if body_part == "pelvis":
                #    print(observation[info_type][body_part])
        #print("***********************************************************************************")
        #for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
        #                  'femur_l', 'femur_r', 'head', 'pelvis',
        #                  'torso', 'pros_foot_r', 'pros_tibia_r']:
        #    res += list(map(lambda v: v/1000.0, observation['body_acc_rot'][body_part]))

        # ground_pelvis
        res += list(map(operator.sub, observation['joint_pos']['ground_pelvis'][0:3], pelvs['body_pos']))
        res += observation['joint_pos']['ground_pelvis'][3:6]
        res += list(map(operator.sub, observation['joint_vel']['ground_pelvis'][0:3], pelvs['body_vel']))
        res += observation['joint_vel']['ground_pelvis'][3:6]
        #res += list(map(lambda a,b: a/100.0-b, observation['joint_acc']['ground_pelvis'][0:3], pelvs['body_acc']))
        #res += list(map(lambda v: v/1000.0, observation['joint_acc']['ground_pelvis'][3:6]))

        # joint
        for info_type in ['joint_pos', 'joint_vel']:
            for joint in ['ankle_l', 'ankle_r', 'back',
                          'hip_l', 'hip_r', 'knee_l', 'knee_r']:
                res += observation[info_type][joint]

        #for joint in ['ankle_l', 'ankle_r', 'back',
        #              'hip_l', 'hip_r', 'knee_l', 'knee_r']:
        #    res += list(map(lambda v: v/1000.0, observation['joint_acc'][joint]))

        # Muscle Observations
        for muscle in ['abd_l', 'abd_r', 'add_l', 'add_r',
                       'bifemsh_l', 'bifemsh_r', 'gastroc_l',
                       'glut_max_l', 'glut_max_r',
                       'hamstrings_l', 'hamstrings_r',
                       'iliopsoas_l', 'iliopsoas_r', 'rect_fem_l', 'rect_fem_r',
                       'soleus_l', 'tib_ant_l', 'vasti_l', 'vasti_r']:
            res.append(observation['muscles'][muscle]['activation'])
            #res.append(observation['muscles'][muscle]['fiber_force']/5000.0)
            res.append(observation['muscles'][muscle]['fiber_length'])
            res.append(observation['muscles'][muscle]['fiber_velocity'])

        return res
    
    def step(self, ac):
        total_reward = .0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(ac, False)
            #v = obs['body_vel']['pelvis'][0]
            #print("{}={}".format(reward, 9.0 - (3.0 - v)**2))
            penalty, strong_done = self._penalty(obs)
            b = self._bonus(obs)
            done = done if done else strong_done
            #total_reward += (reward if done else reward+1.0) - penalty + b
            total_reward += reward - penalty + b
            if done:
                break

        return self._relative_dict_to_list(obs), total_reward, done, info

    def reset(self, **kwargs):
        return self._relative_dict_to_list(self.env.reset(project=False, **kwargs))


class CleanEnv(gym.Wrapper):
    def __init__(self, env, skip=3):
        """
        add 1 to original reward for each timestep except for the terminal one
        repeat an action for 4 timesteps
        """
        gym.Wrapper.__init__(self, env)
        self.observation_space.shape = (223,)
        self._skip = skip

    def _relative_dict_to_list(self, observation):
        res = []

        pelvs = {
            'body_pos': observation['body_pos']['pelvis'],
            'body_vel': observation['body_vel']['pelvis'],
            #'body_acc': list(map(lambda v: v/100.0, observation['body_acc']['pelvis']))
        }

        res += pelvs['body_pos']
        res += pelvs['body_vel']
        #res += pelvs['body_acc']

        # Body Observations
        for info_type in ['body_pos', 'body_vel']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += list(map(operator.sub, observation[info_type][body_part], pelvs[info_type]))

        #for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
        #                  'femur_l', 'femur_r', 'head',
        #                  'torso', 'pros_foot_r', 'pros_tibia_r']:
        #    res += list(map(lambda a,b: a/100.0-b, observation['body_acc'][body_part], pelvs['body_acc']))

        for info_type in ['body_pos_rot', 'body_vel_rot']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head', 'pelvis',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += observation[info_type][body_part]
                #if body_part == "pelvis":
                #    print(observation[info_type][body_part])
        #print("***********************************************************************************")
        #for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
        #                  'femur_l', 'femur_r', 'head', 'pelvis',
        #                  'torso', 'pros_foot_r', 'pros_tibia_r']:
        #    res += list(map(lambda v: v/1000.0, observation['body_acc_rot'][body_part]))

        # ground_pelvis
        res += list(map(operator.sub, observation['joint_pos']['ground_pelvis'][0:3], pelvs['body_pos']))
        res += observation['joint_pos']['ground_pelvis'][3:6]
        res += list(map(operator.sub, observation['joint_vel']['ground_pelvis'][0:3], pelvs['body_vel']))
        res += observation['joint_vel']['ground_pelvis'][3:6]
        #res += list(map(lambda a,b: a/100.0-b, observation['joint_acc']['ground_pelvis'][0:3], pelvs['body_acc']))
        #res += list(map(lambda v: v/1000.0, observation['joint_acc']['ground_pelvis'][3:6]))

        # joint
        for info_type in ['joint_pos', 'joint_vel']:
            for joint in ['ankle_l', 'ankle_r', 'back',
                          'hip_l', 'hip_r', 'knee_l', 'knee_r']:
                res += observation[info_type][joint]

        #for joint in ['ankle_l', 'ankle_r', 'back',
        #              'hip_l', 'hip_r', 'knee_l', 'knee_r']:
        #    res += list(map(lambda v: v/1000.0, observation['joint_acc'][joint]))

        # Muscle Observations
        for muscle in ['abd_l', 'abd_r', 'add_l', 'add_r',
                       'bifemsh_l', 'bifemsh_r', 'gastroc_l',
                       'glut_max_l', 'glut_max_r',
                       'hamstrings_l', 'hamstrings_r',
                       'iliopsoas_l', 'iliopsoas_r', 'rect_fem_l', 'rect_fem_r',
                       'soleus_l', 'tib_ant_l', 'vasti_l', 'vasti_r']:
            res.append(observation['muscles'][muscle]['activation'])
            #res.append(observation['muscles'][muscle]['fiber_force']/5000.0)
            res.append(observation['muscles'][muscle]['fiber_length'])
            res.append(observation['muscles'][muscle]['fiber_velocity'])

        return res
    
    def step(self, ac):
        total_reward = .0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(ac, False)
            total_reward += reward
            if done:
                break

        return self._relative_dict_to_list(obs), total_reward, done, info

    def reset(self, **kwargs):
        return self._relative_dict_to_list(self.env.reset(project=False, **kwargs))


def wrap_opensim(env, contd=False, clean=False, repeat=3):
    if clean:
        return CleanEnv(env, repeat)

    if contd:
        env = RunningEnv(env)
    else:
        env = WalkingEnv(env)
    return env


class Round2WalkingEnv(gym.Wrapper):
    def __init__(self, env, skip=3, use_hcf=False):
        """
        add 1 to original reward for each timestep except for the terminal one
        repeat an action for 4 timesteps
        """
        gym.Wrapper.__init__(self, env)
        self._use_hcf = use_hcf
        self.observation_space.shape = (245 if use_hcf else 224,)
        self._skip = skip
        if use_hcf:
            self.frames = deque([], maxlen=self._skip)
        self.timestep_feature = 0

    def _penalty(self, observation):
        x_head_pelvis = observation['body_pos']['head'][0]-observation['body_pos']['pelvis'][0]

        # height from pelvis to head is around 0.62
        # consider 0.62 * cos(60) first
        # consider 0.62 / sqrt(2) later
        accept_x1 = -0.31
        accept_x2 = 0.31
        if x_head_pelvis < accept_x1:
            pe = .667
            #pe = 5.0
            #done = True
        elif x_head_pelvis < accept_x2:
            pe = 0.0
            #done = False
        else:
            pe = 0.667
            #done = False

        z_head_pelvis = observation['body_pos']['head'][2]-observation['body_pos']['pelvis'][2]
        accept_z1 = -0.31
        accept_z2 = 0.31
        if z_head_pelvis < accept_z1:
            pe += 0.667
            #pe += 5.0
            #done = True
        elif z_head_pelvis < accept_z2:
            pass
        else:
            pe += 0.667
            #pe += 5.0
            #done = True

        # distance between left and right foot
        distance = (observation["body_pos"]["pros_foot_r"][0] - observation["body_pos"]["calcn_l"][0])**2 + (observation["body_pos"]["pros_foot_r"][2] - observation["body_pos"]["calcn_l"][2])**2
        if distance > 0.5:
            pe += 5.0 * (distance - 0.5)

        # cross leg
        theta = observation["body_pos_rot"]["pelvis"][1]
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        pelvis_pos_x, pelvis_pos_z = observation['body_pos']['pelvis'][0], observation['body_pos']['pelvis'][2]
        r_foot_x, r_foot_z =  observation['body_pos']['pros_foot_r'][0]-pelvis_pos_x, observation['body_pos']['pros_foot_r'][2]-pelvis_pos_z
        ip_r = r_foot_x * sin_theta + r_foot_z * cos_theta
        cross_leg_pe_r = max(.0-ip_r, .0)
        l_foot_x, l_foot_z =  observation['body_pos']['calcn_l'][0]-pelvis_pos_x, observation['body_pos']['calcn_l'][2]-pelvis_pos_z
        ip_l = l_foot_x * sin_theta + l_foot_z * cos_theta
        cross_leg_pe_l = max(ip_l-.0, .0)
        pe += 8 * (cross_leg_pe_r + cross_leg_pe_l)

        # heading towards target velocity
        pt = observation['body_pos_rot']['pelvis'][1]
        target_vx = observation['target_vel'][0]
        target_vz = observation['target_vel'][2]
        pe += 20 * (1 - (np.cos(pt)*target_vx - np.sin(pt)*target_vz) / np.sqrt(target_vx**2 + target_vz**2))

        # do NOT jump
        pe += 10 * max(.0, min(observation['body_pos']['pros_foot_r'][1], observation['body_pos']['calcn_l'][1], observation['body_pos']['toes_l'][1]))
        
        done = observation['body_pos']['pelvis'][1] <= 0.65

        return pe, done

    def _engineer_features(self, obs):
        vectors = list()

        # target velocity
        vectors.append((obs["target_vel"][0], obs["target_vel"][2]))

        # current velocity
        vectors.append((obs["body_vel"]["pelvis"][0], obs["body_vel"]["pelvis"][2]))

        # moving averaged velocity
        vectors.append(np.mean(list(self.frames), axis=0))

        # pelvis acceleration
        vectors.append((obs['body_acc']['pelvis'][0] / 100.0, obs['body_acc']['pelvis'][0] / 100.0))

        # pelvis orientation as unit vector
        pelvis_pos_x, pelvis_pos_z = obs["body_pos"]["pelvis"][0], obs["body_pos"]["pelvis"][2]
        pelvis_orientation = obs['body_pos_rot']['pelvis'][1]
        vectors.append((np.cos(pelvis_orientation), -np.sin(pelvis_orientation)))
        
        # right foot position w.r.t. pelvis position
        vectors.append((obs["body_pos"]["pros_foot_r"][0]-pelvis_pos_x, obs["body_pos"]["pros_foot_r"][2]-pelvis_pos_z))

        # left foot position w.r.t. pelvis position
        vectors.append((0.5*(obs["body_pos"]["calcn_l"][0]+obs["body_pos"]["toes_l"][0])-pelvis_pos_x, 0.5*(obs["body_pos"]["calcn_l"][2]+obs["body_pos"]["toes_l"][2])-pelvis_pos_z))

        features = list()
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                features.append(vectors[i][0]*vectors[j][0]+vectors[i][1]*vectors[j][1])
        return features
        
    def _relative_dict_to_list(self, observation):
        if self._use_hcf:
            res = self._engineer_features(observation)
        else:
            res = []

        pelvs = {
            'body_pos': observation['body_pos']['pelvis'],
            'body_vel': observation['body_vel']['pelvis'],
            #'body_acc': list(map(lambda v: v/100.0, observation['body_acc']['pelvis']))
        }

        res += [observation["target_vel"][0], pelvs['body_pos'][1], observation["target_vel"][2]]
        res += pelvs['body_vel']
        #res += pelvs['body_acc']

        # Body Observations
        for info_type in ['body_pos', 'body_vel']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += list(map(operator.sub, observation[info_type][body_part], pelvs[info_type]))

        #for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
        #                  'femur_l', 'femur_r', 'head',
        #                  'torso', 'pros_foot_r', 'pros_tibia_r']:
        #    res += list(map(lambda a,b: a/100.0-b, observation['body_acc'][body_part], pelvs['body_acc']))

        for info_type in ['body_pos_rot', 'body_vel_rot']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head', 'pelvis',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += observation[info_type][body_part]
                #if body_part == "pelvis":
                #    print(observation[info_type][body_part])
        #print("***********************************************************************************")
        #for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
        #                  'femur_l', 'femur_r', 'head', 'pelvis',
        #                  'torso', 'pros_foot_r', 'pros_tibia_r']:
        #    res += list(map(lambda v: v/1000.0, observation['body_acc_rot'][body_part]))

        # ground_pelvis
        res += list(map(operator.sub, observation['joint_pos']['ground_pelvis'][0:3], pelvs['body_pos']))
        res += observation['joint_pos']['ground_pelvis'][3:6]
        res += list(map(operator.sub, observation['joint_vel']['ground_pelvis'][0:3], pelvs['body_vel']))
        res += observation['joint_vel']['ground_pelvis'][3:6]
        #res += list(map(lambda a,b: a/100.0-b, observation['joint_acc']['ground_pelvis'][0:3], pelvs['body_acc']))
        #res += list(map(lambda v: v/1000.0, observation['joint_acc']['ground_pelvis'][3:6]))

        # joint
        for info_type in ['joint_pos', 'joint_vel']:
            for joint in ['ankle_l', 'ankle_r', 'back',
                          'hip_l', 'hip_r', 'knee_l', 'knee_r']:
                res += observation[info_type][joint]

        #for joint in ['ankle_l', 'ankle_r', 'back',
        #              'hip_l', 'hip_r', 'knee_l', 'knee_r']:
        #    res += list(map(lambda v: v/1000.0, observation['joint_acc'][joint]))

        # Muscle Observations
        for muscle in ['abd_l', 'abd_r', 'add_l', 'add_r',
                       'bifemsh_l', 'bifemsh_r', 'gastroc_l',
                       'glut_max_l', 'glut_max_r',
                       'hamstrings_l', 'hamstrings_r',
                       'iliopsoas_l', 'iliopsoas_r', 'rect_fem_l', 'rect_fem_r',
                       'soleus_l', 'tib_ant_l', 'vasti_l', 'vasti_r']:
            res.append(observation['muscles'][muscle]['activation'])
            #res.append(observation['muscles'][muscle]['fiber_force']/5000.0)
            res.append(observation['muscles'][muscle]['fiber_length'])
            res.append(observation['muscles'][muscle]['fiber_velocity'])

        return res
    
    def step(self, ac):
        total_reward = .0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(ac, False)
            if self._use_hcf:
                self.frames.append([obs["body_vel"]["pelvis"][0], obs["body_vel"]["pelvis"][2]])
            penalty, strong_done = self._penalty(obs)
            done = done if done else strong_done
            total_reward += (reward if done else reward+1.0) - penalty
            if done:
                break
        self.timestep_feature += 1

        return self._relative_dict_to_list(obs)+[float(self.timestep_feature)/100.0], total_reward, done, info

    def reset(self, **kwargs):
        ob = self.env.reset(project=False, **kwargs)
        self.timestep_feature = 0
        if self._use_hcf:
            for _ in range(self._skip -1):
                self.frames.append(np.zeros(2, dtype="float32"))
            self.frames.append([ob["body_vel"]["pelvis"][0], ob["body_vel"]["pelvis"][2]])
        return self._relative_dict_to_list(ob) + [float(self.timestep_feature)/100.0]


class Round2CleanEnv(gym.Wrapper):
    def __init__(self, env, skip=3, use_hcf=False):
        """
        add 1 to original reward for each timestep except for the terminal one
        repeat an action for 4 timesteps
        """
        gym.Wrapper.__init__(self, env)
        self._use_hcf = use_hcf
        self.observation_space.shape = (244 if use_hcf else 223,)
        self._skip = skip
        if use_hcf:
            self.frames = deque([], maxlen=self._skip)

    def _penalty(self, observation):
        x_head_pelvis = observation['body_pos']['head'][0]-observation['body_pos']['pelvis'][0]

        # height from pelvis to head is around 0.62
        # consider 0.62 * cos(60) first
        # consider 0.62 / sqrt(2) later
        accept_x1 = -0.31
        accept_x2 = 0.31
        if x_head_pelvis < accept_x1:
            pe = .667
            #pe = 5.0
            #done = True
        elif x_head_pelvis < accept_x2:
            pe = 0.0
            #done = False
        else:
            pe = 0.667
            #done = False

        z_head_pelvis = observation['body_pos']['head'][2]-observation['body_pos']['pelvis'][2]
        accept_z1 = -0.31
        accept_z2 = 0.31
        if z_head_pelvis < accept_z1:
            pe += 0.667
            #pe += 5.0
            #done = True
        elif z_head_pelvis < accept_z2:
            pass
        else:
            pe += 0.667
            #pe += 5.0
            #done = True

        # distance between left and right foot
        distance = (observation["body_pos"]["pros_foot_r"][0] - observation["body_pos"]["calcn_l"][0])**2 + (observation["body_pos"]["pros_foot_r"][2] - observation["body_pos"]["calcn_l"][2])**2
        if distance > 0.5:
            pe += 5.0 * (distance - 0.5)

        # cross leg
        theta = observation["body_pos_rot"]["pelvis"][1]
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        pelvis_pos_x, pelvis_pos_z = observation['body_pos']['pelvis'][0], observation['body_pos']['pelvis'][2]
        r_foot_x, r_foot_z =  observation['body_pos']['pros_foot_r'][0]-pelvis_pos_x, observation['body_pos']['pros_foot_r'][2]-pelvis_pos_z
        ip_r = r_foot_x * sin_theta + r_foot_z * cos_theta
        cross_leg_pe_r = max(.0-ip_r, .0)
        l_foot_x, l_foot_z =  observation['body_pos']['calcn_l'][0]-pelvis_pos_x, observation['body_pos']['calcn_l'][2]-pelvis_pos_z
        ip_l = l_foot_x * sin_theta + l_foot_z * cos_theta
        cross_leg_pe_l = max(ip_l-.0, .0)
        pe += 8 * (cross_leg_pe_r + cross_leg_pe_l)

        # heading towards target velocity
        pt = observation['body_pos_rot']['pelvis'][1]
        target_vx = observation['target_vel'][0]
        target_vz = observation['target_vel'][2]
        pe += 20 * (1 - (np.cos(pt)*target_vx - np.sin(pt)*target_vz) / np.sqrt(target_vx**2 + target_vz**2))

        # do NOT jump
        pe += 10 * max(.0, min(observation['body_pos']['pros_foot_r'][1], observation['body_pos']['calcn_l'][1], observation['body_pos']['toes_l'][1]))
        
        done = observation['body_pos']['pelvis'][1] <= 0.65

        return pe, done

    def _engineer_features(self, obs):
        vectors = list()

        # target velocity
        vectors.append((obs["target_vel"][0], obs["target_vel"][2]))

        # current velocity
        vectors.append((obs["body_vel"]["pelvis"][0], obs["body_vel"]["pelvis"][2]))

        # moving averaged velocity
        vectors.append(np.mean(list(self.frames), axis=0))

        # pelvis acceleration
        vectors.append((obs['body_acc']['pelvis'][0] / 100.0, obs['body_acc']['pelvis'][0] / 100.0))

        # pelvis orientation as unit vector
        pelvis_pos_x, pelvis_pos_z = obs["body_pos"]["pelvis"][0], obs["body_pos"]["pelvis"][2]
        pelvis_orientation = obs['body_pos_rot']['pelvis'][1]
        vectors.append((np.cos(pelvis_orientation), -np.sin(pelvis_orientation)))
        
        # right foot position w.r.t. pelvis position
        vectors.append((obs["body_pos"]["pros_foot_r"][0]-pelvis_pos_x, obs["body_pos"]["pros_foot_r"][2]-pelvis_pos_z))

        # left foot position w.r.t. pelvis position
        vectors.append((0.5*(obs["body_pos"]["calcn_l"][0]+obs["body_pos"]["toes_l"][0])-pelvis_pos_x, 0.5*(obs["body_pos"]["calcn_l"][2]+obs["body_pos"]["toes_l"][2])-pelvis_pos_z))

        features = list()
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                features.append(vectors[i][0]*vectors[j][0]+vectors[i][1]*vectors[j][1])
        return features
        
    def _relative_dict_to_list(self, observation):
        if self._use_hcf:
            res = self._engineer_features(observation)
        else:
            res = []

        pelvs = {
            'body_pos': observation['body_pos']['pelvis'],
            'body_vel': observation['body_vel']['pelvis'],
            #'body_acc': list(map(lambda v: v/100.0, observation['body_acc']['pelvis']))
        }

        res += [observation["target_vel"][0], pelvs['body_pos'][1], observation["target_vel"][2]]
        res += pelvs['body_vel']
        #res += pelvs['body_acc']

        # Body Observations
        for info_type in ['body_pos', 'body_vel']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += list(map(operator.sub, observation[info_type][body_part], pelvs[info_type]))

        #for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
        #                  'femur_l', 'femur_r', 'head',
        #                  'torso', 'pros_foot_r', 'pros_tibia_r']:
        #    res += list(map(lambda a,b: a/100.0-b, observation['body_acc'][body_part], pelvs['body_acc']))

        for info_type in ['body_pos_rot', 'body_vel_rot']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head', 'pelvis',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += observation[info_type][body_part]
                #if body_part == "pelvis":
                #    print(observation[info_type][body_part])
        #print("***********************************************************************************")
        #for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
        #                  'femur_l', 'femur_r', 'head', 'pelvis',
        #                  'torso', 'pros_foot_r', 'pros_tibia_r']:
        #    res += list(map(lambda v: v/1000.0, observation['body_acc_rot'][body_part]))

        # ground_pelvis
        res += list(map(operator.sub, observation['joint_pos']['ground_pelvis'][0:3], pelvs['body_pos']))
        res += observation['joint_pos']['ground_pelvis'][3:6]
        res += list(map(operator.sub, observation['joint_vel']['ground_pelvis'][0:3], pelvs['body_vel']))
        res += observation['joint_vel']['ground_pelvis'][3:6]
        #res += list(map(lambda a,b: a/100.0-b, observation['joint_acc']['ground_pelvis'][0:3], pelvs['body_acc']))
        #res += list(map(lambda v: v/1000.0, observation['joint_acc']['ground_pelvis'][3:6]))

        # joint
        for info_type in ['joint_pos', 'joint_vel']:
            for joint in ['ankle_l', 'ankle_r', 'back',
                          'hip_l', 'hip_r', 'knee_l', 'knee_r']:
                res += observation[info_type][joint]

        #for joint in ['ankle_l', 'ankle_r', 'back',
        #              'hip_l', 'hip_r', 'knee_l', 'knee_r']:
        #    res += list(map(lambda v: v/1000.0, observation['joint_acc'][joint]))

        # Muscle Observations
        for muscle in ['abd_l', 'abd_r', 'add_l', 'add_r',
                       'bifemsh_l', 'bifemsh_r', 'gastroc_l',
                       'glut_max_l', 'glut_max_r',
                       'hamstrings_l', 'hamstrings_r',
                       'iliopsoas_l', 'iliopsoas_r', 'rect_fem_l', 'rect_fem_r',
                       'soleus_l', 'tib_ant_l', 'vasti_l', 'vasti_r']:
            res.append(observation['muscles'][muscle]['activation'])
            #res.append(observation['muscles'][muscle]['fiber_force']/5000.0)
            res.append(observation['muscles'][muscle]['fiber_length'])
            res.append(observation['muscles'][muscle]['fiber_velocity'])

        return res
    
    def step(self, ac):
        total_reward = .0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(ac, False)
            if self._use_hcf:
                self.frames.append([obs["body_vel"]["pelvis"][0], obs["body_vel"]["pelvis"][2]])
            penalty, strong_done = self._penalty(obs)
            #done = done if done else strong_done
            #total_reward += (reward if done else reward+1.0) - penalty
            total_reward += reward
            if done:
                break

        return self._relative_dict_to_list(obs), total_reward, done, info

    def reset(self, **kwargs):
        ob = self.env.reset(project=False, **kwargs)
        if self._use_hcf:
            for _ in range(self._skip -1):
                self.frames.append(np.zeros(2, dtype="float32"))
            self.frames.append([ob["body_vel"]["pelvis"][0], ob["body_vel"]["pelvis"][2]])
        return self._relative_dict_to_list(ob)


def wrap_round2_opensim(env, skip=3, use_hcf=False, clean=False):
    if clean:
        return Round2CleanEnv(env, skip=skip, use_hcf=use_hcf)
    return Round2WalkingEnv(env, skip=skip, use_hcf=use_hcf)
