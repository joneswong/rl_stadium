import operator
import numpy as np
import gym
from gym import spaces
#import cv2
#cv2.ocl.setUseOpenCL(False)


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


def wrap_opensim(env, contd=False):
    if contd:
        env = RunningEnv(env)
    else:
        env = WalkingEnv(env)
    return env
