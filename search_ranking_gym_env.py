from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gym
from gym.spaces import Box
from gym.envs.registration import EnvSpec
import numpy as np


class SearchRankingEnv(gym.Env):
    """
    environment for search ranking projects
    obeys OpenAI gym inferfaces and supports
    traversing and shuffling on toy dataset
    """
    def __init__(self, config):
        """
        load raw data
        config is a dict
        required keys: input_path
        optional keys: shuffle, num_epochs, num_item_per_page
        """
        self.config = deepcopy(config)
        
        self.num_item_per_page = config.get("num_item_per_page", 10)
        with open(config["input_path"], 'r') as ips:
            self._raw_data = self._read_data(ips)
        if config.get("shuffle", False):
            np.random.shuffle(self._raw_data)
        self._epoch_cnt = 0
        self._cur_idx = 0

    def _read_data(self, ips):
        raise NotImplementedError

    def reset(self):
        return self._get_observation()

    def _get_observation(self, idx):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


class GoodStuffBanditEnv(SearchRankingEnv):
    """
    simulate the re-ranking process
    without state transition
    """
    def __init__(self, config):
        super(GoodStuffBanditEnv, self).__init__(config)

        self.action_space = Box(
            .0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = Box(
            -float("inf"), float("inf"), shape=(5,), dtype=np.float32)

        self.ws = config.get("ws", 3)
        
    def _read_data(self, ips):
        data = list()
        is_first_line = True

        for line in ips:
            if is_first_line:
                is_first_line = False
                continue

            cols = line.strip().split('\t')
            info_list = [v.split(':') for v in cols[-1].split(',')]
            pctr_list = np.array([float(v[2]) for v in info_list])
            novelty_list = np.array([float(v[4]) for v in info_list])
            cate_list = [int(v[5]) for v in info_list]

            pctr_mean = np.mean(pctr_list)
            pctr_stdiv = np.std(pctr_list)
            novelty_mean = np.mean(novelty_list)
            novelty_stdiv = np.std(novelty_list)
            cate_richness = len(set(cate_list)) / float(len(cate_list))

            state = np.array([pctr_mean, pctr_stdiv, novelty_mean, novelty_stdiv, cate_richness]).astype(np.float32)
            data.append((state, info_list))

        return data

    def _get_observation(self):
        return self._raw_data[self._cur_idx][0]

    def step(self, action):
        cate_rwd, click_rwd = self._simulate(self._raw_data[self._cur_idx][1], action)
        info = {"click_rwd": click_rwd}
        self._cur_idx += 1
        if self._cur_idx == len(self._raw_data):
            self._cur_idx = 0
            self._epoch_cnt += 1
            if self._epoch_cnt >= self.config.get("num_epochs", 99999999):
                info["is_end"] = True
            elif self.config.get("shuffle", False):
                np.random.shuffle(self._raw_data)
        next_state = self._get_observation()

        return next_state, cate_rwd+click_rwd, True, info

    def _simulate(self, info_list, act):
        click_list = [int(v[0]) for v in info_list]
        pos_list = [int(v[1]) for v in info_list]
        pctr_list = [float(v[2]) for v in info_list]
        novelty_list = [float(v[4]) for v in info_list]
        cate_list = [int(v[5]) for v in info_list]

        num_displayed_items = min(self.num_item_per_page, len(click_list))
        weight_unit = 2.0 / (num_displayed_items * (num_displayed_items + num_displayed_items + 1))
        # regard scriptor as item identifier in the following code block
        remaining = set(range(len(pos_list)))
        new_ranking = list()
        clicked_distinct_cates = set()
        cate_rwd, click_rwd = .0, .0

        for new_pos in range(len(pos_list)):
            scores = [(i, act[0]*pctr_list[i]+act[1]*self._calc_diversity(cate_list, new_ranking, i)+act[2]*novelty_list[i]) for i in remaining]
            selected = scores[np.argmax([v[1] for v in scores])][0]
            new_ranking.append(selected)
            remaining.discard(selected)
            if click_list[selected] > 0:
                if cate_list[selected] not in clicked_distinct_cates:
                    cate_rwd += weight_unit * (num_displayed_items - new_pos)
                clicked_distinct_cates.add(cate_list[selected])
                click_rwd += weight_unit * (num_displayed_items - new_pos)

            if len(new_ranking) >= self.num_item_per_page:
                break

        return cate_rwd, click_rwd
        
    def _calc_diversity(self, cate_list, prevs, cur):
        if len(prevs) == 0:
            return .0
        cur_cate = cate_list[cur]
        diff_cate_item_cnt = 0
        for i in range(self.ws):
            if i >= len(prevs):
                break
            if cate_list[prevs[-(i+1)]] != cur_cate:
                diff_cate_item_cnt += 1
        return float(diff_cate_item_cnt) / float(min(self.ws, len(prevs)))


class GoodStuffBaselineBanditEnv(GoodStuffBanditEnv):
    """
    include the action generated by the online serving policy
    """
    def _read_data(self, ips):
        data = list()
        is_first_line = True

        for line in ips:
            if is_first_line:
                is_first_line = False
                continue

            cols = line.strip().split('\t')
            info_list = [v.split(':') for v in cols[-1].split(',')]
            pctr_list = np.array([float(v[2]) for v in info_list])
            novelty_list = np.array([float(v[4]) for v in info_list])
            cate_list = [int(v[5]) for v in info_list]

            pctr_mean = np.mean(pctr_list)
            pctr_stdiv = np.std(pctr_list)
            novelty_mean = np.mean(novelty_list)
            novelty_stdiv = np.std(novelty_list)
            cate_richness = len(set(cate_list)) / float(len(cate_list))

            state = np.array([pctr_mean, pctr_stdiv, novelty_mean, novelty_stdiv, cate_richness]).astype(np.float32)

            action = cols[-6].split('^')
            beta = float(action[0])
            gamma = float(action[1])
            action = np.array([1.0-beta-gamma, beta, gamma]).astype("float32")
            state = np.concatenate([state, action])

            data.append((state, info_list))

        return data


class GoodStuffEpisodicEnv(SearchRankingEnv):
    """
    true rl environment simulating user behaviors based on
    session log search_ranking/episodic_data.tsv
    """

    def __init__(self, config):
        super(GoodStuffEpisodicEnv, self).__init__(config)

        self.action_space = Box(
            .0, 1.0, shape=(3,), dtype=np.float32)
        # page number, number of sess clicked, number of sess clicked distinct category
        # pctr mean, pctr stdiv, novelty mean, novelty stdiv, category richness
        self.observation_space = Box(
            -float("inf"), float("inf"), shape=(3+5,), dtype=np.float32)

        # each row in self._raw_data includes an entire episode
        self._in_episode_idx = 0
        # calculate diversity by taking the last ws items of last page into consideration
        # and count clicked distinct item categories in a whole session
        self._last_item_cates = list()
        self._clicked_cates = set()

    def _read_data(self, ips):
        task_index = self.config.get("task_index", -1)
        num_partitions = self.config.get("num_partitions", 1)

        data = list()
        is_first_line = True
        
        for line in ips:
            if is_first_line:
                is_first_line = False
                schema = line.strip().split(',')
                col_names = [col.split(':')[0] for col in schema]
                ws_col_idx = col_names.index("context__sess_window")
                continue

            cols = line.strip().split('\t')
            pv_list = [tuple(v.split('|')) for v in cols[-1].split(';')]
            sorted(pv_list, key=lambda x: x[0])
            ws_list = [tuple(v.split('|')) for v in cols[ws_col_idx].split(';')]
            sorted(ws_list, key=lambda x: x[0])
            
            sess_states = list()
            sess_infos = list()
            sess_ws = list()
            page_nb = 0
            for pv, ws in zip(pv_list, ws_list):
                info_list = [v.split(':') for v in pv[1].split(',')]
                pctr_list = np.array([float(v[2]) for v in info_list])
                novelty_list = np.array([float(v[4]) for v in info_list])
                cate_list = [int(v[5]) for v in info_list]

                page_nb += 1
                pctr_mean = np.mean(pctr_list)
                pctr_stdiv = np.std(pctr_list)
                novelty_mean = np.mean(novelty_list)
                novelty_stdiv = np.std(novelty_list)
                cate_richness = len(set(cate_list)) / float(len(cate_list))

                state = np.array([page_nb, 0, 0, pctr_mean, pctr_stdiv, novelty_mean, novelty_stdiv, cate_richness]).astype(np.float32)

                sess_states.append(state)
                sess_infos.append(info_list)
                sess_ws.append(int(float(ws[1])))

            data.append((sess_states, sess_infos, sess_ws))

        if task_index != -1:
            data = data[task_index::num_partitions]
        return data

    def _get_observation(self):
        return self._raw_data[self._cur_idx][0][self._in_episode_idx]

    def step(self, action):
        self.ws = 2 + self._raw_data[self._cur_idx][2][self._in_episode_idx]
        cate_rwd, click_rwd = self._simulate(self._raw_data[self._cur_idx][1][self._in_episode_idx], action)
        info = {"click_rwd": click_rwd}
        self._in_episode_idx += 1
        if self._in_episode_idx == len(self._raw_data[self._cur_idx][1]):
            done = True
            self._last_item_cates = list()
            self._clicked_cates = set()
            self._in_episode_idx = 0
            self._cur_idx += 1
        else:
            # specify the next state
            self._raw_data[self._cur_idx][0][self._in_episode_idx][1] = self._raw_data[self._cur_idx][0][self._in_episode_idx-1][1] + click_rwd
            self._raw_data[self._cur_idx][0][self._in_episode_idx][2] = self._raw_data[self._cur_idx][0][self._in_episode_idx-1][2] + cate_rwd
            done = False

        if self._cur_idx == len(self._raw_data):
            self._cur_idx = 0
            self._epoch_cnt += 1
            if self._epoch_cnt >= self.config.get("num_epochs", 99999999):
                info["is_end"] = True
            elif self.config.get("shuffle", False):
                np.random.shuffle(self._raw_data)
        next_state = self._get_observation()

        return next_state, cate_rwd+click_rwd, done, info

    def _simulate(self, info_list, act):
        click_list = [int(v[0]) for v in info_list]
        pos_list = [int(v[1]) for v in info_list]
        pctr_list = [float(v[2]) for v in info_list]
        novelty_list = [float(v[4]) for v in info_list]
        cate_list = [int(v[5]) for v in info_list]

        num_displayed_items = min(self.num_item_per_page, len(click_list))
        weight_unit = 2.0 / (num_displayed_items * (num_displayed_items + num_displayed_items + 1))
        # regard scriptor as item identifier in the following code block
        remaining = set(range(len(pos_list)))
        new_ranking = list()
        #clicked_distinct_cates = set()
        cate_rwd, click_rwd = .0, .0

        for new_pos in range(len(pos_list)):
            scores = [(i, act[0]*pctr_list[i]+act[1]*self._calc_diversity(cate_list[i])+act[2]*novelty_list[i]) for i in remaining]
            selected = scores[np.argmax([v[1] for v in scores])][0]
            new_ranking.append(selected)
            remaining.discard(selected)
            self._last_item_cates.append(cate_list[selected])

            if click_list[selected] > 0:
                if cate_list[selected] not in self._clicked_cates:#clicked_distinct_cates:
                    cate_rwd += 1.0
                    #cate_rwd += weight_unit * (num_displayed_items - new_pos)
                self._clicked_cates.add(cate_list[selected])
                #clicked_distinct_cates.add(cate_list[selected])
                click_rwd += weight_unit * (num_displayed_items - new_pos)

            if len(new_ranking) >= self.num_item_per_page:
                break

        return cate_rwd, click_rwd
        
    def _calc_diversity(self, cur_cate):
        if len(self._last_item_cates) == 0:
            return .0

        diff_cate_item_cnt = 0
        for i in range(self.ws):
            if i >= len(self._last_item_cates):
                break
            if self._last_item_cates[-(i+1)] != cur_cate:
                diff_cate_item_cnt += 1
        return float(diff_cate_item_cnt) / float(min(self.ws, len(self._last_item_cates)))


class GoodStuffBaselineEpisodicEnv(GoodStuffEpisodicEnv):
    """
    include the action generated by the online serving policy
    """
    def _read_data(self, ips):
        data = list()
        is_first_line = True
        
        for line in ips:
            if is_first_line:
                is_first_line = False
                schema = line.strip().split(',')
                col_names = [col.split(':')[0] for col in schema]
                ws_col_idx = col_names.index("context__sess_window")
                act_col_idx = col_names.index("action")
                continue

            cols = line.strip().split('\t')
            actions = [tuple(v.split('|')) for v in cols[act_col_idx].split(';')]
            sorted(actions, key=lambda x: x[0])
            pv_list = [tuple(v.split('|')) for v in cols[-1].split(';')]
            sorted(pv_list, key=lambda x: x[0])
            ws_list = [tuple(v.split('|')) for v in cols[ws_col_idx].split(';')]
            sorted(ws_list, key=lambda x: x[0])
            
            sess_states = list()
            sess_infos = list()
            sess_ws = list()
            for act, pv, ws in zip(actions, pv_list, ws_list):
                info_list = [v.split(':') for v in pv[1].split(',')]
                pctr_list = np.array([float(v[2]) for v in info_list])
                novelty_list = np.array([float(v[4]) for v in info_list])
                cate_list = [int(v[5]) for v in info_list]

                pctr_mean = np.mean(pctr_list)
                pctr_stdiv = np.std(pctr_list)
                novelty_mean = np.mean(novelty_list)
                novelty_stdiv = np.std(novelty_list)
                cate_richness = len(set(cate_list)) / float(len(cate_list))

                action = act[1].split('^')
                beta = float(action[0])
                gamma = float(action[1])
                action = np.array([1.0-beta-gamma, beta, gamma]).astype("float32")
                state = np.array([pctr_mean, pctr_stdiv, novelty_mean, novelty_stdiv, cate_richness]).astype(np.float32)
                state = np.concatenate([state, action])

                sess_states.append(state)
                sess_infos.append(info_list)
                sess_ws.append(int(float(ws[1])))

            data.append((sess_states, sess_infos, sess_ws))

        return data
