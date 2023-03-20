from utils import get_user_adjacencies, calc_user_campaign_exposure, chunk_timestamps, match_mapped_ids_2
import numpy as np
import math
import random
from scipy import stats
from termcolor import colored
from model_predictor import Hawkes


class SocialNetwork(object):
    """this class acts as an environment for a controller model (LA), the environment in this case is a social network
    where default settings and initial states are stored and also new updated states of the network are tracked. The
    environment class object is also responsible for sending feedback to the controller model when interacting with
    the network. Feedback are determined according to measuring new states compared to older states of the network
    and optimizing a noisy/ non-stationary objective function that tries to achieve fair misinformation mitigation over
     the network"""

    def __init__(self, mis_timestamps, norm_timestamps, adjacency_matrix, realization_id, realization_bounds,
                 config, norm_MU, norm_A, normal_timestamps_before_simu, mapping_info = None, probs_tracker= None, bias_against_MU = None,
                 bias_against_A= None, hawkes_bias_against = None, B_MU = None, B_A = None, hawkes_B = None):

        self.adjacency_matrix = adjacency_matrix
        self.mis_timestamps = mis_timestamps
        self.norm_timestamps = norm_timestamps
        self.realization_id = realization_id
        self.realization_bounds = realization_bounds
        self.norm_MU = norm_MU
        self.norm_A = norm_A
        self.realization_period = config.pred_realizations_period_A
        self.norm_decay_factor = config.decay_factor_type4
        self.budget = config.budget
        self.config = config
        self.normal_timestamps_before_simu = normal_timestamps_before_simu
        self.edges = self.initialize_adjacency()

        self.mapping_info = mapping_info
        self.probs_tracker = probs_tracker
        self.bias_against_MU = bias_against_MU
        self.bias_against_A = bias_against_A
        self.hawkes_bias_against = hawkes_bias_against
        self.B_MU = B_MU
        self.B_A = B_A
        self.hawkes_B = hawkes_B
        self.probs_bias_temporal = dict()
        self.probs_circles_temporal = dict()


    def initialize_adjacency(self):
        '''initializes  dictionary with each user id and all its adjacent users'''
        result =  get_user_adjacencies(self.adjacency_matrix)

        return result

    def get_user_adjacency(self, user_id):
        """return the given user id adjacent user(s)"""
        return self.edges[user_id]

    def default_user_exposure_mis(self, user_id):
        """calculate default user exposures to misinformation"""

        timestamps_chunks = chunk_timestamps(self.mis_timestamps, self.realization_bounds)
        adjacent_users = self.get_user_adjacency(user_id)
        user_misinformation_exposure = calc_user_campaign_exposure(adjacent_users, timestamps_chunks,
                                    None, self.realization_id)
        return user_misinformation_exposure

    def default_user_exposure_norm(self, user_id):
        """calculate default user exposures to normal content"""

        timestamps_chunks = chunk_timestamps(self.norm_timestamps, self.realization_bounds)
        adjacent_users = self.get_user_adjacency(user_id)
        user_normal_content_exposure = calc_user_campaign_exposure(adjacent_users, timestamps_chunks,
                                    None, self.realization_id)
        return user_normal_content_exposure

    def get_user_exposure_norm(self, user_id, norm_timestamps):
        """calculate current user exposures to normal content"""

        timestamps_chunks = chunk_timestamps(norm_timestamps, self.realization_bounds)
        adjacent_users = self.get_user_adjacency(user_id)
        user_normal_content_exposure = calc_user_campaign_exposure(adjacent_users, timestamps_chunks,
                                    None, self.realization_id)
        return user_normal_content_exposure

    def evaluate(self, u_id, norm_timestamps, prev_v, prev_state, current_state, move):
        """an evaluation function over the default states of the social network and the new state after applying
        some changes that led to different events propagation of normal content. The function is used as
        a signal from a social network environment to an external controller model (LA)"""

        adjacent_users_ids = self.get_user_adjacency(u_id)
        user_values = []  # a list of the current user adjacent user(s) ratio(s) t/f
        for id in adjacent_users_ids:
            if id != u_id:
                f = self.default_user_exposure_mis(id)
                t = self.get_user_exposure_norm(id, norm_timestamps)
                user_values.append((0.0001+t)/(0.0001+f*self.config.balance_factor))

        obj_func_v = self.get_obj_func_value(user_values)
        slop = self.calc_sub_slops(obj_func_v, prev_v, current_state, prev_state)

        if slop <= 0 and move == 'r':
            return 0, obj_func_v
        elif slop >= 0 and move == 'l':
            return 0, obj_func_v
        elif slop >= 0 and move == 'r':
            return 1, obj_func_v
        if slop <= 0 and move == 'l':
            return 1, obj_func_v

    def evaluate_2(self, u_id, norm_timestamps, prev_v, prev_state, current_state, move,
                   pred_bias_against_timestamps, pred_B_timestamps):
        """an evaluation function over the default states of the social network and the new state after applying
        some changes that led to different events propagation of normal content. The function is used as
        a signal from a social network environment to an external controller model (LA)"""

        c_bias = len(pred_bias_against_timestamps[u_id])

        org_id = match_mapped_ids_2(u_id, self.mapping_info)
        self.probs_tracker.bias_users_probs[org_id][0]+= c_bias
        self.probs_tracker.bias_users_probs[org_id][1]+= c_bias
        if org_id in self.probs_bias_temporal.keys():
            self.probs_bias_temporal[org_id].append(self.probs_tracker.bias_users_probs[org_id][0]/self.probs_tracker.bias_users_probs[org_id][1])
        else:
            self.probs_bias_temporal.update({org_id: [self.probs_tracker.bias_users_probs[org_id][0]/self.probs_tracker.bias_users_probs[org_id][1]]})
        main_user_bias_agreeing_prob = 0
        if self.probs_tracker.bias_users_probs[org_id][1] != 0:
            main_user_bias_agreeing_prob = self.probs_tracker.bias_users_probs[org_id][0]/ self.probs_tracker.bias_users_probs[org_id][1]

        adjacent_users_ids = self.get_user_adjacency(u_id)
        user_values = []  # a list of the current user adjacent user(s) ratio(s) t/f
        for id in adjacent_users_ids:
            if id != u_id:
                org_id = match_mapped_ids_2(id, self.mapping_info)
                c_bias = len(pred_bias_against_timestamps[id])

                self.probs_tracker.bias_users_probs[org_id][0] += c_bias
                self.probs_tracker.bias_users_probs[org_id][1] += c_bias
                if org_id in self.probs_bias_temporal.keys():
                    self.probs_bias_temporal[org_id].append(
                        self.probs_tracker.bias_users_probs[org_id][0] / self.probs_tracker.bias_users_probs[org_id][1])
                else:
                    self.probs_bias_temporal.update({org_id: [self.probs_tracker.bias_users_probs[org_id][0] /
                                                             self.probs_tracker.bias_users_probs[org_id][1]]})

                bias_agreeing_prob = 0
                if self.probs_tracker.bias_users_probs[org_id][1] != 0:
                    bias_agreeing_prob = self.probs_tracker.bias_users_probs[org_id][0] / self.probs_tracker.bias_users_probs[org_id][1]

                c_B = len(pred_B_timestamps[id])

                self.probs_tracker.social_circles_users_probs[org_id][0] += c_B
                self.probs_tracker.social_circles_users_probs[org_id][1] += c_B
                if org_id in self.probs_circles_temporal.keys():
                    self.probs_circles_temporal[org_id].append(
                        self.probs_tracker.social_circles_users_probs[org_id][0] / self.probs_tracker.social_circles_users_probs[org_id][1])
                else:
                    self.probs_circles_temporal.update({org_id: [self.probs_tracker.social_circles_users_probs[org_id][0] /
                                                             self.probs_tracker.social_circles_users_probs[org_id][1]]})
                B_prob = 0
                if self.probs_tracker.social_circles_users_probs[org_id][1] != 0:
                    B_prob = self.probs_tracker.social_circles_users_probs[org_id][0] / self.probs_tracker.social_circles_users_probs[org_id][1]

                f = self.default_user_exposure_mis(id)
                t = self.get_user_exposure_norm(id, norm_timestamps)
                user_values.append(((0.0001 + t) / (0.0001 + f * self.config.balance_factor)) + (1 - B_prob * bias_agreeing_prob) )

        obj_func_v = self.get_obj_func_value_2(user_values, main_user_bias_agreeing_prob)
        slop = self.calc_sub_slops(obj_func_v, prev_v, current_state, prev_state)

        if slop <= 0 and move == 'r':
            return 0, obj_func_v
        elif slop >= 0 and move == 'l':
            return 0, obj_func_v
        elif slop >= 0 and move == 'r':
            return 1, obj_func_v
        if slop <= 0 and move == 'l':
            return 1, obj_func_v


    def get_obj_func_value(self, user_ratios):
        """calculates an objective function value based on the underlying ratios of a user. The ratios represent
        how much of normal content/misinformation each adjacent user to that main user is composed to now"""

        obj = []
        for r in user_ratios:
            obj.append((1-r)**2)
        return np.sum(obj)

    def get_obj_func_value_2(self, user_ratios, p):
        """calculates an objective function value based on the underlying ratios of a user. The ratios represent
        how much of normal content/misinformation each adjacent user to that main user is composed to now"""

        obj = []
        for r in user_ratios:
            obj.append((1-r)**2)
        return (1 - p) + np.sum(obj)

    def set_initial_obj_fun_v(self, u_id):
        """set initial ratios of misinformation and normal content for each user on the network, then set the initial
        value of the objective function"""

        adjacent_users_ids = set(self.get_user_adjacency(u_id))
        user_values = []  # a list of the current user adjacent users ratios t/f
        for id in adjacent_users_ids:
            if id != u_id:
                f = self.default_user_exposure_mis(id)
                t = self.default_user_exposure_norm(id)
                user_values.append((0.0001+t) / (0.0001+f*self.config.balance_factor))

        obj_func_v = self.get_obj_func_value(user_values)

        return obj_func_v

    def calc_sub_slops(self, current_obj, prev_obj, current_state, prev_state):
        """here the function calculates the slop between two obtained objective functions"""

        s = stats.linregress([prev_state, current_state], [prev_obj, current_obj])[0]
        if math.isnan(s):
            if prev_obj <= current_obj:
                return -1
            else:
                return 1
        return s

    def trace_probs(self, la):
        """a method to interactively show the current states and moves probs of each LA associated
         with a user on the network"""
        print(colored([la.feedback, la.id, la.state_space.index(la.current_state), la.current_state, la.aimed_move], 'green'))
        print(colored(la.state_transition_matrix, 'green'))
        print(colored(
            [la.state_transition_matrix[la.state_space.index(la.current_state)][la.state_space.index(la.current_state)],
            '+/-'], 'green'))
        if la.current_state != self.budget:
            print(colored([la.state_transition_matrix[la.state_space.index(la.current_state)][
                      la.state_space.index(la.current_state) + 1], '+'], 'green'))
        if la.current_state != 0:
            print(colored([la.state_transition_matrix[la.state_space.index(la.current_state)][
                      la.state_space.index(la.current_state) - 1], '-'], 'green'))


