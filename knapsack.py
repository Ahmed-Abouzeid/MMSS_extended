from mitigation_la import LA
import multiprocessing as mp
import random
from tqdm import tqdm
from termcolor import colored
from model_predictor import Hawkes
from utils import network_converged, merge_timestamps, match_mapped_ids_2

bounds = [x * 7200 for x in [1, 2, 3, 4, 5]]


class Knapsack(object):
    """the class performs a joint random walk using multiple LA models to perform different random walks over
    a shared state of a given budget for a knapsack problem"""

    def __init__(self, user_ids, config, social_network):
        self.users_ids = user_ids
        self.config = config
        self.social_network = social_network

    def random_walk(self, sample_code, loss_function):
        """ the function that performs the multidimensional random walk, the function is designed to support
        parallel computing in future"""
        las = []
        knapsack_capacity = 0
        for id in tqdm(self.users_ids, 'Initializing Automata: Object Creation'):
            la = LA(id, self.config, self.social_network, self.config.la_m_depth)
            la.scores.append(la.initial_value)
            la.state_updates.append(la.current_state)
            la.iters.append(la.i)
            las.append(la)
        controlled_mu = self.social_network.norm_MU.copy()
        if loss_function == 2:
            controlled_bias_against_MU = self.social_network.bias_against_MU.copy()
            controlled_B_MU = self.social_network.B_MU.copy()
        else:
            controlled_bias_against_MU = None
            controlled_B_MU = None
        initial_loop_step = True
        rewards = []

        while not network_converged(las):
            # we set priorities from which users are more in need for the mitigation, so that we scheduel them first
            # for the budget consumption
            if initial_loop_step:
                mhp = Hawkes(None, self.social_network.norm_decay_factor, self.social_network.config)
                pred_norm_time_stamps = mhp.simulate(controlled_mu, self.social_network.norm_A, None, False)

                las_diffs = dict()
                _, merged_norm_timestamps = merge_timestamps({}, {},
                                                             self.social_network.normal_timestamps_before_simu,
                                                             pred_norm_time_stamps)
                exp_defaults = {}
                for la in tqdm(las, 'Calculating Exposures Defaults'):
                    x = self.social_network.default_user_exposure_mis(la.id)
                    y = self.social_network.get_user_exposure_norm(la.id, merged_norm_timestamps)
                    exp_defaults.update({la.id: (x, y)})
                for la in tqdm(las, 'Initializing Automata: Setting Defaults'):
                    user_values = []
                    adjacent_users_ids = self.social_network.get_user_adjacency(la.id)
                    if loss_function == 2:
                        org_id = match_mapped_ids_2(la.id, self.social_network.mapping_info)
                        main_user_bias_agreeing_prob = 0
                        if self.social_network.probs_tracker.bias_users_probs[org_id][1] != 0:
                            main_user_bias_agreeing_prob = self.social_network.probs_tracker.bias_users_probs[org_id][0] / \
                                                           self.social_network.probs_tracker.bias_users_probs[org_id][1]

                    for id in adjacent_users_ids:
                        if id != la.id:
                            if loss_function == 2:
                                org_id = match_mapped_ids_2(id, self.social_network.mapping_info)
                                bias_agreeing_prob = 0
                                if self.social_network.probs_tracker.bias_users_probs[org_id][1] != 0:
                                    bias_agreeing_prob = self.social_network.probs_tracker.bias_users_probs[org_id][0] / \
                                                         self.social_network.probs_tracker.bias_users_probs[org_id][1]

                                B_prob = 0
                                if self.social_network.probs_tracker.social_circles_users_probs[org_id][1] != 0:
                                    B_prob = self.social_network.probs_tracker.social_circles_users_probs[org_id][0] / \
                                             self.social_network.probs_tracker.social_circles_users_probs[org_id][1]

                            x = exp_defaults[id][0]
                            y = exp_defaults[id][1]
                            if loss_function == 1:
                                user_values.append((0.0001 + y) / (0.0001 + x * self.config.balance_factor))
                            elif loss_function == 2:
                                user_values.append(((0.0001 + y) / (0.0001 + x * self.config.balance_factor)) + (1 - B_prob * bias_agreeing_prob))
                    if loss_function == 1:
                        obj_func_v = self.social_network.get_obj_func_value(user_values)
                    elif loss_function == 2:
                        obj_func_v = self.social_network.get_obj_func_value_2(user_values, main_user_bias_agreeing_prob)
                    la.initial_value = obj_func_v
                    la.obj_fun_v = obj_func_v
                    la.scores = [obj_func_v]
                    la.prev_obj = obj_func_v
                    las_diffs.update({la.id: obj_func_v})
                priority_las = sorted(las_diffs.items(), key=lambda x: x[1], reverse=True)
                schedueled_las_ids = []
                for la_id, _ in tqdm(priority_las, 'Arranging Scheduled LA Access'):
                    schedueled_las_ids.append(la_id)

            initial_loop_step = False
            for la_id in schedueled_las_ids:
                if not las[la_id].converged:
                    knapsack_capacity, controlled_mu, pred_norm_time_stamps, controlled_bias_against_MU,\
                    controlled_B_MU, penalty_value = las[la_id].control(knapsack_capacity, controlled_mu, las, pred_norm_time_stamps, loss_function,
                                                         controlled_bias_against_MU, controlled_B_MU)
                    if penalty_value is not None:
                        rewards.append(1-penalty_value)
                    #rewards.append(current_reward)

                    if self.config.verbose:
                        self.social_network.trace_probs(las[la_id])
                else:
                    if self.config.verbose:
                        print(colored([la_id, ' CONVERGED :/)'], 'yellow'))

            with open("learning_trace/loss_" + str(loss_function) + "_" + str(sample_code)+"_total_pen.txt", "a") as f:
                f.write(str(sum(rewards)) + "\n")
        x = 0
        for la in las:
            x += la.current_state

        ks_value = knapsack_capacity
        if self.config.verbose == True:
            print('Knapsack Loaded Amount: ', ks_value)
            print('Automata Sum of States: ', x)
        if round(x, 3) != round(ks_value, 3):
            raise Exception(colored('Control Had Misused Constraint Knapsack Optimization Budget.'
                            ' Fix the Issue Between the Above Diff Values and Rerun Optimization!', 'red'))
        else:
            print("Knapsack Optimization Verified For Loss Function ", loss_function, " , Consumed Budget:", x)
            print("-----------------------------------------------------------")
            print("")

        return las, x