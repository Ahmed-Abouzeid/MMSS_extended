from argparse import ArgumentParser
from model_predictor import Hawkes
from tracker import Tracker
import math

import pickle

from utils import get_network_metadata, get_categorized_timestamps, shrink_network, create_synthetic_network,\
    get_pegypt_metadata, get_categorized_timestamps_bias, get_categorized_timestamps_propaganda,\
    get_categorized_timestamps_circles, calc_final_pegypt_results, format_hawkes_stamps, get_pegypt_sample_metadata, \
    create_adjacency_matrix_pegypt, calc_avg_abs_err, calc_avg_num_events_per_user,\
    plot_simulation_performance_non_timestamps, plot_user_event_dist
import pickle
from demo import run_experiments, run_avg_method, run_pegypt_experiments
import time
import numpy as np
import collections

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--retweet_path', type=str, default="data/raw/covid19/tree")
    parser.add_argument('--labels_path', type=str, default="data/raw/pegypt")
    parser.add_argument('--realizations_n_A', type=int, default=32)
    parser.add_argument('--realizations_n_B', type=int, default=96)
    parser.add_argument('--realizations_n_C', type=int, default=48)
    parser.add_argument('--realizations_bounds_A', type=object, default = [x * 180 for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]])
    parser.add_argument('--realizations_bounds_B', type=object, default = [x * 60 for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                                                                             33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                                                                                            49, 50, 51, 52, 53, 54, 55, 56, 57,
                                                                                            58, 59, 60, 61, 62, 63, 64,
                                                                                            65, 66, 67, 68, 69, 70, 71,
                                                                                            72, 73, 74, 75, 76, 77, 78,
                                                                                            79, 80,
                                                                                            81, 82, 83, 84, 85, 86, 87,
                                                                                            88, 89, 90, 91, 92, 93, 94,
                                                                                            95, 96
                                                                                            ]])
    parser.add_argument('--realizations_bounds_C', type=object, default = [x * 120 for x in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                                                                             33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]])
    parser.add_argument('--decay_factor_type1', type=float, default=.7)
    parser.add_argument('--decay_factor_type2', type=float, default=.6)
    parser.add_argument('--decay_factor_type3', type=float, default=.9)
    parser.add_argument('--decay_factor_type4', type=float, default=.6)
    parser.add_argument('--decay_factor_type5', type=float, default=.9)
    parser.add_argument('--decay_factor_type6', type=float, default=.9)
    parser.add_argument('--decay_factor_type7', type=float, default=.75)
    parser.add_argument('--decay_factor_type8', type=float, default=.9)
    parser.add_argument('--decay_factor_type9', type=float, default=.8)
    parser.add_argument('--decay_factor_type10', type=float, default=.8)
    parser.add_argument('--decay_factor_type11', type=float, default=.8)
    parser.add_argument('--pred_realizations_period_A', type=int, default=180)  # we predict next n hours, in minutes
    parser.add_argument('--pred_start_time_A', type=int, default=5580)
    parser.add_argument('--shrinked_network_size', type=int, default=1000)
    parser.add_argument('--real_network_end_time_A', type=int, default=5580)
    parser.add_argument('--pred_realizations_period_B', type=int, default=60)  # we predict next n hours, in minutes
    parser.add_argument('--pred_start_time_B', type=int, default=5700)
    parser.add_argument('--real_network_end_time_B', type=int, default=5700)
    parser.add_argument('--pred_realizations_period_C', type=int, default=120)  # we predict next n hours, in minutes
    parser.add_argument('--pred_start_time_C', type=int, default=5640)
    parser.add_argument('--real_network_end_time_C', type=int, default=5640)
    parser.add_argument('--average_out_runs', type=int, default=1)
    parser.add_argument('--budget', type=float, default=2)
    parser.add_argument('--step', type=float, default=0.000001)
    parser.add_argument('--balance_factor', type=float, default=2)
    parser.add_argument('--threshold', type=float, default=.8)
    parser.add_argument('--la_m_depth', type=int, default=500)
    parser.add_argument('--epsilon', type=float, default=0)
    parser.add_argument('--network_type', type=str, default='pegypt')
    parser.add_argument('--synthetic_skewness', type=float, default=.07)
    parser.add_argument('--verbose', type=bool, default = False)
    parser.add_argument('--parallel', type=bool, default = False)
    parser.add_argument('--is_avg_method', type=bool, default = False)
    parser.add_argument('--graphs', type=bool, default = False)
    parser.add_argument('--sample_size', type=int, default = 100)
    parser.add_argument('--iterations', type=int, default = 100)
    parser.add_argument('--adjacents_sample_size', type=int, default = None)
    parser.add_argument('--model', type=int, default=0)

    config = parser.parse_args()

    all_users_mapping_org, sub_network_user_ids_org, data_org = get_pegypt_metadata(config.labels_path,
                                                                        config.shrinked_network_size)

    bias_for_timestamps_org, neutral_bias_timestamps_org, bias_against_timestamps_org = get_categorized_timestamps_bias(data_org,
                                                                                                            all_users_mapping_org,
                                                                                                            sub_network_user_ids_org)

    propaganda_timestamps_org, non_propaganda_timestamps_org = get_categorized_timestamps_propaganda(data_org, all_users_mapping_org,
                                                                                             sub_network_user_ids_org)


    users_biases_freq_after_1 = dict()
    users_circles_acceptances_freq_after_1 = dict()
    users_biases_freq_after_2 = dict()
    users_circles_acceptances_freq_after_2 = dict()

    i_1 = open("users_incentives_old.txt", "r")
    i_2 = open("users_incentives_2_old.txt", "r")

    results_1 = i_1.readlines()
    results_2 = i_2.readlines()

    i_1.close()
    i_2.close()



    #-----------------------------2---------------------------

    no_keys_non_propaganda_timestamps = format_hawkes_stamps(non_propaganda_timestamps_org)
    hawkes_non_propaganda = Hawkes(no_keys_non_propaganda_timestamps, config.decay_factor_type3, config, event_category='A')
    non_propaganda_MU, non_propaganda_A = hawkes_non_propaganda.estimate_params(config.average_out_runs)

    non_propaganda_MU_up_1 = non_propaganda_MU.copy()
    non_propaganda_MU_up_2 = non_propaganda_MU.copy()

    for e, r in enumerate(results_1):
        print("non propaganda 1:", e)
        non_propaganda_MU_up_1[e] += float(r.split(",")[1].strip("\n"))

    for ee, rr in enumerate(results_2):
        print("non propaganda 2:", ee)
        non_propaganda_MU_up_2[ee] += float(rr.split(",")[1].strip("\n"))

    pred_non_propaganda_timestamps_org_1 = hawkes_non_propaganda.simulate(non_propaganda_MU_up_1, non_propaganda_A, None, verbose=False, sampling=False)
    pred_non_propaganda_timestamps_org_2 = hawkes_non_propaganda.simulate(non_propaganda_MU_up_2, non_propaganda_A, None, verbose=False, sampling=False)

    #--------------------------------------------------------



    #-----------------------------3---------------------------


    no_keys_bias_against_timestamps = format_hawkes_stamps(bias_against_timestamps_org)
    hawkes_bias_against = Hawkes(no_keys_bias_against_timestamps, config.decay_factor_type3, config, event_category='A')
    bias_against_MU, bias_against_A = hawkes_bias_against.estimate_params(config.average_out_runs)
    bias_against_MU_up_1 = bias_against_MU.copy()
    bias_against_MU_up_2 = bias_against_MU.copy()

    for e, r in enumerate(results_1):
        print("bias against 1:", e)
        bias_against_MU_up_1[e] += float(r.split(",")[1].strip("\n"))

    for ee, rr in enumerate(results_2):
        print("bias against 2:", ee)
        bias_against_MU_up_2[ee] += float(rr.split(",")[1].strip("\n"))

    pred_bias_against_timestamps_org_1 = hawkes_bias_against.simulate(bias_against_MU_up_1, bias_against_A, None, verbose=False, sampling=False)
    pred_bias_against_timestamps_org_2 = hawkes_bias_against.simulate(bias_against_MU_up_2, bias_against_A, None, verbose=False, sampling=False)

    #--------------------------------------------------------




    for k, v in pred_bias_against_timestamps_org_1.items():



        c1 , c2 = len(v), len(pred_non_propaganda_timestamps_org_1[k])
        users_biases_freq_after_1.update({k: c1})
        if c1 > 0 and c2 > 0:
            users_circles_acceptances_freq_after_1.update({k: np.mean([c1, c2])})
        else:
            users_circles_acceptances_freq_after_1.update({k: 0})



    for k, v in pred_bias_against_timestamps_org_2.items():
        c1, c2 = len(v), len(pred_non_propaganda_timestamps_org_2[k])
        users_biases_freq_after_2.update({k: c1})
        if c1 > 0 and c2 > 0:
            users_circles_acceptances_freq_after_2.update({k: np.mean([c1, c2])})
        else:
            users_circles_acceptances_freq_after_2.update({k: 0})


    print(users_biases_freq_after_1)
    print(users_biases_freq_after_2)
    print(users_circles_acceptances_freq_after_1)
    print(users_circles_acceptances_freq_after_2)

    f1 = open("bias_after_1.pkl", "wb")
    f2 = open("bias_after_2.pkl", "wb")
    f3 = open("circle_after_1.pkl", "wb")
    f4 = open("circle_after_2.pkl", "wb")

    pickle.dump(users_biases_freq_after_1, f1)
    pickle.dump(users_biases_freq_after_2, f2)
    pickle.dump(users_circles_acceptances_freq_after_1, f3)
    pickle.dump(users_circles_acceptances_freq_after_2, f4)