from argparse import ArgumentParser
from model_predictor import Hawkes
from tracker import Tracker
import math
from scipy.stats import entropy

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
    parser.add_argument('--budget', type=float, default=.25)
    parser.add_argument('--step', type=float, default=0.000001)
    parser.add_argument('--balance_factor', type=float, default=2)
    parser.add_argument('--threshold', type=float, default=.8)
    parser.add_argument('--la_m_depth', type=int, default=500)
    parser.add_argument('--epsilon', type=float, default=0)
    parser.add_argument('--network_type', type=str, default='pegypt')
    parser.add_argument('--synthetic_skewness', type=float, default=.07)
    parser.add_argument('--verbose', type=bool, default = True)
    parser.add_argument('--parallel', type=bool, default = False)
    parser.add_argument('--is_avg_method', type=bool, default = False)
    parser.add_argument('--graphs', type=bool, default = False)
    parser.add_argument('--sample_size', type=int, default = 100)
    parser.add_argument('--iterations', type=int, default = 30)
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

    A_timestamps_org, B_timestamps_org, C_timestamps_org, D_timestamps_org, E_timestamps_org, F_timestamps_org = get_categorized_timestamps_circles(
        data_org, all_users_mapping_org, sub_network_user_ids_org)

    no_keys_propaganda_timestamps = format_hawkes_stamps(propaganda_timestamps_org)
    hawkes_propaganda = Hawkes(no_keys_propaganda_timestamps, config.decay_factor_type3, config, event_category='A')
    propaganda_MU, propaganda_A = hawkes_propaganda.estimate_params(config.average_out_runs)
    pred_propaganda_timestamps_org = hawkes_propaganda.simulate(propaganda_MU, propaganda_A, None, verbose=False, sampling=False)


    no_keys_non_propaganda_timestamps_org = format_hawkes_stamps(non_propaganda_timestamps_org)
    hawkes_non_propaganda = Hawkes(no_keys_non_propaganda_timestamps_org, config.decay_factor_type4, config, event_category='A')
    MU, A = hawkes_non_propaganda.estimate_params(config.average_out_runs)

    pred_events_num_per_user, real_events_num_per_user, er1, er2, \
    plot_real_values, plot_pred_values, err1_std, real_nums_Std, pred_nums_std, \
    pred_bias_against_timestamps, pred_bias_for_timestamps = \
        run_pegypt_experiments(None, config, sub_network_user_ids_org, bias_against_timestamps_org, bias_for_timestamps_org, None, None, None, 1, None)

    std1_divided_by_population_size = real_nums_Std[0] / math.sqrt(len(sub_network_user_ids_org))
    std2_divided_by_population_size = pred_nums_std[0] / math.sqrt(len(sub_network_user_ids_org))

    print("")
    print("#######################################################")
    print("Biased Against ERR1: ", er1[0])
    print("Biased Against ERR1 STD: ", err1_std[0])
    print("Biased Against ERR2: ", er2[0])
    print("Biased Against Z Statistic: ", abs(pred_events_num_per_user[0] - real_events_num_per_user[0])/
          (math.sqrt(std1_divided_by_population_size + std2_divided_by_population_size)))

    print("Biased Against AVG Real Events Num Per User: ", real_events_num_per_user[0])
    print("Biased Against AVG Real Events Num Per User STD: ", real_nums_Std[0])
    print("Biased Against AVG Pred Events Num Per User: ", pred_events_num_per_user[0])
    print("Biased Against AVG Pred Events Num Per User STD: ", pred_nums_std[0])

    X = [i for i in range(len(sub_network_user_ids_org))]
    plot_simulation_performance_non_timestamps(X[:100], plot_real_values[0][:100], plot_pred_values[0][:100],
                                               "samples_simulations/biased_against.png")

    print("#######################################################")
    print("")
    print("#######################################################")

    std1_divided_by_population_size = real_nums_Std[1] / math.sqrt(len(sub_network_user_ids_org))
    std2_divided_by_population_size = pred_nums_std[1] / math.sqrt(len(sub_network_user_ids_org))

    print("Biased For ERR1: ", er1[1])
    print("Biased For ERR1 STD: ", err1_std[1])
    print("Biased For ERR2: ", er2[1])
    print("Biased For Z Statistic: ", abs(pred_events_num_per_user[1] - real_events_num_per_user[1]) /
          (math.sqrt(std1_divided_by_population_size + std2_divided_by_population_size)))

    print("Biased For AVG Real Events Num Per User: ", real_events_num_per_user[1])
    print("Biased For AVG Real Events Num Per User STD: ", real_nums_Std[1])
    print("Biased For AVG Pred Events Num Per User: ", pred_events_num_per_user[1])
    print("Biased For AVG Pred Events Num Per User STD: ", pred_nums_std[1])


    X = [i for i in range(len(sub_network_user_ids_org))]
    plot_simulation_performance_non_timestamps(X[:100], plot_real_values[1][:100], plot_pred_values[1][:100],
                                               "samples_simulations/biased_for.png")
    print("#######################################################")


    pred_events_num_per_user, real_events_num_per_user, er1, er2, \
    plot_real_values, plot_pred_values, err1_std, real_nums_Std, pred_nums_std , \
    pred_A_timestamps, pred_B_timestamps, pred_C_timestamps, pred_E_timestamps, pred_F_timestamps= \
        run_pegypt_experiments(None, config, sub_network_user_ids_org, A_timestamps_org, B_timestamps_org,
                               C_timestamps_org, E_timestamps_org, F_timestamps_org, 2, None)

    std1_divided_by_population_size = real_nums_Std[0] / math.sqrt(len(sub_network_user_ids_org))
    std2_divided_by_population_size = pred_nums_std[0] / math.sqrt(len(sub_network_user_ids_org))

    print("")
    print("#######################################################")

    print("Circle A ERR1: ", er1[0])
    print("Circle A ERR1 STD: ", err1_std[0])
    print("Circle A ERR2: ", er2[0])
    print("Circle A Z Statistic: ", abs(pred_events_num_per_user[0] - real_events_num_per_user[0]) /
          (math.sqrt(std1_divided_by_population_size + std2_divided_by_population_size)))

    print("Circle A AVG Real Events Num Per User: ", real_events_num_per_user[0])
    print("Circle A AVG Real Events Num Per User STD: ", real_nums_Std[0])
    print("Circle A AVG Pred Events Num Per User: ", pred_events_num_per_user[0])
    print("Circle A AVG Pred Events Num Per User STD: ", pred_nums_std[0])

    X = [i for i in range(len(sub_network_user_ids_org))]
    plot_simulation_performance_non_timestamps(X[:100], plot_real_values[0][:100], plot_pred_values[0][:100],
                                               "samples_simulations/Circle_A.png")
    print("#######################################################")
    print("")
    std1_divided_by_population_size = real_nums_Std[1] / math.sqrt(len(sub_network_user_ids_org))
    std2_divided_by_population_size = pred_nums_std[1] / math.sqrt(len(sub_network_user_ids_org))
    print("#######################################################")
    print("Circle B ERR1: ", er1[1])
    print("Circle B ERR1 STD: ", err1_std[1])
    print("Circle B ERR2: ", er2[1])
    print("Circle B Z Statistic: ", abs(pred_events_num_per_user[1] - real_events_num_per_user[1]) /
          (math.sqrt(std1_divided_by_population_size + std2_divided_by_population_size)))

    print("Circle B AVG Real Events Num Per User: ", real_events_num_per_user[1])
    print("Circle B AVG Real Events Num Per User STD: ", real_nums_Std[1])
    print("Circle B AVG Pred Events Num Per User: ", pred_events_num_per_user[1])
    print("Circle B AVG Pred Events Num Per User STD: ", pred_nums_std[1])

    X = [i for i in range(len(sub_network_user_ids_org))]
    plot_simulation_performance_non_timestamps(X[:100], plot_real_values[1][:100], plot_pred_values[1][:100],
                                               "samples_simulations/Circle_B.png")
    print("#######################################################")

    std1_divided_by_population_size = real_nums_Std[2] / math.sqrt(len(sub_network_user_ids_org))
    std2_divided_by_population_size = pred_nums_std[2] / math.sqrt(len(sub_network_user_ids_org))
    print("")
    print("#######################################################")
    print("Circle C ERR1: ", er1[2])
    print("Circle C ERR1 STD: ", err1_std[2])
    print("Circle C ERR2: ", er2[2])
    print("Circle C Z Statistic: ", abs(pred_events_num_per_user[2] - real_events_num_per_user[2]) /
          (math.sqrt(std1_divided_by_population_size + std2_divided_by_population_size)))

    print("Circle C AVG Real Events Num Per User: ", real_events_num_per_user[2])
    print("Circle C AVG Real Events Num Per User STD: ", real_nums_Std[2])
    print("Circle C AVG Pred Events Num Per User: ", pred_events_num_per_user[2])
    print("Circle C AVG Pred Events Num Per User STD: ", pred_nums_std[2])

    X = [i for i in range(len(sub_network_user_ids_org))]
    plot_simulation_performance_non_timestamps(X[:100], plot_real_values[2][:100], plot_pred_values[2][:100],
                                               "samples_simulations/Circle_C.png")
    print("#######################################################")
    std1_divided_by_population_size = real_nums_Std[3] / math.sqrt(len(sub_network_user_ids_org))
    std2_divided_by_population_size = pred_nums_std[3] / math.sqrt(len(sub_network_user_ids_org))
    print("")
    print("#######################################################")
    print("Circle E ERR1: ", er1[3])
    print("Circle E ERR1 STD: ", err1_std[3])
    print("Circle E ERR2: ", er2[3])
    print("Circle E Z Statistic: ", abs(pred_events_num_per_user[3] - real_events_num_per_user[3]) /
          (math.sqrt(std1_divided_by_population_size + std2_divided_by_population_size)))

    print("Circle E AVG Real Events Num Per User: ", real_events_num_per_user[3])
    print("Circle E AVG Real Events Num Per User STD: ", real_nums_Std[3])
    print("Circle E AVG Pred Events Num Per User: ", pred_events_num_per_user[3])
    print("Circle E AVG Pred Events Num Per User STD: ", pred_nums_std[3])

    X = [i for i in range(len(sub_network_user_ids_org))]
    plot_simulation_performance_non_timestamps(X[:100], plot_real_values[3][:100], plot_pred_values[3][:100],
                                               "samples_simulations/Circle_E.png")
    print("#######################################################")

    std1_divided_by_population_size = real_nums_Std[4] / math.sqrt(len(sub_network_user_ids_org))
    std2_divided_by_population_size = pred_nums_std[4] / math.sqrt(len(sub_network_user_ids_org))
    print("")
    print("#######################################################")
    print("Circle F ERR1: ", er1[4])
    print("Circle F ERR1 STD: ", err1_std[4])
    print("Circle F ERR2: ", er2[4])
    print("Circle F Z Statistic: ", abs(pred_events_num_per_user[4] - real_events_num_per_user[4]) /
          (math.sqrt(std1_divided_by_population_size + std2_divided_by_population_size)))

    print("Circle F AVG Real Events Num Per User: ", real_events_num_per_user[4])
    print("Circle F AVG Real Events Num Per User STD: ", real_nums_Std[4])
    print("Circle F AVG Pred Events Num Per User: ", pred_events_num_per_user[4])
    print("Circle F AVG Pred Events Num Per User STD: ", pred_nums_std[4])

    X = [i for i in range(len(sub_network_user_ids_org))]
    plot_simulation_performance_non_timestamps(X[:100], plot_real_values[4][:100], plot_pred_values[4][:100],
                                               "samples_simulations/Circle_F.png")
    print("#######################################################")

    print("")
    users_MUs = dict()
    users_MUs_2 = dict()

    propaganda_real_events_nums = []
    propaganda_pred_events_nums = []
    propaganda_errs1 = []
    propaganda_errs2 = []
    propaganda_errs1_std = []
    propaganda_real_events_nums_std = []
    propaganda_pred_events_nums_std = []
    propaganda_plot_real_values = []
    propaganda_plot_pred_values = []

    non_propaganda_real_events_nums = []
    non_propaganda_pred_events_nums = []
    non_propaganda_errs1 = []
    non_propaganda_errs2 = []
    non_propaganda_errs1_std = []
    non_propaganda_real_events_nums_std = []
    non_propaganda_pred_events_nums_std = []
    non_propaganda_plot_real_values = []
    non_propaganda_plot_pred_values = []

    propaganda_samples_Z = []
    non_propaganda_samples_Z = []

    bias_against_real_events_nums = []
    bias_against_pred_events_nums = []
    bias_against_errs1 = []
    bias_against_errs2 = []
    bias_against_errs1_std = []
    bias_against_real_events_nums_std = []
    bias_against_pred_events_nums_std = []
    bias_against_plot_real_values = []
    bias_against_plot_pred_values = []

    B_real_events_nums = []
    B_pred_events_nums = []
    B_errs1 = []
    B_errs2 = []
    B_errs1_std = []
    B_real_events_nums_std = []
    B_pred_events_nums_std = []
    B_plot_real_values = []
    B_plot_pred_values = []

    bias_against_samples_Z = []
    B_samples_Z = []

    for i in sub_network_user_ids_org:
        users_MUs.update({i:[]})

    for i in sub_network_user_ids_org:
        users_MUs_2.update({i:[]})

    to_estimate_MU = np.zeros(len(sub_network_user_ids_org))
    to_estimate_MU_2 = np.zeros(len(sub_network_user_ids_org))

    probs_tracker = Tracker(sub_network_user_ids_org, config.realizations_bounds_A, [bias_against_timestamps_org,
                                                       bias_for_timestamps_org], [A_timestamps_org, B_timestamps_org,
                                                                                  C_timestamps_org,
                                                                                  E_timestamps_org, F_timestamps_org],
                            [pred_bias_for_timestamps, pred_A_timestamps, pred_C_timestamps, pred_E_timestamps, pred_F_timestamps])

    sample_counter = 0
    while sample_counter < config.iterations:
        try:
            print("Sample ", sample_counter)
            all_users_mapping, sub_network_user_ids, data = get_pegypt_sample_metadata(sub_network_user_ids_org, config)
            propaganda_timestamps, non_propaganda_timestamps = get_categorized_timestamps_propaganda(data, all_users_mapping, sub_network_user_ids)
            _, _, bias_against_timestamps_org = get_categorized_timestamps_bias(data, all_users_mapping, sub_network_user_ids)
            _, B_timestamps_org, _, _, _, _ = get_categorized_timestamps_circles(data, all_users_mapping, sub_network_user_ids)

            ai_mu, ai_mu_2, pred_events_num_per_user, real_events_num_per_user , er1, er2,\
            plot_real_values, plot_pred_values, err1_std, real_nums_Std, pred_nums_std = \
                run_pegypt_experiments(sample_counter, config, sub_network_user_ids, propaganda_timestamps, non_propaganda_timestamps,
                                       bias_against_timestamps_org, B_timestamps_org, None, 0, probs_tracker, all_users_mapping)
            propaganda_real_events_nums.append(real_events_num_per_user[0])
            propaganda_pred_events_nums.append(pred_events_num_per_user[0])
            propaganda_errs1.append(er1[0])
            propaganda_errs2.append(er2[0])
            propaganda_errs1_std.append(err1_std[0])
            propaganda_real_events_nums_std.append(real_nums_Std[0])
            propaganda_pred_events_nums_std.append(pred_nums_std[0])
            propaganda_plot_pred_values.append(plot_pred_values[0])
            propaganda_plot_real_values.append(plot_real_values[0])

            non_propaganda_real_events_nums.append(real_events_num_per_user[1])
            non_propaganda_pred_events_nums.append(pred_events_num_per_user[1])
            non_propaganda_errs1.append(er1[1])
            non_propaganda_errs2.append(er2[1])
            non_propaganda_errs1_std.append(err1_std[1])
            non_propaganda_real_events_nums_std.append(real_nums_Std[1])
            non_propaganda_pred_events_nums_std.append(pred_nums_std[1])
            non_propaganda_plot_pred_values.append(plot_pred_values[1])
            non_propaganda_plot_real_values.append(plot_real_values[1])

            std1_divided_by_sample_size = real_nums_Std[0]/math.sqrt(len(sub_network_user_ids))
            std2_divided_by_sample_size = pred_nums_std[0]/math.sqrt(len(sub_network_user_ids))
            propaganda_samples_Z.append(abs(real_events_num_per_user[0] - pred_events_num_per_user[0])
                                        / math.sqrt(abs(std1_divided_by_sample_size + std2_divided_by_sample_size)))

            std1_divided_by_sample_size = real_nums_Std[1] / math.sqrt(len(sub_network_user_ids))
            std2_divided_by_sample_size = pred_nums_std[1] / math.sqrt(len(sub_network_user_ids))
            non_propaganda_samples_Z.append(abs(real_events_num_per_user[1] - pred_events_num_per_user[1])
                                        / math.sqrt(abs(std1_divided_by_sample_size + std2_divided_by_sample_size)))

            bias_against_real_events_nums.append(real_events_num_per_user[2])
            bias_against_pred_events_nums.append(pred_events_num_per_user[2])
            bias_against_errs1.append(er1[2])
            bias_against_errs2.append(er2[2])
            bias_against_errs1_std.append(err1_std[2])
            bias_against_real_events_nums_std.append(real_nums_Std[2])
            bias_against_pred_events_nums_std.append(pred_nums_std[2])
            bias_against_plot_pred_values.append(plot_pred_values[2])
            bias_against_plot_real_values.append(plot_real_values[2])

            B_real_events_nums.append(real_events_num_per_user[3])
            B_pred_events_nums.append(pred_events_num_per_user[3])
            B_errs1.append(er1[3])
            B_errs2.append(er2[3])
            B_errs1_std.append(err1_std[3])
            B_real_events_nums_std.append(real_nums_Std[3])
            B_pred_events_nums_std.append(pred_nums_std[3])
            B_plot_pred_values.append(plot_pred_values[3])
            B_plot_real_values.append(plot_real_values[3])

            std1_divided_by_sample_size = real_nums_Std[2] / math.sqrt(len(sub_network_user_ids))
            std2_divided_by_sample_size = pred_nums_std[2] / math.sqrt(len(sub_network_user_ids))
            bias_against_samples_Z.append(abs(real_events_num_per_user[2] - pred_events_num_per_user[2])
                                        / math.sqrt(abs(std1_divided_by_sample_size + std2_divided_by_sample_size)))

            std1_divided_by_sample_size = real_nums_Std[3] / math.sqrt(len(sub_network_user_ids))
            std2_divided_by_sample_size = pred_nums_std[3] / math.sqrt(len(sub_network_user_ids))
            B_samples_Z.append(abs(real_events_num_per_user[3] - pred_events_num_per_user[3])
                                            / math.sqrt(abs(std1_divided_by_sample_size + std2_divided_by_sample_size)))

            for e, i in enumerate(sub_network_user_ids):
                users_MUs[sub_network_user_ids_org[sub_network_user_ids_org.index(i)]].append(ai_mu[e])


            for e_2, i_2 in enumerate(sub_network_user_ids):
                users_MUs_2[sub_network_user_ids_org[sub_network_user_ids_org.index(i_2)]].append(ai_mu_2[e_2])

            sample_counter += 1

        except:
            print("Error Occurred. Trying Sampling Again...")
            continue
    X = [x for x, _ in enumerate(sub_network_user_ids)]

    for e, reals in enumerate(propaganda_plot_real_values):
        plot_simulation_performance_non_timestamps(X, reals, propaganda_plot_pred_values[e], "samples_simulations/"+str(e)+"_propaganda.png")
        plot_simulation_performance_non_timestamps(X, non_propaganda_plot_real_values[e],
                                                   non_propaganda_plot_pred_values[e], "samples_simulations/"+str(e)+"_non_propaganda.png")

        plot_simulation_performance_non_timestamps(X, bias_against_plot_real_values[e],
                                                   bias_against_plot_pred_values[e],
                                                   "samples_simulations/" + str(e) + "_bias_against.png")


        plot_simulation_performance_non_timestamps(X, B_plot_real_values[e],
                                                   B_plot_pred_values[e],
                                                   "samples_simulations/" + str(e) + "_B.png")

    print("")
    print("---------------------------Simulation Evaluation------------------------------")
    print("")
    print("Number of Samples Used: ", sample_counter)
    print("Sampled Propaganda AVG ERR1: ", np.mean(propaganda_errs1))
    print("Sampled Propaganda AVG ERR1 STD: ", np.mean(propaganda_errs1_std))
    print("Sampled Propaganda AVG ERR2: ", np.mean(propaganda_errs2))
    print("Sampled Propaganda AVG Z Statistic: ", np.mean(propaganda_samples_Z))
    print("Sampled Propaganda AVG Z Statistic STD: ", np.std(propaganda_samples_Z))

    print("Sampled Propaganda AVG Real Events Num Per User: ", np.mean(propaganda_real_events_nums))
    print("Sampled Propaganda AVG Real Events Num Per User STD: ", np.mean(propaganda_real_events_nums_std))
    print("Sampled Propaganda AVG Pred Events Num Per User: ", np.mean(propaganda_pred_events_nums))
    print("Sampled Propaganda AVG Pred Events Num Per User STD: ", np.mean(non_propaganda_pred_events_nums_std))

    print("Sampled Non Propaganda AVG ERR1: ", np.mean(non_propaganda_errs1))
    print("Sampled Non Propaganda AVG ERR1 STD: ", np.mean(non_propaganda_errs1_std))
    print("Sampled Non Propaganda AVG ERR2: ", np.mean(non_propaganda_errs2))
    print("Sampled Non Propaganda AVG Z Statistic: ", np.mean(non_propaganda_samples_Z))
    print("Sampled Non Propaganda AVG Z Statistic STD: ", np.std(non_propaganda_samples_Z))

    print("Sampled Non Propaganda AVG Real Events Num Per User: ", np.mean(non_propaganda_real_events_nums))
    print("Sampled Non Propaganda AVG Real Events Num Per User STD: ", np.mean(non_propaganda_real_events_nums_std))
    print("Sampled Non Propaganda AVG Pred Events Num Per User: ", np.mean(non_propaganda_pred_events_nums))
    print("Sampled Non Propaganda AVG Pred Events Num Per User STD: ", np.mean(non_propaganda_pred_events_nums_std))

    print("Sampled Bias Against AVG ERR1: ", np.mean(bias_against_errs1))
    print("Sampled Bias Against AVG ERR1 STD: ", np.mean(bias_against_errs1_std))
    print("Sampled Bias Against AVG ERR2: ", np.mean(bias_against_errs2))
    print("Sampled Bias Against AVG Z Statistic: ", np.mean(bias_against_samples_Z))
    print("Sampled Bias Against AVG Z Statistic STD: ", np.std(bias_against_samples_Z))

    print("Sampled Bias Against AVG Real Events Num Per User: ", np.mean(bias_against_real_events_nums))
    print("Sampled Bias Against AVG Real Events Num Per User STD: ", np.mean(bias_against_real_events_nums_std))
    print("Sampled Bias Against AVG Pred Events Num Per User: ", np.mean(bias_against_pred_events_nums))
    print("Sampled Bias Against AVG Pred Events Num Per User STD: ", np.mean(bias_against_pred_events_nums_std))

    print("Sampled Circle B AVG ERR1: ", np.mean(B_errs1))
    print("Sampled Circle B AVG ERR1 STD: ", np.mean(B_errs1_std))
    print("Sampled Circle B AVG ERR2: ", np.mean(B_errs2))
    print("Sampled Circle B AVG Z Statistic: ", np.mean(B_samples_Z))
    print("Sampled Circle B AVG Z Statistic STD: ", np.std(B_samples_Z))

    print("Sampled Circle B AVG Real Events Num Per User: ", np.mean(B_real_events_nums))
    print("Sampled Circle B AVG Real Events Num Per User STD: ", np.mean(B_real_events_nums_std))
    print("Sampled Circle B AVG Pred Events Num Per User: ", np.mean(B_pred_events_nums))
    print("Sampled Circle B AVG Pred Events Num Per User STD: ", np.mean(B_pred_events_nums_std))

    incentives_file = open("users_incentives_old.txt", "a")
    entropy_file = open("entropy/entropies_1.txt", "a")

    for k, v in users_MUs.items():
        index = 0
        incentive = np.sum([vv * cc / len(collections.Counter(v).most_common()) for vv, cc in collections.Counter(v).most_common()])
        H = entropy([cc / len(collections.Counter(v).most_common()) for vv, cc in collections.Counter(v).most_common()])
        to_estimate_MU[index] += incentive
        index += 1
        incentives_file.write(str(k) + ", " + str(incentive) + "\n")
        entropy_file.write(str(k) + ", " + str(H) + "\n")


    incentives_file.close()

    incentives_file = open("users_incentives_2_old.txt", "a")
    entropy_file = open("entropy/entropies_2.txt", "a")

    for k_2, v_2 in users_MUs_2.items():
        index = 0
        incentive = np.sum([vv * cc / len(collections.Counter(v_2).most_common()) for vv, cc in collections.Counter(v_2).most_common()])
        H = entropy([cc / len(collections.Counter(v_2).most_common()) for _, cc in collections.Counter(v_2).most_common()])
        to_estimate_MU_2[index] += incentive
        index += 1
        incentives_file.write(str(k_2) + ", " + str(incentive) + "\n")
        entropy_file.write(str(k_2) + ", " + str(H) + "\n")

    incentives_file.close()
    entropy_file.close()

    temp_f = open(config.labels_path + '/final_symmetric_mtrx.pkl', 'rb')
    whole_network_adj = pickle.load(temp_f)
    temp_f.close()

    temp_f = open(config.labels_path + '/sorted_all_users_ids.pkl', 'rb')
    whole_network_users = pickle.load(temp_f)
    temp_f.close()

    adj_mtrx = create_adjacency_matrix_pegypt(whole_network_adj, whole_network_users, sub_network_user_ids_org)


    print("")
    print("----------------Final Mitigation Results (Loss Function 1)----------------")
    print("")
    budget_used = calc_final_pegypt_results(hawkes_non_propaganda, to_estimate_MU, MU, A,
                                  propaganda_timestamps_org, pred_propaganda_timestamps_org, non_propaganda_timestamps_org,
                                  config.realizations_bounds_A, adj_mtrx, config.balance_factor, 31, config)
    print("Knapsack Carried Budget: ", budget_used)

    print("")
    print("----------------Final Mitigation Results (Loss Function 2)----------------")
    print("")
    budget_used = calc_final_pegypt_results(hawkes_non_propaganda, to_estimate_MU_2, MU, A,
                                  propaganda_timestamps_org, pred_propaganda_timestamps_org, non_propaganda_timestamps_org,
                                  config.realizations_bounds_A, adj_mtrx, config.balance_factor, 31, config)
    print("Knapsack Carried Budget: ", budget_used)







