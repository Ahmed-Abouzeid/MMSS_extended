from model_predictor import Hawkes
from utils import create_simu_vis_graph, format_hawkes_stamps, add_predicted_tweets, merge_timestamps,\
    calc_network_fairness, calc_avg_abs_err, calc_avg_num_events_per_user, create_adjacency_matrix_pegypt, \
    calc_pegypt_propaganda_mitigation, get_plot_data
from model_control import Uniform_Cont, AI_Cont, AI_Cont_Pegypt, AI_Cont_Pegypt_circles
import numpy as np
import pickle
from Comparable_Intervention import *
from Comparable_Network_Measures import *
import time


def run_experiments(config, mis_timestamps, normal_timestamps, tweeting_time, influence_relations, is_avg_method):
    no_keys_mis_timestamps = format_hawkes_stamps(mis_timestamps)
    no_keys_normal_timestamps = format_hawkes_stamps(normal_timestamps)
    """runs all experiments: LA, uniform, and before mitigation"""

    hawkes_mis = Hawkes(no_keys_mis_timestamps, config.decay_factor_mis, config)
    mis_MU, mis_A = hawkes_mis.estimate_params(config.average_out_runs)
    pred_mis_timestamps = hawkes_mis.simulate(mis_MU, mis_A, None, verbose=False, sampling=False)
    if config.show_simu_error:
        err, std = calc_avg_abs_err(pred_mis_timestamps, mis_timestamps, realization_bounds=config.realizations_bounds)
        print('Mis Content Simulation Error and STD: ', err, std)
    del hawkes_mis

    hawkes_normal = Hawkes(no_keys_normal_timestamps, config.decay_factor_norm, config)
    norm_MU, norm_A = hawkes_normal.estimate_params(config.average_out_runs)
    pred_norm_timestamps = hawkes_normal.simulate(norm_MU, norm_A, None, verbose=False, sampling=False)
    if config.show_simu_error:
        err, std = calc_avg_abs_err(pred_norm_timestamps, normal_timestamps, realization_bounds=config.realizations_bounds)
        print('Norm Content Simulation Error and STD: ', err, std)
    merged_mis_timestamps, merged_norm_timestamps = merge_timestamps(mis_timestamps, pred_mis_timestamps,
                                                                     normal_timestamps, pred_norm_timestamps)
    adjaceny_matrix = np.zeros((len(norm_A), len(norm_A)))
    old = norm_MU.copy()
    # iterate through rows
    for i in range(len(norm_A)):
        # iterate through columns
        adjaceny_matrix[i][i] += 1
        for j in range(len(norm_A[0])):
            if i != j:
                if norm_A[i][j] + mis_A[i][j] > 0:
                    adjaceny_matrix[i][j] += 1

    all_percs_la, all_percs_uniform, all_percs_before = [], [], []
    consumed_budgets = []
    all_fairness_la, all_fairness_uniform, all_fairness_before = [], [], []
    for i in range(1):
        if i == 0:
            all_percs_la = []
            all_fairness_la = []
            all_compu_speeds = []
            for r in range(config.average_out_runs):  # we run the experiment n times to average out the error
                t_1 = time.time()
                if is_avg_method:
                    norm_MU, la_consumed_budget = run_avg_method(config, merged_norm_timestamps,
                                                                 merged_mis_timestamps, adjaceny_matrix, norm_MU,
                                                                 norm_A)
                else:
                    ai_cont = AI_Cont(merged_mis_timestamps, merged_norm_timestamps, adjaceny_matrix, 4, config,
                                      old, norm_A, normal_timestamps)
                    norm_MU, la_consumed_budget = ai_cont.control()
                t_2 = time.time()
                pred_norm_timestamps = hawkes_normal.simulate(norm_MU, norm_A, None, verbose=False, sampling=False)
                # now we merge real events + predicted events from future realization(s)
                merged_mis_timestamps_after_control, merged_norm_timestamps_after_control = merge_timestamps(
                    mis_timestamps, pred_mis_timestamps,
                    normal_timestamps,
                    pred_norm_timestamps)
                network_fairness_loss = calc_network_fairness(merged_norm_timestamps_after_control,
                                                              merged_mis_timestamps_after_control,
                                                              config.realizations_bounds,
                                                              adjaceny_matrix,
                                                              norm_MU, config.balance_factor)
                print('Network Fairness Loss: (Method=' + str(i) + ')', network_fairness_loss)

                tweeting_time_merged, normal_simu_tweets, mis_simu_tweets = add_predicted_tweets(tweeting_time,
                                                                                                 pred_mis_timestamps,
                                                                                                 pred_norm_timestamps)

                mis_perc_last_stage = create_simu_vis_graph(range(config.realizations_n), config.realizations_bounds,
                                                            config.labels_path, adjaceny_matrix,
                                                            merged_mis_timestamps_after_control,
                                                            merged_norm_timestamps_after_control,
                                                            tweeting_time_merged,
                                                            influence_relations,
                                                            normal_simu_tweets, mis_simu_tweets, str(i))
                all_percs_la.append(mis_perc_last_stage)
                all_fairness_la.append(network_fairness_loss)
                all_compu_speeds.append(t_2-t_1)
                consumed_budgets.append(la_consumed_budget)

        elif i == 1:
            all_percs_uniform = []
            all_fairness_uniform = []
            for r in range(config.average_out_runs):  # we run the experiment n times to average out the error
                uniform_cont = Uniform_Cont(np.mean(consumed_budgets), old)
                norm_MU = uniform_cont.control()
                pred_norm_timestamps = hawkes_normal.simulate(norm_MU, norm_A, None, verbose=False, sampling=False)
                # now we merge real events + predicted events from future realization(s)
                merged_mis_timestamps_after_control, merged_norm_timestamps_after_control = merge_timestamps(
                    mis_timestamps, pred_mis_timestamps,
                    normal_timestamps,
                    pred_norm_timestamps)
                network_fairness_loss = calc_network_fairness(merged_norm_timestamps_after_control,
                                                              merged_mis_timestamps_after_control,
                                                              config.realizations_bounds,
                                                              adjaceny_matrix,
                                                              norm_MU, config.balance_factor)
                print('Network Fairness Loss: (Method=' + str(i) + ')', network_fairness_loss)

                tweeting_time_merged, normal_simu_tweets, mis_simu_tweets = add_predicted_tweets(tweeting_time,
                                                                                                 pred_mis_timestamps,
                                                                                                 pred_norm_timestamps)
                mis_perc_last_stage = create_simu_vis_graph(range(config.realizations_n), config.realizations_bounds,
                                                            config.labels_path, adjaceny_matrix,
                                                            merged_mis_timestamps_after_control,
                                                            merged_norm_timestamps_after_control,
                                                            tweeting_time_merged,
                                                            influence_relations,
                                                            normal_simu_tweets, mis_simu_tweets, str(i))
                all_percs_uniform.append(mis_perc_last_stage)
                all_fairness_uniform.append(network_fairness_loss)

        elif i == 2:
            all_percs_before = []
            all_fairness_before = []
            for r in range(config.average_out_runs):  # we run the experiment n times to average out the error
                norm_MU = old
                pred_norm_timestamps = hawkes_normal.simulate(norm_MU, norm_A, None, verbose=False, sampling=False)
                # now we merge real events + predicted events from future realization(s)
                merged_mis_timestamps, merged_norm_timestamps = merge_timestamps(mis_timestamps, pred_mis_timestamps,
                                                                                 normal_timestamps,
                                                                                 pred_norm_timestamps)
                network_fairness_loss = calc_network_fairness(merged_norm_timestamps, merged_mis_timestamps,
                                                              config.realizations_bounds,
                                                              adjaceny_matrix,
                                                              norm_MU, config.balance_factor)
                print('Network Fairness Loss: (Method=' + str(i) + ')', network_fairness_loss)

                tweeting_time_merged, normal_simu_tweets, mis_simu_tweets = add_predicted_tweets(tweeting_time,
                                                                                                 pred_mis_timestamps,
                                                                                                 pred_norm_timestamps)

                mis_perc_last_stage = create_simu_vis_graph(range(config.realizations_n), config.realizations_bounds,
                                                            config.labels_path, adjaceny_matrix,
                                                            merged_mis_timestamps, merged_norm_timestamps,
                                                            tweeting_time_merged,
                                                            influence_relations,
                                                            normal_simu_tweets, mis_simu_tweets, str(i))
                all_percs_before.append(mis_perc_last_stage)
                all_fairness_before.append(network_fairness_loss)

    print('\n########################################################################################################')
    print('Averaged Results of Mis Perc Last Stage (LA - Uniform - Before ANy Intervention):',
          np.mean(all_percs_la), np.mean(all_percs_uniform), np.mean(all_percs_before))
    print('Averaged Results of achieved network fairness (LA - Uniform - Before ANy Intervention):',
          np.mean(all_fairness_la), np.mean(all_fairness_uniform), np.mean(all_fairness_before))
    print('Consumed Budget On Average: ', np.mean(consumed_budgets))
    print('Computation Speed On Average (Seconds): ', np.mean(all_compu_speeds))
    print('Computation Speed On Average (Minutes): ', np.mean(all_compu_speeds)/60)
    print('STD for Fairness Error: ', np.std(all_fairness_la))
    print('STD for Final Mis Info Perc: ', np.std(all_percs_la))
    print('STD for Mis Info Perc Before Intervention: ', np.std(all_percs_before))
    print('STD for Compu Speed: ', np.std(all_compu_speeds))
    print('########################################################################################################')


def run_avg_method(config, merged_norm_timestamps, merged_mis_timestamps, adjaceny_matrix, norm_MU, norm_A):
    '''runs other mitigation method in the literature (Abouzeid et.al 2021) where no fairness is considered'''
    """runs all experiments: LA, uniform, and before mitigation"""

    results_before_mitigation = get_high_exposures_users(merged_norm_timestamps, merged_mis_timestamps, adjaceny_matrix,
                                                         5, config.realizations_bounds, None)
    mitigated_users_per_stage = get_selected_users_per_stage(results_before_mitigation)

    fake_news_data = get_stages_avg_exposures(results_before_mitigation)[1]

    sample_size = 200
    sens_param = 0.001
    epochs = 20
    n = Network(.98, config.budget, config.shrinked_network_size, 1, epochs, norm_MU,
                config.decay_factor_mis, config.decay_factor_norm, adjaceny_matrix, norm_A, config.realizations_n, config.realizations_bounds,
                config.la_m_depth, fake_news_data, merged_mis_timestamps, mitigated_users_per_stage, sample_size, sens_param)
    la_mhp_results = n.run(4)
    consumed_la_budget = sum(la_mhp_results.values())
    for user_id, _ in enumerate(norm_MU):
        norm_MU[user_id] += la_mhp_results[user_id]

    return norm_MU, consumed_la_budget


def run_pegypt_experiments(sample_code, config, subnetwork_users, timestamps_1, timestamps_2, timestamps_3 = None, timestamps_4 = None,
                           timestamps_5 = None, mode= 0, probs_tracker = None, mapping_info = None):

    errs1 = []
    errs2 = []
    errs1_std = []
    real_nums = []
    pred_nums = []
    real_nums_std = []
    pred_nums_std = []
    plot_real_values = []
    plot_pred_values = []

    if mode == 0:
        no_keys_propaganda_timestamps = format_hawkes_stamps(timestamps_1)
        hawkes_propaganda = Hawkes(no_keys_propaganda_timestamps, config.decay_factor_type3, config, event_category='A')
        propaganda_MU, propaganda_A = hawkes_propaganda.estimate_params(config.average_out_runs)
        pred_propaganda_timestamps = hawkes_propaganda.simulate(propaganda_MU, propaganda_A, None, verbose=False, sampling=False)
        err, std, err_ = calc_avg_abs_err(pred_propaganda_timestamps, timestamps_1,
                               realization_bounds=config.realizations_bounds_A)
        real_avg_event_num, real_std, pred_avg_event_num, pred_std = \
            calc_avg_num_events_per_user(pred_propaganda_timestamps, timestamps_1,
                                         realization_bounds=config.realizations_bounds_A)
        X , real_values, pred_values = get_plot_data(pred_propaganda_timestamps, timestamps_1, config.realizations_bounds_A)


        errs1.append(err)
        errs2.append(err_)
        errs1_std.append(std)

        real_nums.append(real_avg_event_num)
        pred_nums.append(pred_avg_event_num)

        real_nums_std.append(real_std)
        pred_nums_std.append(pred_std)

        plot_real_values.append(real_values)
        plot_pred_values.append(pred_values)

        del hawkes_propaganda

        no_keys_non_propaganda_timestamps = format_hawkes_stamps(timestamps_2)
        hawkes_non_propaganda = Hawkes(no_keys_non_propaganda_timestamps, config.decay_factor_type4, config, event_category='A')
        non_propaganda_MU, non_propaganda_A = hawkes_non_propaganda.estimate_params(config.average_out_runs)
        pred_non_propaganda_timestamps = hawkes_non_propaganda.simulate(non_propaganda_MU, non_propaganda_A, None, verbose=False, sampling=False)
        err, std, err_ = calc_avg_abs_err(pred_non_propaganda_timestamps, timestamps_2,
                               realization_bounds=config.realizations_bounds_A)
        real_avg_event_num, real_std, pred_avg_event_num, pred_std = \
            calc_avg_num_events_per_user(pred_non_propaganda_timestamps, timestamps_2,
                                         realization_bounds=config.realizations_bounds_A)

        X , real_values, pred_values = get_plot_data(pred_non_propaganda_timestamps, timestamps_2, config.realizations_bounds_A)

        errs1.append(err)
        errs2.append(err_)
        errs1_std.append(std)

        real_nums.append(real_avg_event_num)
        pred_nums.append(pred_avg_event_num)

        real_nums_std.append(real_std)
        pred_nums_std.append(pred_std)

        plot_real_values.append(real_values)
        plot_pred_values.append(pred_values)

        del hawkes_non_propaganda

        no_keys_bias_against_timestamps = format_hawkes_stamps(timestamps_3)
        hawkes_bias_against = Hawkes(no_keys_bias_against_timestamps, config.decay_factor_type8, config, event_category='C')
        bias_against_MU, bias_against_A = hawkes_bias_against.estimate_params(config.average_out_runs)
        pred_bias_against_timestamps = hawkes_bias_against.simulate(bias_against_MU, bias_against_A, None, verbose=False,
                                                                sampling=False)
        err, std, err_ = calc_avg_abs_err(pred_bias_against_timestamps, timestamps_3,
                                          realization_bounds=config.realizations_bounds_C)
        real_avg_event_num, real_std, pred_avg_event_num, pred_std = \
            calc_avg_num_events_per_user(pred_bias_against_timestamps, timestamps_3,
                                         realization_bounds=config.realizations_bounds_C)
        X, real_values, pred_values = get_plot_data(pred_bias_against_timestamps, timestamps_3,
                                                    config.realizations_bounds_C)

        errs1.append(err)
        errs2.append(err_)
        errs1_std.append(std)

        real_nums.append(real_avg_event_num)
        pred_nums.append(pred_avg_event_num)

        real_nums_std.append(real_std)
        pred_nums_std.append(pred_std)

        plot_real_values.append(real_values)
        plot_pred_values.append(pred_values)

        no_keys_B_timestamps = format_hawkes_stamps(timestamps_4)
        hawkes_B = Hawkes(no_keys_B_timestamps, config.decay_factor_type9, config,
                                     event_category='C')
        B_MU, B_A = hawkes_B.estimate_params(config.average_out_runs)
        pred_B_timestamps = hawkes_B.simulate(B_MU, B_A, None,
                                                                    verbose=False,
                                                                    sampling=False)
        err, std, err_ = calc_avg_abs_err(pred_B_timestamps, timestamps_4,
                                          realization_bounds=config.realizations_bounds_C)
        real_avg_event_num, real_std, pred_avg_event_num, pred_std = \
            calc_avg_num_events_per_user(pred_B_timestamps, timestamps_4,
                                         realization_bounds=config.realizations_bounds_C)
        X, real_values, pred_values = get_plot_data(pred_B_timestamps, timestamps_4,
                                                    config.realizations_bounds_C)

        errs1.append(err)
        errs2.append(err_)
        errs1_std.append(std)

        real_nums.append(real_avg_event_num)
        pred_nums.append(pred_avg_event_num)

        real_nums_std.append(real_std)
        pred_nums_std.append(pred_std)

        plot_real_values.append(real_values)
        plot_pred_values.append(pred_values)

        temp_f = open(config.labels_path + '/final_symmetric_mtrx.pkl', 'rb')
        whole_network_adj = pickle.load(temp_f)
        temp_f.close()

        temp_f = open(config.labels_path + '/sorted_all_users_ids.pkl', 'rb')
        whole_network_users = pickle.load(temp_f)
        temp_f.close()

        adj_mtrx = create_adjacency_matrix_pegypt(whole_network_adj, whole_network_users, subnetwork_users)
        MU = non_propaganda_MU
        A = non_propaganda_A

        merged_propaganda_timestamps, merged_non_propaganda_timestamps = merge_timestamps(timestamps_1,
                                                                                          pred_propaganda_timestamps,
                                                                                          timestamps_2,
                                                                                          pred_non_propaganda_timestamps)

        ai_cont_pegypt = AI_Cont_Pegypt(sample_code, merged_propaganda_timestamps, merged_non_propaganda_timestamps,
                                        adj_mtrx, MU, A, timestamps_2, 31, config)

        non_propaganda_incentives_MU, la_consumed_budget = ai_cont_pegypt.control()

        ai_cont_pegypt_2 = AI_Cont_Pegypt_circles(sample_code, merged_propaganda_timestamps, merged_non_propaganda_timestamps,
                                        adj_mtrx, MU, A, timestamps_2, 31, config)

        non_propaganda_incentives_MU_2, la_consumed_budget_2 = ai_cont_pegypt_2.control(mapping_info, probs_tracker,
                                                                                        bias_against_MU,
                                                                                        bias_against_A,
                                                                                        hawkes_bias_against,
                                                                                        B_MU,
                                                                                        B_A,
                                                                                        hawkes_B)

        del hawkes_bias_against
        del hawkes_B

        return non_propaganda_incentives_MU, non_propaganda_incentives_MU_2, pred_nums, real_nums, errs1, errs2, plot_real_values, \
               plot_pred_values, errs1_std, real_nums_std, pred_nums_std

    elif mode == 1:
        no_keys_bias_against_timestamps = format_hawkes_stamps(timestamps_1)
        hawkes_bias_against = Hawkes(no_keys_bias_against_timestamps, config.decay_factor_type1, config, event_category='A')
        bias_against_MU, bias_against_A = hawkes_bias_against.estimate_params(config.average_out_runs)
        pred_bias_against_timestamps = hawkes_bias_against.simulate(bias_against_MU, bias_against_A, None, verbose=False,
                                                                sampling=False)
        err, std, err_ = calc_avg_abs_err(pred_bias_against_timestamps, timestamps_1,
                                          realization_bounds=config.realizations_bounds_A)
        real_avg_event_num, real_std, pred_avg_event_num, pred_std = \
            calc_avg_num_events_per_user(pred_bias_against_timestamps, timestamps_1,
                                         realization_bounds=config.realizations_bounds_A)
        X, real_values, pred_values = get_plot_data(pred_bias_against_timestamps, timestamps_1,
                                                    config.realizations_bounds_A)


        errs1.append(err)
        errs2.append(err_)
        errs1_std.append(std)

        real_nums.append(real_avg_event_num)
        pred_nums.append(pred_avg_event_num)

        real_nums_std.append(real_std)
        pred_nums_std.append(pred_std)

        plot_real_values.append(real_values)
        plot_pred_values.append(pred_values)

        del hawkes_bias_against

        no_keys_bias_for_timestamps = format_hawkes_stamps(timestamps_2)
        hawkes_bias_for = Hawkes(no_keys_bias_for_timestamps, config.decay_factor_type2, config,
                                       event_category='A')
        bias_for_MU, bias_for_A = hawkes_bias_for.estimate_params(config.average_out_runs)
        pred_bias_for_timestamps = hawkes_bias_for.simulate(bias_for_MU, bias_for_A, None,
                                                                        verbose=False, sampling=False)
        err, std, err_ = calc_avg_abs_err(pred_bias_for_timestamps, timestamps_2,
                                          realization_bounds=config.realizations_bounds_A)
        real_avg_event_num, real_std, pred_avg_event_num, pred_std = \
            calc_avg_num_events_per_user(pred_bias_for_timestamps, timestamps_2,
                                         realization_bounds=config.realizations_bounds_A)

        X, real_values, pred_values = get_plot_data(pred_bias_for_timestamps, timestamps_2,
                                                    config.realizations_bounds_A)

        errs1.append(err)
        errs2.append(err_)
        errs1_std.append(std)

        real_nums.append(real_avg_event_num)
        pred_nums.append(pred_avg_event_num)

        real_nums_std.append(real_std)
        pred_nums_std.append(pred_std)

        plot_real_values.append(real_values)
        plot_pred_values.append(pred_values)

        del hawkes_bias_for


        return pred_nums, real_nums, errs1, errs2, plot_real_values, \
               plot_pred_values, errs1_std, real_nums_std, pred_nums_std, pred_bias_against_timestamps,\
               pred_bias_for_timestamps


    elif mode == 2:
        no_keys_A_timestamps = format_hawkes_stamps(timestamps_1)
        hawkes_A = Hawkes(no_keys_A_timestamps, config.decay_factor_type5, config,
                                     event_category='C')
        A_MU, A_A = hawkes_A.estimate_params(config.average_out_runs)
        pred_A_timestamps = hawkes_A.simulate(A_MU, A_A, None,
                                                                    verbose=False,
                                                                    sampling=False)
        err, std, err_ = calc_avg_abs_err(pred_A_timestamps, timestamps_1,
                                          realization_bounds=config.realizations_bounds_C)
        real_avg_event_num, real_std, pred_avg_event_num, pred_std = \
            calc_avg_num_events_per_user(pred_A_timestamps, timestamps_1,
                                         realization_bounds=config.realizations_bounds_C)
        X, real_values, pred_values = get_plot_data(pred_A_timestamps, timestamps_1,
                                                    config.realizations_bounds_C)

        errs1.append(err)
        errs2.append(err_)
        errs1_std.append(std)

        real_nums.append(real_avg_event_num)
        pred_nums.append(pred_avg_event_num)

        real_nums_std.append(real_std)
        pred_nums_std.append(pred_std)

        plot_real_values.append(real_values)
        plot_pred_values.append(pred_values)

        del hawkes_A

        no_keys_B_timestamps = format_hawkes_stamps(timestamps_2)
        hawkes_B = Hawkes(no_keys_B_timestamps, config.decay_factor_type6, config,
                                 event_category='C')
        B_MU, B_A = hawkes_B.estimate_params(config.average_out_runs)
        pred_B_timestamps = hawkes_B.simulate(B_MU, B_A, None,
                                                            verbose=False, sampling=False)
        err, std, err_ = calc_avg_abs_err(pred_B_timestamps, timestamps_2,
                                          realization_bounds=config.realizations_bounds_C)
        real_avg_event_num, real_std, pred_avg_event_num, pred_std = \
            calc_avg_num_events_per_user(pred_B_timestamps, timestamps_2,
                                         realization_bounds=config.realizations_bounds_C)

        X, real_values, pred_values = get_plot_data(pred_B_timestamps, timestamps_2,
                                                    config.realizations_bounds_C)

        errs1.append(err)
        errs2.append(err_)
        errs1_std.append(std)

        real_nums.append(real_avg_event_num)
        pred_nums.append(pred_avg_event_num)

        real_nums_std.append(real_std)
        pred_nums_std.append(pred_std)

        plot_real_values.append(real_values)
        plot_pred_values.append(pred_values)

        del hawkes_B

        no_keys_C_timestamps = format_hawkes_stamps(timestamps_3)
        hawkes_C = Hawkes(no_keys_C_timestamps, config.decay_factor_type7, config,
                                 event_category='B')
        C_MU, C_A = hawkes_C.estimate_params(config.average_out_runs)
        pred_C_timestamps = hawkes_C.simulate(C_MU, C_A, None,
                                                            verbose=False, sampling=False)
        err, std, err_ = calc_avg_abs_err(pred_C_timestamps, timestamps_3,
                                          realization_bounds=config.realizations_bounds_B)
        real_avg_event_num, real_std, pred_avg_event_num, pred_std = \
            calc_avg_num_events_per_user(pred_C_timestamps, timestamps_3,
                                         realization_bounds=config.realizations_bounds_B)

        X, real_values, pred_values = get_plot_data(pred_C_timestamps, timestamps_3,
                                                    config.realizations_bounds_B)

        errs1.append(err)
        errs2.append(err_)
        errs1_std.append(std)

        real_nums.append(real_avg_event_num)
        pred_nums.append(pred_avg_event_num)

        real_nums_std.append(real_std)
        pred_nums_std.append(pred_std)

        plot_real_values.append(real_values)
        plot_pred_values.append(pred_values)

        del hawkes_C

        no_keys_E_timestamps = format_hawkes_stamps(timestamps_4)
        hawkes_E = Hawkes(no_keys_E_timestamps, config.decay_factor_type5, config,
                          event_category='C')
        E_MU, E_A = hawkes_E.estimate_params(config.average_out_runs)
        pred_E_timestamps = hawkes_E.simulate(E_MU, E_A, None,
                                              verbose=False,
                                              sampling=False)
        err, std, err_ = calc_avg_abs_err(pred_E_timestamps, timestamps_4,
                                          realization_bounds=config.realizations_bounds_C)
        real_avg_event_num, real_std, pred_avg_event_num, pred_std = \
            calc_avg_num_events_per_user(pred_E_timestamps, timestamps_4,
                                         realization_bounds=config.realizations_bounds_C)
        X, real_values, pred_values = get_plot_data(pred_E_timestamps, timestamps_4,
                                                    config.realizations_bounds_C)

        errs1.append(err)
        errs2.append(err_)
        errs1_std.append(std)

        real_nums.append(real_avg_event_num)
        pred_nums.append(pred_avg_event_num)

        real_nums_std.append(real_std)
        pred_nums_std.append(pred_std)

        plot_real_values.append(real_values)
        plot_pred_values.append(pred_values)

        del hawkes_E

        no_keys_F_timestamps = format_hawkes_stamps(timestamps_5)
        hawkes_F = Hawkes(no_keys_F_timestamps, config.decay_factor_type5, config,
                          event_category='C')
        F_MU, F_A = hawkes_F.estimate_params(config.average_out_runs)
        pred_F_timestamps = hawkes_F.simulate(F_MU, F_A, None,
                                              verbose=False,
                                              sampling=False)
        err, std, err_ = calc_avg_abs_err(pred_F_timestamps, timestamps_5,
                                          realization_bounds=config.realizations_bounds_C)
        real_avg_event_num, real_std, pred_avg_event_num, pred_std = \
            calc_avg_num_events_per_user(pred_F_timestamps, timestamps_5,
                                         realization_bounds=config.realizations_bounds_C)
        X, real_values, pred_values = get_plot_data(pred_F_timestamps, timestamps_5,
                                                    config.realizations_bounds_C)

        errs1.append(err)
        errs2.append(err_)
        errs1_std.append(std)

        real_nums.append(real_avg_event_num)
        pred_nums.append(pred_avg_event_num)

        real_nums_std.append(real_std)
        pred_nums_std.append(pred_std)

        plot_real_values.append(real_values)
        plot_pred_values.append(pred_values)

        del hawkes_F

        return pred_nums, real_nums, errs1, errs2, plot_real_values, \
               plot_pred_values, errs1_std, real_nums_std, pred_nums_std, pred_A_timestamps, pred_B_timestamps,\
               pred_C_timestamps, pred_E_timestamps, pred_F_timestamps



