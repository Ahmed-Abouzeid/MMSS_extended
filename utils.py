from __future__ import division
import os
import pickle

from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import multiprocessing as mp
from matplotlib import cm
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
output = mp.Queue()


def get_counts(chunked_stamps, compare_realization_id, priority_x = True, x = []):
    """helper function to get counts from both prediction and real events"""

    y = []
    for e, user_stamps in enumerate(chunked_stamps):
        c = len(list(user_stamps[compare_realization_id-1]))
        if priority_x:
            if c > 0:
                y.append(c)
                x.append(e)
        else:
            if e in x:
                y.append(c)

    return x, y


def plot_pred_vs_real(pred_stamps, real_stamps, realization_bounds, compare_realization_id,
                      priority_x= 'real'):
    """plotting the prediction performance by drawing the projection of tweets generated from
    the Hawkes predictor with the actual tweets from real data over the same period of time"""

    pred_chunked = chunk_timestamps(pred_stamps, realization_bounds)
    real_chunked = chunk_timestamps(real_stamps, realization_bounds)

    if priority_x == 'real':
        x, real_y = get_counts(real_chunked, compare_realization_id, True, [])
        x, pred_y = get_counts(pred_chunked, compare_realization_id, False, x)
    else:
        x, pred_y = get_counts(pred_chunked, compare_realization_id, True, [])
        x, real_y = get_counts(real_chunked, compare_realization_id, False, x)

    plt.scatter(x, real_y, marker = 'x', label='real', color='royalblue')
    plt.scatter(x, pred_y, s=80, facecolors='none', edgecolors='lightsteelblue', label='pred')
    plt.grid()
    plt.legend()
    plt.xlabel('Users Indices')
    plt.ylabel('Point Process Counts')

    plt.show()


def plot_simulation_performance(pred_stamps, real_stamps, realization_bounds, fig_tag='simu_error'):
    """plotting the prediction performance by drawing the projection of tweets counts generated from
    the Hawkes predictor with the actual tweets from real data over the same period of time for a list of users"""

    pred_chunked = chunk_timestamps(pred_stamps, realization_bounds)
    real_chunked = chunk_timestamps(real_stamps, realization_bounds)

    event_nums_real = []
    event_nums_pred = []
    X = []
    for e, user in enumerate(pred_chunked[:100]):
        X.append(e)
        event_nums_pred.append(len(user[-1]))
        event_nums_real.append(len(real_chunked[e][-1]))

    plt.plot(X, event_nums_real, '-*', label='Reality', color='royalblue')
    plt.plot(X, event_nums_pred, '-+', color='red', label='Prediction')
    plt.grid()
    plt.legend()
    plt.xlabel('Users Indices')
    plt.ylabel('Point Process Counts')

    plt.savefig('simulation_error_'+fig_tag+'.png')
    plt.cla()


def plot_simulation_performance_non_timestamps(X, Y1, Y2, figure_path):

    plt.plot(X, Y1, '-*', label='Reality', color='royalblue')
    plt.plot(X, Y2, '-+', color='red', label='Prediction')
    plt.grid()
    plt.legend()
    plt.xlabel('Users Indices')
    plt.ylabel('Point Process Counts')

    plt.savefig(figure_path)
    plt.cla()


def plot_user_event_dist(X, Y, figure_path, caption):

    if caption == "Real":
        plt.plot(X, Y, label=caption, color='royalblue')
    else:
        plt.plot(X, Y, label=caption, color='red')

    plt.grid()
    plt.legend()
    plt.xlabel('User Events Distribution')
    plt.ylabel('Probability')

    plt.savefig(figure_path)
    plt.cla()


def get_plot_data(pred_stamps, real_stamps, realization_bounds):

    pred_chunked = chunk_timestamps(pred_stamps, realization_bounds)
    real_chunked = chunk_timestamps(real_stamps, realization_bounds)

    event_nums_real = []
    event_nums_pred = []
    X = []
    for e, user in enumerate(pred_chunked):
        X.append(e)
        event_nums_pred.append(len(user[-1]))
        event_nums_real.append(len(real_chunked[e][-1]))

    return X, event_nums_real, event_nums_pred


def calc_avg_abs_err(pred_stamps, real_stamps, realization_bounds):
    """calculate the avg absolute difference error on the last tim realization (the predicted one)"""

    pred_chunked = chunk_timestamps(pred_stamps, realization_bounds)
    real_chunked = chunk_timestamps(real_stamps, realization_bounds)

    errors = []
    errors_exc_zeros = []
    for e, user in enumerate(pred_chunked):
        errors.append(abs(len(user[-1]) - len(real_chunked[e][-1])))
        if len(user[-1]) != 0 or len(real_chunked[e][-1]) != 0:
            errors_exc_zeros.append(abs(len(user[-1]) - len(real_chunked[e][-1])))

    #plot_simulation_performance(pred_stamps, real_stamps, realization_bounds, fig_tag)

    return np.mean(errors), np.std(errors), np.mean(errors_exc_zeros)


def calc_avg_num_events_per_user(pred_stamps, real_stamps, realization_bounds):
    """calculate the avg num of events each user created from real and predicted data. on the last tim realization."""

    real_chunked = chunk_timestamps(real_stamps, realization_bounds)
    pred_chunked = chunk_timestamps(pred_stamps, realization_bounds)

    real_events_num = []
    pred_events_num = []
    for e, user in enumerate(real_chunked):
        real_events_num.append(len(user[-1]))
        pred_events_num.append(len(pred_chunked[e][-1]))

    return np.mean(real_events_num), np.std(real_events_num), np.mean(pred_events_num), np.std(pred_events_num)


def get_network_metadata(retweets_path):
    """function to return the user ids and timing of acting on certain events
    and it returns also the influencing and influenced users in a social network (following or sharing)."""

    influencing = set()
    influenced = set()
    all_users = set()
    users_tweeting_time = {}
    influence_relations = {}

    files_list = os.listdir(retweets_path)

    for f in tqdm(files_list, 'processing dataset files'):
        reader = open(retweets_path + '/' + f, 'r')
        text = reader.readlines()
        reader.close()
        for e, line in enumerate(text):
            div_line = line.split('->')
            try:
                user_1 = int(div_line[0].strip("[]").split(",")[0].strip("'"))
            except Exception:
                continue
            if user_1 == "ROOT":
                continue  # skipping first useless line if necessary
            user_1_tweet_id = div_line[0].strip("[]").split(",")[1].strip("'").rstrip("'").lstrip(" '")
            user_1_timing = float(div_line[0].strip("[]").split(",")[2].strip('"').rstrip("'").rstrip("]") \
                                  .lstrip(" '").strip("\n")) * 60  # convert minutes to seconds
            if user_1 not in users_tweeting_time.keys():
                users_tweeting_time.update({user_1: [(user_1_tweet_id, user_1_timing)]})
            else:
                if (user_1_tweet_id, user_1_timing) not in users_tweeting_time[user_1]:
                    users_tweeting_time[user_1].append((user_1_tweet_id, user_1_timing))

            user_2 = int(div_line[1].strip("[]").split(",")[0].strip("'"))
            user_2_tweet_id = div_line[1].strip("[]").split(",")[1].strip("'").rstrip("'").lstrip(" '")
            user_2_timing = float(div_line[1].strip("[]").split(",")[2].strip('"').rstrip("'").rstrip("]")\
                .lstrip(" '").strip("\n").rstrip("']")) * 60  # convert minutes to seconds
            if user_2 not in users_tweeting_time.keys():
                users_tweeting_time.update({user_2: [(user_2_tweet_id, user_2_timing)]})
            else:
                if (user_2_tweet_id, user_2_timing) not in users_tweeting_time[user_2]:
                    users_tweeting_time[user_2].append((user_2_tweet_id, user_2_timing))

            if user_1 not in influencing:
                influencing.add(user_1)
            if user_2 not in influenced:
                influenced.add(user_2)

            if user_1 not in influence_relations.keys():
                influence_relations.update({user_1: [(user_2, user_2_timing)]})  # we add timing to know when influence first occurred
            else:
                influence_relations[user_1].append((user_2, user_2_timing))

            all_users.add(user_1)
            all_users.add(user_2)

    mapped_users_ids, newids_influencing, newids_influenced, newids_tweeting_time, newids_influence_relations =\
        map_original_ids(all_users, influencing, influenced, users_tweeting_time, influence_relations)

    return mapped_users_ids, newids_influencing, newids_influenced, newids_tweeting_time, newids_influence_relations


def create_pegypt_sub_network(data_path, sub_network_size):
    users_f = open(data_path + '/sorted_all_users_ids.pkl', "rb")
    users = pickle.load(users_f)
    data_f = open(data_path + '/final_labeled_dataset.pkl', "rb")
    data = pickle.load(data_f)

    data_f.close()
    users_f.close()

    users_event_nums = []
    for i in users:
        n = 0
        for row in data:
            if row['tweet_user_id'] == str(i):
                n += 1
        users_event_nums.append((i, n))

    sorted_by_activity_users = [ent[0] for ent in sorted(users_event_nums, key=lambda x: x[1], reverse=True)]
    sub_network_data = []
    for_perc = 0
    against_perc = 0
    propaganda_perc = 0
    sub_network_users = set()
    for_counter = 0
    against_counter = 0
    propaganda_counter = 0
    biased_for_only = []
    biased_against_only = []
    for user_id in sorted_by_activity_users:

        if for_perc >= 0.40 and against_perc >= 0.50:
            print('Sub-Network Final Size:', len(sub_network_users))
            print('Sub-Network Events Num:', len(sub_network_data))
            print('Sub-Network Biased For Perc:', for_perc)
            print('Sub-Network Biased Against Perc:', against_perc)
            print('Sub-Network Propaganda Perc:', propaganda_perc)
            print('Sub-Network Non-Propaganda Perc:', 1-propaganda_perc)
            max_events_num_per_user = max(
                [ent[1] for ent in sorted(users_event_nums, key=lambda x: x[1], reverse=True) if ent[0] in sub_network_users])
            min_events_num_per_user = min(
                [ent[1] for ent in sorted(users_event_nums, key=lambda x: x[1], reverse=True) if ent[0] in sub_network_users])
            print('Sub-Network Max Events Num Per user:', max_events_num_per_user)
            print('Sub-Network Min Events Num Per user:', min_events_num_per_user)
            sub_data_f = open('sub_network_data.pkl', 'wb')
            sub_users_f = open('sub_network_users.pkl', 'wb')
            pickle.dump(sub_network_data, sub_data_f)
            pickle.dump(sub_network_users, sub_users_f)
            sub_data_f.close()
            sub_users_f.close()
            return sub_network_data, sorted(list(sub_network_users))
        for row in data:
            if row['tweet_user_id'] == str(user_id) and row['tweet_bias_label'] == 1 and for_perc < 0.44:
                if user_id not in sub_network_users and user_id not in biased_against_only:
                    for_counter += 1
                    for_perc = for_counter/sub_network_size
                    biased_for_only.append(user_id)
                sub_network_users.add(user_id)
                sub_network_data.append(row)
                if row['tweet_is_propaganda_label'] == 1:
                    propaganda_counter += 1
                    propaganda_perc = propaganda_counter/len(sub_network_data)

            elif row['tweet_user_id'] == str(user_id) and row['tweet_bias_label'] == -1 and against_perc < 0.51:
                if user_id not in sub_network_users and user_id not in biased_for_only:
                    against_counter += 1
                    against_perc = against_counter/sub_network_size
                    biased_for_only.append(user_id)
                sub_network_users.add(user_id)
                sub_network_data.append(row)
                if row['tweet_is_propaganda_label'] == 1:
                    propaganda_counter += 1
                    propaganda_perc = propaganda_counter/len(sub_network_data)


    return None, None




def create_pegypt_sample_sub_network(all_data, all_users, config):
            sampled_users = sorted(random.sample(all_users, config.sample_size))
            sampled_network_data = []
            for x in all_data:
                if int(x["tweet_user_id"]) in sampled_users:
                    sampled_network_data.append(x)
            return sampled_network_data, sampled_users


def calc_final_pegypt_results(point_process, non_propaganda_incentives_MU, original_MU, A,
                              propaganda_timestamps, pred_propaganda_timestamps, non_propaganda_timestamps,
                              realizations_bounds_A, adj_mtrx, balance_factor, realization_id, config):

    calculated_budjet = 0
    for ai_i, ai_v in enumerate(non_propaganda_incentives_MU):
        calculated_budjet += abs(ai_v - original_MU[ai_i])

    m = calculated_budjet/len(original_MU)
    uniformed_MU = original_MU.copy()

    for e, _ in enumerate(original_MU):
        uniformed_MU[e] += m

    pred_non_propaganda_timestamps = point_process.simulate(non_propaganda_incentives_MU, A,
                                                           None, verbose=False, sampling=False)

    merged_propaganda_timestamps_after_control, merged_non_propaganda_timestamps_after_control = \
        merge_timestamps(propaganda_timestamps, pred_propaganda_timestamps, non_propaganda_timestamps, pred_non_propaganda_timestamps)

    network_fairness_loss = calc_network_fairness(merged_non_propaganda_timestamps_after_control,
                                                  merged_propaganda_timestamps_after_control,
                                                  realizations_bounds_A,
                                                  adj_mtrx,
                                                  original_MU, balance_factor, realization_id)
    print('AI - Network Fairness Loss: ', network_fairness_loss)
    avg_propaganda_after, avg_propaganda_before = calc_pegypt_propaganda_mitigation(
        merged_propaganda_timestamps_after_control,
        merged_non_propaganda_timestamps_after_control,
        non_propaganda_timestamps,
        adj_mtrx, config.realizations_bounds_B, config.realizations_bounds_A)

    print('Avg Propaganda Percentage on The Network Before Control: ', avg_propaganda_before)
    print('AI - Avg Propaganda Percentage on The Network After Control: ', avg_propaganda_after)
    print('AI - Mitigation Efficiency: ', 1 - (avg_propaganda_after / avg_propaganda_before))

    print("########################################")
    pred_non_propaganda_timestamps = point_process.simulate(uniformed_MU, A,
                                                           None, verbose=False, sampling=False)

    merged_propaganda_timestamps_after_control, merged_non_propaganda_timestamps_after_control = \
        merge_timestamps(propaganda_timestamps, pred_propaganda_timestamps, non_propaganda_timestamps, pred_non_propaganda_timestamps)

    network_fairness_loss = calc_network_fairness(merged_non_propaganda_timestamps_after_control,
                                                  merged_propaganda_timestamps_after_control,
                                                  realizations_bounds_A,
                                                  adj_mtrx,
                                                  uniformed_MU, balance_factor, realization_id)
    print('UNIFORM - Network Fairness Loss: ', network_fairness_loss)
    avg_propaganda_after, avg_propaganda_before = calc_pegypt_propaganda_mitigation(
        merged_propaganda_timestamps_after_control,
        merged_non_propaganda_timestamps_after_control,
        non_propaganda_timestamps,
        adj_mtrx, config.realizations_bounds_B, config.realizations_bounds_A)

    print('Avg Propaganda Percentage on The Network Before Control: ', avg_propaganda_before)
    print('UNIFORM - Avg Propaganda Percentage on The Network After Control: ', avg_propaganda_after)
    print('UNIFORM - Mitigation Efficiency: ', 1 - (avg_propaganda_after / avg_propaganda_before))
    return calculated_budjet





def get_pegypt_metadata(data_path, sub_network_size):
    """function to return the user ids and engagement info."""

    sub_network_data, sub_network_users = create_pegypt_sub_network(data_path, sub_network_size)

    mapped_users_ids = [(e, u_id) for e, u_id in enumerate(sub_network_users)]
    # influence_relations = dict()
    #
    # for e, user_id in enumerate(all_original_users):
    #     if user_id in sub_network_users:
    #         engaged_users = [(e_u_id, 0) for e_u_id, col in enumerate(adj[e]) if col > 0 and match_mapped_ids_2(e_u_id, mapped_users_ids) in sub_network_users]
    #         influence_relations.update({e:engaged_users})

    return mapped_users_ids, sub_network_users, sub_network_data



def get_pegypt_sample_metadata(all_users, config):
    """function to return the user ids and engagement info."""
    data_f = open(config.labels_path + '/final_labeled_dataset.pkl', "rb")
    all_data = pickle.load(data_f)

    data_f.close()
    sample_network_data, sample_network_users = create_pegypt_sample_sub_network(all_data, all_users, config)

    mapped_users_ids = [(e, u_id) for e, u_id in enumerate(sample_network_users)]
    # influence_relations = dict()
    #
    # for e, user_id in enumerate(all_original_users):
    #     if user_id in sub_network_users:
    #         engaged_users = [(e_u_id, 0) for e_u_id, col in enumerate(adj[e]) if col > 0 and match_mapped_ids_2(e_u_id, mapped_users_ids) in sub_network_users]
    #         influence_relations.update({e:engaged_users})

    return mapped_users_ids, sample_network_users, sample_network_data


def map_original_ids(users_ids, influencing_ids, influenced_ids, users_tweeting_time, influence_relations):
    """function to map the original ids to new zero starting indices ids to match with the vanilla Hawkes process
     matrix and vector parameters"""

    mapped_users_ids = []
    newids_influencing = []
    newids_influenced = []
    newids_tweeting_time = {}
    newids_influence_relations = {}

    for e, user_id in tqdm(enumerate(sorted(list(users_ids))), 'creating mapping ids'): # creating new users ids that starts from zero
        mapped_users_ids.append((e, user_id))
        newids_tweeting_time.update({e: users_tweeting_time[user_id]})

        if user_id in influencing_ids:
            newids_influencing.append(e)

        if user_id in influenced_ids:
            newids_influenced.append(e)

    for user_id in tqdm(set(users_ids), 'finalizing mapping ids'):
        if user_id in set(influencing_ids):
            new_ids = []
            for influenced_user_id, timing in set(influence_relations[user_id]):
                new_ids.append((match_mapped_ids(influenced_user_id, mapped_users_ids), timing))
            newids_influence_relations.update({match_mapped_ids(user_id, mapped_users_ids): new_ids})

    return mapped_users_ids, newids_influencing, newids_influenced, newids_tweeting_time, newids_influence_relations


def match_mapped_ids(id, mapping_struct):
    """helper function to fetch an old id from a mapping structure to get its new id"""

    for new_id, old_id in mapping_struct:
        if old_id == id:
            return new_id

    return None

def match_mapped_ids_2(id, mapping_struct):
    """helper function to fetch a new id from a mapping structure to get its old id"""

    for new_id, old_id in mapping_struct:
        if new_id == id:
            return old_id

    return None

def format_hawkes_stamps(timestamps):
    """eliminate the dict keys from the timestamps dicts and return only list of each user timestamps withou that
    key, this is required by the format of the hawkes estimator function input"""

    no_keys_timestamps = []
    for key in timestamps.keys():
        no_keys_timestamps.append(timestamps[key])

    return no_keys_timestamps


def chunk_timestamps(timestamps, realization_bounds):
    """function to transform timestamps into chunks per hour/or some minutes, in order to be able to evaluate
     Hawkes simulation results per each time realization"""

    timestamps_per_realization = []
    for user_id in timestamps.keys():
        user_chunks = []
        for e, bound in enumerate(realization_bounds):
            chunk = []
            for t in timestamps[user_id]:

                if e * realization_bounds[0] < t <= bound:
                    chunk.append(t)

            user_chunks.append(chunk)

        timestamps_per_realization.append(user_chunks)

    return timestamps_per_realization


def create_adjacency_matrix(influence_relations, network_size):
    """takes a dictionary of users as keys and their influenced users (who retweeted them) as values, the constructs
     an adjaceny matrix to build up a network graph of such relations"""

    A = np.zeros((network_size, network_size))
    for influencer in tqdm(sorted(list(influence_relations.keys())), 'creating adjacency matrix'):
        A[influencer][influencer] = 1  # means we consider each influencer user is adjacent to herself, so their past actions reflect
        # on them still
        for influenced, _ in sorted(list(influence_relations[influencer])):
            A[influencer][influenced] = 1
            A[influenced][influencer] = 1  # means no matter who influence whom, we just make the relation bidirectional

    # now we loop over all users to make sure all users at least have themselves as adjacent
    for e, row in enumerate(A):
        A[e][e] = 1
    return A


def create_adjacency_matrix_pegypt(whole_adj, whole_network_users, subnetwork_users):
    """create a subnetwork adjaceny matrix from a whole network adjacency matrix to build up a network graph of such relations"""

    A = np.zeros((len(subnetwork_users), len(subnetwork_users)))
    # we firs convert subnetwork user ids into indices from the original sorted all users ids
    sub_indices = []
    for i in whole_network_users:
        if i in subnetwork_users:
            index = whole_network_users.index(i)
            sub_indices.append(index)

    for e, ind in enumerate(sorted(sub_indices)):
        for row_ind, v in enumerate(whole_adj[ind]):
            if row_ind in sub_indices:
                col_indx = subnetwork_users.index(whole_network_users[row_ind])
                A[e][col_indx] = v


    # A = np.zeros((len(sub_network_users), len(sub_network_users)))
    # for influencer in tqdm(sorted(list(influence_relations.keys())), 'creating adjacency matrix'):
    #     if match_mapped_ids_2(influencer, mapping_info) in sub_network_users:
    #         A[influencer][influencer] = 1  # means we consider each influencer user is adjacent to herself, so their past actions reflect
    #         # on them still
    #         if list(influence_relations[influencer]) != []:
    #             for influenced, _ in sorted(list(influence_relations[influencer])):
    #                 if match_mapped_ids_2(influenced, mapping_info) in sub_network_users:
    #                     A[influencer][influenced] = 1
    #                     A[influenced][influencer] = 1  # means no matter who influence whom, we just make the relation bidirectional
    #
    # # now we loop over all users to make sure all users at least have themselves as adjacent
    # for e, row in enumerate(A):
    #     A[e][e] = 1

    return A


def get_user_adjacencies(adjacency_matrix):
    """helper function to retrieve the adjacent users for each user"""

    adjacents = {}
    for user_index in range(len(adjacency_matrix)):
        adjacent_users = set()
        for e, col in enumerate(adjacency_matrix[user_index]):
            if adjacency_matrix[user_index][e] > 0:
                adjacent_users.add(e)

        for e, row in enumerate(adjacency_matrix):
            for e_col, col in enumerate(row):
                if e_col == user_index:
                    if row[user_index] > 0:
                        adjacent_users.add(e)
        adjacents.update({user_index: adjacent_users})

    return adjacents


def get_influencers(user_index, adjacency_matrix):
    """helper function to retrieve the influencing users for a certain user by passing the latter index to the function"""

    influencers = set()
    for e, col in enumerate(adjacency_matrix[user_index]):
        if adjacency_matrix[user_index][e] > 0:
            influencers.add(e)

    return influencers


def calc_user_campaign_exposure(adjacent_users, timestamps_chunks, realizations_n=None, realization_id=None):
    """calcs the impact of exposure to contents by calc the exposure to misinformation or
     normal information (counts of re/tweets) of the users adjacent other users"""

    if realizations_n is not None and realization_id is None:
        realizations_impacts = []
        for r in range(realizations_n):
            realization_impacts = 0
            for adj_user_id in adjacent_users:
                for prev_realization in range(r):
                    realization_impacts += len(timestamps_chunks[adj_user_id][prev_realization])
                realization_impacts += len(timestamps_chunks[adj_user_id][r])
            realizations_impacts.append(realization_impacts)

        return realizations_impacts

    elif realizations_n is None and realization_id is not None:
        realization_impact = 0
        for user_ind in adjacent_users:
            for prev_realization in range(realization_id):
                realization_impact += len(timestamps_chunks[user_ind][prev_realization])
            realization_impact += len(timestamps_chunks[user_ind][realization_id])

        return realization_impact


def get_event_type(tweet_id, labels_path, normal_tweets=[], mis_tweets=[]):
    """takes a tweet_id, and a groundtruth file path, then returns the type of this tweet either misinformation
     or normal"""

    if tweet_id in normal_tweets:
        return 'n'
    if tweet_id in mis_tweets:
        return 'm'

    reader = open(labels_path, 'r', encoding='cp1252')
    tweets = reader.readlines()
    for tweet in tweets:
        div_line = tweet.split(":")
        if len(div_line) > 1:
            if labels_path.split('/')[2] != 'covid19':
                if tweet_id == div_line[1].strip("\n").strip(' '):
                    if div_line[0] == 'false':
                        return 'm'
                    else:
                        return 'n'
            else:
                if tweet_id == div_line[1].split(',')[0].strip("\n"):
                    if div_line[0] == 'false':
                        return 'm'
                    else:
                        return 'n'
    return None


def create_influence_relations_pegypt(whole_relations, mapping_info, sub_network_users_ids):
    sub_network_relations = {}
    for u_id, info in whole_relations.items():
        if match_mapped_ids_2(u_id, mapping_info) in sub_network_users_ids:
            sub_network_relations.update({u_id:[]})
            for (inf_id, t) in info:
                if match_mapped_ids_2(inf_id, mapping_info) in sub_network_users_ids:
                    sub_network_relations[u_id].append((inf_id, t))

    return sub_network_relations


def merge_timestamps(mis_timestamps, pred_mis_timestamps, normal_timestamps, pred_norm_timestamps):
    """merges misinformation real and prediction, and does the same for the normal information, this is
    for building up the final network after maybe applying some control on the predicted events for the sake of
    mitigating misinformation"""

    merged_mis = {}
    merged_normal = {}

    for key in mis_timestamps.keys():
        merged_mis.update({key: np.concatenate((mis_timestamps[key], pred_mis_timestamps[key]), axis=0)})
    for key in normal_timestamps.keys():
        merged_normal.update({key: np.concatenate((normal_timestamps[key], pred_norm_timestamps[key]), axis=0)})

    return merged_mis, merged_normal

def get_matrix_sample_value(index_1, index_2, original_A):
    """
helper function to retrieve original ids of sampled users from the whole network
    """
    return original_A[index_1][index_2]


def transform_to_sub_network(original_A, original_MU, sample_users_ids):
    """ sample the whole network to a subnetwork to run the Hawkes simulation faster on smaller scale then merge again results
    of the only sampled and propably changed users timestamps with others (not simulated users)
    """
    sample_size = len(sample_users_ids)
    sampled_A = np.zeros((sample_size, sample_size), dtype=float)
    sampled_MU = np.zeros(sample_size, dtype=float)
    mapping = []
    sample_index_1 = 0
    for original_index_1 in sample_users_ids:
        sample_index_2 = 0
        for original_index_2 in sample_users_ids:
            sampled_A[sample_index_1][sample_index_2] = get_matrix_sample_value(original_index_1, original_index_2, original_A)
            sample_index_2 += 1
        mapping.append((sample_index_1, original_index_1))
        sample_index_1 += 1

    for user_id, base_intensity in enumerate(original_MU):
        if user_id in sample_users_ids:
            sampled_MU[fetch_mapped_id(user_id, mapping)] = base_intensity

    return sampled_A, sampled_MU, mapping


def sample_network(controlled_user_id, original_network_size, sample_size):
    '''this function runs random sampling over all network to return only subnetwork of users ids to be simulated later in Hawkes'''

    sample_users_ids = random.sample([u_id for u_id in range(original_network_size)], sample_size-1)
    if controlled_user_id not in sample_users_ids:
        sample_users_ids.append(controlled_user_id)

    return sample_users_ids


def fetch_mapped_id(old_id, mapping_info):
    """ a helper function used to do mapping between ids in the sampled network and the whole network
    """
    for info in mapping_info:
        if info[1] == old_id:
            return info[0]


def get_categorized_timestamps(tweeting_time, labels_path):
    """given all real data tweeting times, and the ground truth of these tweets, the function returns
    the dict of misinformation and normal tweets times where user id is the dict key"""

    mis_timestamps = {}
    normal_timestamps = {}
    for user_id in tqdm(sorted(list(tweeting_time.keys())), 'processing users tweeting timestamps'):
        mis_events = []
        normal_events = []
        for tweet_id, timing in sorted(tweeting_time[user_id], key= lambda tup : tup[1]):
            type = get_event_type(tweet_id, labels_path)
            if type == 'm':
                mis_events.append(timing)
            else:
                normal_events.append(timing)
        mis_timestamps.update({user_id: np.array(mis_events)})  # we pass np array instead of a list for Hawkes required input format
        normal_timestamps.update({user_id: np.array(normal_events)})

    return mis_timestamps, normal_timestamps


def get_categorized_timestamps_bias(data, mapping_info, sub_network):
    """given all real data tweeting times, and the ground truth of these tweets, the function returns
    the dict of bias for, bias neutral, and bias against tweets times where user id is the dict key."""

    bias_for_timestamps = {}
    bias_neutral_timestamps = {}
    bias_against_timestamps = {}

    for org_u_id in sub_network:
        for v in data:
            if int(v['tweet_user_id']) == org_u_id:
                u_id = fetch_mapped_id(org_u_id, mapping_info)
                if u_id is not None:
                    if u_id not in bias_against_timestamps.keys():
                        bias_against_timestamps.update({u_id: []})
                    if u_id not in bias_for_timestamps.keys():
                        bias_for_timestamps.update({u_id: []})
                    if u_id not in bias_neutral_timestamps.keys():
                        bias_neutral_timestamps.update({u_id: []})

                    if int(v['tweet_bias_label']) == -1:
                        bias_against_timestamps[u_id].append(v['tweet_time'])
                    elif int(v['tweet_bias_label']) == 1:
                        bias_for_timestamps[u_id].append(v['tweet_time'])
                    elif int(v['tweet_bias_label']) == 0:
                        bias_neutral_timestamps[u_id].append(v['tweet_time'])

    return convert_timestamps_to_seconds(bias_for_timestamps), convert_timestamps_to_seconds(bias_neutral_timestamps),\
           convert_timestamps_to_seconds(bias_against_timestamps)


def get_categorized_timestamps_propaganda(data, mapping_info, sub_network):
    """given all real data tweeting times, and the ground truth of these tweets, the function returns
    the dict of propaganda , and non propaganda tweets times where user id is the dict key."""

    propaganda_timestamps = {}
    non_propaganda_timestamps = {}

    for org_u_id in sub_network:
        for v in data:
            if int(v['tweet_user_id']) == org_u_id:
                u_id = fetch_mapped_id(org_u_id, mapping_info)
                if u_id is not None:
                    if u_id not in propaganda_timestamps.keys():
                        propaganda_timestamps.update({u_id: []})
                    if u_id not in non_propaganda_timestamps.keys():
                        non_propaganda_timestamps.update({u_id: []})

                    if int(v['tweet_is_propaganda_label']) == 1:
                        propaganda_timestamps[u_id].append(v['tweet_time'])
                    elif int(v['tweet_is_propaganda_label']) == 0:
                        non_propaganda_timestamps[u_id].append(v['tweet_time'])

    return convert_timestamps_to_seconds(propaganda_timestamps), convert_timestamps_to_seconds(non_propaganda_timestamps)


def get_categorized_timestamps_circles(data, mapping_info, sub_network):
    """given all real data tweeting times, and the ground truth of these tweets, the function returns
    the dict of circles A, B, C, D, E, and F event times where user id is the dict key."""

    A = {}
    B = {}
    C = {}
    D = {}
    E = {}
    F = {}

    for org_u_id in sub_network:
        for v in data:
            if int(v['tweet_user_id']) == org_u_id:
                u_id = fetch_mapped_id(org_u_id, mapping_info)
                if u_id is not None:
                    if u_id not in A.keys():
                        A.update({u_id: []})
                    if u_id not in B.keys():
                        B.update({u_id: []})
                    if u_id not in C.keys():
                        C.update({u_id: []})
                    if u_id not in D.keys():
                        D.update({u_id: []})
                    if u_id not in E.keys():
                        E.update({u_id: []})
                    if u_id not in F.keys():
                        F.update({u_id: []})

                    if 'A' in v['tweet_temporal_circles_edges']:
                        A[u_id].append(v['tweet_time'])
                    elif 'B' in v['tweet_temporal_circles_edges']:
                        B[u_id].append(v['tweet_time'])
                    elif 'C' in v['tweet_temporal_circles_edges']:
                        C[u_id].append(v['tweet_time'])
                    elif 'D' in v['tweet_temporal_circles_edges']:
                        D[u_id].append(v['tweet_time'])
                    elif 'E' in v['tweet_temporal_circles_edges']:
                        E[u_id].append(v['tweet_time'])
                    elif 'F' in v['tweet_temporal_circles_edges']:
                        F[u_id].append(v['tweet_time'])

    return convert_timestamps_to_seconds(A), convert_timestamps_to_seconds(B), convert_timestamps_to_seconds(C), \
           convert_timestamps_to_seconds(D), convert_timestamps_to_seconds(E), convert_timestamps_to_seconds(F)


def convert_timestamps_to_seconds(users_timestamps):
    """this function converts the dict of users timestamps to seconds, starts with lowest date to convert to 0 and so on.
    this is required to match with the Hawkes Simulation Conventions."""
    converted_users_timestamps = dict()
    days = []
    hours = []
    for k, v in users_timestamps.items():
        converted_times = []
        for t in v:
            converted_entry = 0
            day = int(t.split('T')[0].split('-')[-1])
            hour = int(t.split('T')[1].split(':')[0])
            minute = int(t.split('T')[1].split(':')[1])
            second = int(t.split('T')[1].split(':')[2].split('.')[0])
            if day == 24:
                converted_entry = 0 + hour * 60 + minute + second/60
            elif day == 25:
                converted_entry = 1440 + hour * 60 + minute + second/60
            elif day == 26:
                converted_entry = 2 * 1440 + hour * 60 + minute + second/60
            elif day == 27:
                converted_entry = 3 * 1440 + hour * 60 + minute + second/60

            days.append(day)
            hours.append(hour)
            converted_times.append(float(converted_entry))
        converted_users_timestamps.update({k:np.array(sorted(converted_times))})
    #print(min(days), min(hours), '----', max(days), max(hours))
    return converted_users_timestamps


def is_valid_process(timing, realization_bounds, realization_id):
    """helper function to check if the current graph node or edge is valid to be added in the current time
     realization"""

    if realization_id != 0:
        if realization_bounds[realization_id] >= timing > realization_bounds[realization_id - 1]:
            return True
        else:
            return False
    else:
        if timing <= realization_bounds[realization_id]:
            return True
        else:
            return False


def get_realization_created_nodes(realization_created_nodes, node_id):
    """helper function to help get the nodes and the realization when a node first created"""

    nodes = []
    for node_org_id, r in realization_created_nodes:
        if node_org_id == node_id:
            nodes.append((node_org_id, r))

    return nodes


def add_predicted_tweets(tweeting_time, pred_mis_timestamps, pred_norm_timestamps):
    """an important function to add the predicted events for the future time realization(s)"""

    #w = open(labels_path, 'a')
    new_tweeting_time_dict = {}
    normal_new_tweets = []
    mis_new_tweets = []
    for key in tweeting_time.keys():
        user_pred_tweeting_times = []
        for t in list(pred_norm_timestamps[key]):
            pred_tw_id = str(key) + '++' + str(t) + '##' + str(random.random())
            #w.write('true:' + pred_tw_id + ',simulated' + '\n')
            user_pred_tweeting_times.append((pred_tw_id, t))
            normal_new_tweets.append(pred_tw_id)
        for t in list(pred_mis_timestamps[key]):
            pred_tw_id = str(key) + '+' +  str(t) + '#' + str(random.random())
            #w.write('false:' + pred_tw_id + ',simulated' + '\n')
            user_pred_tweeting_times.append((pred_tw_id, t))
            mis_new_tweets.append(pred_tw_id)

        x = tweeting_time[key] + sorted(user_pred_tweeting_times, key= lambda tup: tup[1])
        new_tweeting_time_dict.update({key: x})
    return new_tweeting_time_dict, normal_new_tweets, mis_new_tweets


def shrink_network(mis_timestamps, normal_timestamps, tweeting_time,
                   influence_relations, network_size):
    """shrinks the network size by selecting subset of users and eliminating the rest from the meta data
    of the network. Useful when interactive data visualization needed for light rendering motion graphs"""

    sh_adj_mtrx = np.zeros((network_size, network_size))
    sh_mis_stamps = {}
    sh_norm_stamps = {}
    sh_tw_time = {}
    sh_inf_rel = {}

    # for e, row in enumerate(adjaceny_matrix[: network_size]):
    #     sh_adj_mtrx[e] = row[:network_size]

    for key in list(mis_timestamps.keys())[:network_size]:
        sh_mis_stamps.update({key: mis_timestamps[key]})
        sh_norm_stamps.update({key: normal_timestamps[key]})
        sh_tw_time.update({key: tweeting_time[key]})
        if key in influence_relations.keys():
            sh_relations = []
            for influenced_id, timing in influence_relations[key]:
                if influenced_id in range(network_size):
                    sh_relations.append((influenced_id, timing))
            if sh_relations != []:
                sh_inf_rel.update({key: sh_relations})

    return sh_mis_stamps, sh_norm_stamps, sh_tw_time, sh_inf_rel


def calc_pegypt_propaganda_mitigation(propaganda_timestamps, non_propaganda_timestamps, non_propaganda_timestamps_before_control,
                                      adj_mtrx, re_b1, re_b2):
    propaganda_chunks = chunk_timestamps(propaganda_timestamps, re_b1)
    non_propaganda_chunks = chunk_timestamps(non_propaganda_timestamps, re_b2)
    non_propaganda_chunks_before = chunk_timestamps(non_propaganda_timestamps_before_control, re_b2)

    adjacency = get_user_adjacencies(adj_mtrx)
    users_propaganda_exposure_perc_after = []
    users_propaganda_exposure_perc_before = []

    for u_id in range(len(adj_mtrx)):
        adjacent_users = adjacency[u_id]
        user_propaganda_exposure = calc_user_campaign_exposure(adjacent_users, propaganda_chunks, None, len(re_b1)-1)
        user_non_propaganda_exposure = calc_user_campaign_exposure(adjacent_users, non_propaganda_chunks, None, len(re_b2)-1)
        user_non_propaganda_exposure_before = calc_user_campaign_exposure(adjacent_users, non_propaganda_chunks_before,
                                                                          None, len(re_b2)-1)
        propganda_perc_after = (user_propaganda_exposure + 0.0000001)/(user_propaganda_exposure + user_non_propaganda_exposure + 0.0000001)
        propganda_perc_before = (user_propaganda_exposure + 0.0000001) /(user_propaganda_exposure + user_non_propaganda_exposure_before + 0.0000001)

        users_propaganda_exposure_perc_after.append(propganda_perc_after)
        users_propaganda_exposure_perc_before.append(propganda_perc_before)

    avg1 = np.mean(users_propaganda_exposure_perc_after)
    avg2 = np.mean(users_propaganda_exposure_perc_before)

    return avg1, avg2


def create_simu_vis_graph(realizations, realization_bounds, labels_path, adjaceny_matrix, mis_timestamps,
                          normal_timestamps, tweeting_time, influence_relations, normal_tweets, mis_tweets, method_id):
    """this method creates nodes and edges for the social network simulation visualization"""

    # the below list stores the info of misinformation exposed users on the network at each realization
    exposure_counts = []
    node_w = open('./graphs/nodes_'+method_id+ '.csv', 'a')
    edge_w = open('./graphs/edges_'+method_id+'.csv', 'a')

    # node_type indicates either user or a re/tweet
    node_w.write('starttime, end_time, id, label, node_color, size' + '\n')
    edge_w.write('starttime, end_time, source, target' + '\n')

    mis_news_chunks = chunk_timestamps(mis_timestamps, realization_bounds)
    normal_news_chunks = chunk_timestamps(normal_timestamps, realization_bounds)

    node_realization_created = {}
    adjacency = get_user_adjacencies(adjaceny_matrix)
    # for time realization, we create user nodes with there tweets, each realization some users might have different exposure so they will be assigned
    # differnt colors and will be dealt with as new nodes after we make older version disapear by the end of each realization except for the last one
    # which is the final state of the node.
    for realization_id in tqdm(realizations, 'creating graph for time realizations'):
        green_exposure_count = 0
        orange_exposure_count = 0
        red_exposure_count = 0
        for main_node_id in tweeting_time.keys():  # node_id is same as user_id, we just use graph terms
            sub_nodes = []
            sub_nodes_times = []
            for e, (tweet_id, tweet_time) in enumerate(tweeting_time[main_node_id]):
                sub_node_id = str(main_node_id) + '_' +str(e) +'_'+tweet_id
                sub_nodes.append(sub_node_id)
                sub_nodes_times.append(tweet_time)
            if min(sub_nodes_times) <= realization_bounds[realization_id]:
                if is_valid_process(min(sub_nodes_times), realization_bounds, realization_id):
                    main_node_time = min(sub_nodes_times)
                else:
                    main_node_time = realization_bounds[realization_id-1] + 200
            else:
                continue
            adjacent_users = adjacency[main_node_id]
            user_mis_exposure = calc_user_campaign_exposure(adjacent_users, mis_news_chunks, None, realization_id)
            user_normal_exposure = calc_user_campaign_exposure(adjacent_users, normal_news_chunks, None, realization_id)
            if user_normal_exposure == user_mis_exposure:
                node_color = 'orange'
                orange_exposure_count += 1
            elif user_normal_exposure >= user_mis_exposure:
                node_color = 'green'
                green_exposure_count += 1
            else:
                node_color = 'red'
                red_exposure_count += 1
            if realization_id == 4:
                main_node_end_time = 100000
            else:
                main_node_end_time = realization_bounds[realization_id]
            node_w.write(str(main_node_time) + ',' + str(main_node_end_time) + ',' + str(main_node_id)+ '_'+ str(realization_id) + ',' + 'U' + ',' + node_color + ','+ '100' +'\n')
            if main_node_id in influence_relations.keys():
                node_type = 'source'
            else:
                node_type = 'target'
            if realization_id not in node_realization_created.keys():
                node_realization_created.update({realization_id: [(main_node_id, main_node_time, main_node_end_time, node_type)]})
            else:
                node_realization_created[realization_id].append((main_node_id, main_node_time, main_node_end_time, node_type))

            # now we record the event nodes of that user, and add an edges between
            for e, sub_node_id in enumerate(sub_nodes):
                sub_node_time = sub_nodes_times[e]
                if sub_node_time <= realization_bounds[realization_id]:
                    if is_valid_process(sub_node_time, realization_bounds, realization_id):
                        sub_node_time = sub_node_time
                    else:
                        sub_node_time = realization_bounds[realization_id-1] + 200
                else:
                    continue
                original_sub_node_id = sub_node_id.split('_')[-1]
                # get the information about that event, either misinformation or normal information
                event_type = get_event_type(original_sub_node_id, labels_path, normal_tweets, mis_tweets)
                if event_type == 'm':
                    sub_node_color = 'red'
                elif event_type == 'n':
                    sub_node_color = 'green'
                else:
                    raise Exception('Cannot find event type in ground truth file. You Must Check the IDs: '
                                    , sub_node_id, original_sub_node_id)

                if realization_id == 4:
                    sub_node_end_time = 100000
                else:
                    sub_node_end_time = realization_bounds[realization_id]
                node_w.write(str(sub_node_time) + ',' + str(sub_node_end_time) + ',' + str(
                    sub_node_id)+ '_' + str(realization_id) + ',' + 'E' + ',' + sub_node_color + ','+ '10'+ '\n')

                # now we add an edge
                edge_w.write(str(sub_node_time) + ',' + str(sub_node_end_time) + ',' + str(main_node_id) + '_' + str(realization_id) +
                             ',' + str(sub_node_id) + '_' + str(realization_id) + '\n')
        exposure_counts.append((green_exposure_count, orange_exposure_count, red_exposure_count))

    # now we add edges between main nodes
    for realization_id in tqdm(realizations, 'creating edges between main nodes'):
        for node_id, node_start_time, node_end_time, node_type in node_realization_created[realization_id]:
            if node_type == 'source':
                target_nodes_info = influence_relations[node_id]
                for target_node, timing in target_nodes_info:
                    if timing <= realization_bounds[realization_id]:
                        if is_valid_process(timing, realization_bounds, realization_id):
                            edge_start_time = timing
                        else:
                            edge_start_time = realization_bounds[realization_id - 1] + 200
                    else:
                        continue
                    if realization_id == 4:
                        edge_end_time = 100000
                    else:
                        edge_end_time = realization_bounds[realization_id]
                    edge_w.write(
                        str(edge_start_time) + ',' + str(edge_end_time) + ',' + str(node_id) + '_' + str(
                            realization_id)
                        + ',' + str(target_node) + '_' + str(realization_id) + '\n')

    perc = []
    for r in exposure_counts:
        normal_exposure_perc = r[0]/(r[0] + r[1] + r[2])
        equal_exposure_perc = r[1]/(r[0] + r[1] + r[2])
        mis_exposure_perc = r[2]/(r[0] + r[1] + r[2])
        perc.append((normal_exposure_perc, equal_exposure_perc, mis_exposure_perc))

    print('Exposures (Normal - Equal - Misinformation) over ', len(realizations), ' realizations:', perc)
    node_w.close()
    edge_w.close()
    return perc[-1][-1]


def estimate_responsivity(mis_MU, norm_MU):
    """from the estimated base intensities, we return onlu users which have zero base intensity for misinformation
    and greater than zero base intensity for normal content. """

    no_bad_intentions_ids = []
    for id in range(len(mis_MU)):
        if mis_MU[id] == 0 and norm_MU[id] != 0:
            no_bad_intentions_ids.append(id)
    return no_bad_intentions_ids


def save_obj_f(x, y, user_id):
    """save the objective function value over different attemps to average it later
     ,for a certain user while optimizing its performance to achieve fairness mitigation"""

    w = open('./obj_func/'+str(user_id) + '.txt', 'a')
    w.write(str(x) + '#' + str(y) + '\n')
    w.close()


def plot_single_point_obj_func(X, Y, user_id, current_state, f):
    """plotting a user objective function after passing the averaged out noise from the different mitigation runs"""

    plt.plot(X, Y, color='red')
    opt_y = Y[X.index(current_state)]
    plt.scatter(x=np.array([current_state]), y=np.array([opt_y]), s= np.array([50]), label = 'Obtained State')
    plt.xlabel('State Value')
    plt.ylabel('Fairness Loss Function')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.grid()
    plt.legend()
    plt.savefig('./optim_figs/'+str(user_id)+ "_" + str(f) + '.png')
    plt.cla()


def plot_two_points_obj_func(user_i_id, user_j_id, user_i_x, user_j_x, user_i_y, user_j_y, demo=False):
    """plotting a two users joint objective function values after passing the averaged out noise from the
    different mitigation runs"""

    if len(user_i_x) > 1 and len(user_j_x) > 1:
        x = np.array(user_i_x)
        y = np.array(user_j_x)
        X, Y = np.meshgrid(x, y)
        sums = np.zeros((len(X), len(X[0])))

        for e_r, row in enumerate(X):
            for e_c, c in enumerate(row):
                sums[e_r][e_c] = user_i_y[user_i_x.index(c)] + user_j_y[user_j_x.index(Y[e_r][e_c])]

        if demo:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            cp = ax.plot_surface(X, Y, sums, cmap=cm.Spectral_r, linewidth=0.1)
            cb = fig.colorbar(cp, shrink=0.5, aspect=5)
            cb.set_label('Fairness Loss Function f(i, j)', rotation=270, labelpad=12)
            ax.set_xlabel('i states')
            ax.set_ylabel('j states')
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.savefig('./optim_figs/' + str(user_i_id) + '_' + str(user_j_id) + '.png')
            plt.cla()


        else:
            fig, ax = plt.subplots(1, 1)
            cp = ax.contourf(X, Y, sums)
            cb = fig.colorbar(cp)

            cb.set_label('Fairness Loss Function f(i, j)', rotation=270, labelpad=12)
            ax.set_xlabel('i state value')
            ax.set_ylabel('j state value')
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            start_positions_x = []
            end_positions_x = []
            start_positions_y = []
            end_positions_y = []

            opt_x_index = user_i_y.index(min(user_i_y))
            opt_y_index = user_j_y.index(min(user_j_y))
            for e, x in enumerate(user_i_x[:opt_x_index+1]):
                if e < len(user_i_x[:opt_x_index+1]) - 1:
                    start_positions_x.append(x)
                    end_positions_x.append(user_i_x[e+1])

            for e, y in enumerate(user_j_x[:opt_y_index+1]):
                if e < len(user_j_x[:opt_y_index+1]) - 1:
                    start_positions_y.append(y)
                    end_positions_y.append(user_j_x[e+1])

            positions_joined = zip(start_positions_x, start_positions_y, end_positions_x, end_positions_y)
            if end_positions_x != [] and end_positions_y != []:
                for start_x, start_y, end_x, end_y in positions_joined:
                    arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                            arrowstyle='simple', color='r', mutation_scale=10)
                    ax.add_patch(arrow)

                if opt_x_index > opt_y_index:
                    remaining_start_end_x = zip(start_positions_x[opt_y_index:], end_positions_x[opt_y_index:])
                    for start_x, end_x in remaining_start_end_x:
                        arrow = FancyArrowPatch((start_x, user_j_x[opt_y_index]), (end_x, user_j_x[opt_y_index]),
                                                arrowstyle='simple', color='r', mutation_scale=10)
                        ax.add_patch(arrow)

                if opt_x_index < opt_y_index:
                    remaining_start_end_y = zip(start_positions_y[opt_x_index:], end_positions_y[opt_x_index:])
                    for start_y, end_y in remaining_start_end_y:
                        arrow = FancyArrowPatch((user_i_x[opt_x_index], start_y), (user_i_x[opt_x_index],
                                                                                                 end_y),
                                                arrowstyle='simple', color='r', mutation_scale=10)
                        ax.add_patch(arrow)

                plt.savefig('./optim_figs/' + str(user_i_id) + '_' + str(user_j_id) + '.png')
                plt.cla()


def average_out_obj_func(data_path, user_id):
    """this function average out all runs of a user objective function and returns the final iteration x axis
    and loss values for the y axis """

    r = open(data_path+'/'+str(user_id)+'.txt', 'r')
    runs = r.readlines()
    all_x = []
    all_y = []
    lengthes = []
    for run in runs:
        x = run.split('#')[0].strip('\n').strip("['. ']").split(',')
        y = run.split('#')[1].strip('\n').strip("[', ']").split(',')
        all_x.append(x)
        all_y.append(y)
        lengthes.append(len(x))
    longest_x = all_x[lengthes.index(max(lengthes))]
    longest_y = all_y[lengthes.index(max(lengthes))]
    longest_x_floated = []
    longest_y_floated = []
    for x in longest_x:
        longest_x_floated.append(float(x))

    for y in longest_y:
        longest_y_floated.append(float(y))

    counter = 1  # we count how many runs had similar state trajectories so we count on them when averaging,
    # we start with a count of 1 for the already recognized longest length run, we keep adding only same length runs
    for y in all_y:
        if len(y) == len(longest_y):
            counter += 1
            for e, v in enumerate(y):
                longest_y_floated[e] += float(v)

    averaged_y = []
    for v in longest_y_floated:
        averaged_y.append(v/counter)

    return longest_x_floated, averaged_y


def save_converged_state(user_id, converged_state):
    """saves the converged state of an automaton"""

    w = open('./converged_states/'+ str(user_id) + '.txt', 'a')
    w.write(str(converged_state) + '\n')
    w.close()


def load_converged_state(user_id):
    """loads the converged state of an automaton"""

    r = open("./converged_states/"+str(user_id)+'.txt', 'r')
    line = r.readlines()
    r.close()
    return float(line[0].strip('\n'))


def network_converged(las):
    """check if all network las are converged"""

    for la in las:
        if not la.converged:
            return False
    else:
        return True


def plot_network_default_diffs(merged_norm_timestamps, merged_mis_timestamps, bounds, adjaceny_matrix, norm_MU):
    """plotting the default betwork diffs between normal and misinformation across all users, by
    displaying the distribution of these diffs, skewed data requires AI-based mitigation strategy"""

    diffs = []
    n = chunk_timestamps(merged_norm_timestamps, bounds)
    m = chunk_timestamps(merged_mis_timestamps, bounds)
    adjacency = get_user_adjacencies(adjaceny_matrix)
    for user_id, _ in enumerate(norm_MU):
        adjacent_users = adjacency[user_id]
        user_norm_exposure = calc_user_campaign_exposure(adjacent_users, n, None, 4)
        user_mis_exposure = calc_user_campaign_exposure(adjacent_users, m, None, 4)
        diff = user_norm_exposure - user_mis_exposure
        diffs.append(diff)

    sns.displot(diffs)
    plt.show()


def calc_network_fairness(merged_norm_timestamps, merged_mis_timestamps, bounds, adjaceny_matrix, norm_MU, balance_factor,
                          realization_id):
    """calculates network fairness by calculating each user ratio between normal and misinformation, and then obtain
    the sum of the pre-defined objective function over all ratios"""

    n = chunk_timestamps(merged_norm_timestamps, bounds)
    m = chunk_timestamps(merged_mis_timestamps, bounds)
    user_ratios = []
    adjacency = get_user_adjacencies(adjaceny_matrix)
    for user_id, _ in enumerate(norm_MU):
        adjacent_users = adjacency[user_id]
        t = calc_user_campaign_exposure(adjacent_users, n, None, realization_id)
        f = calc_user_campaign_exposure(adjacent_users, m, None, realization_id)
        user_ratios.append((1+t)/(1+f*balance_factor))

    obj = []
    for r in user_ratios:
        obj.append((1 - r)** 2)
    return np.sum(obj)


def run_avg_test():
    """calc average from each dist when
    applying LA or uniform or no intervention"""

    r_a = open('uniform_diffs.txt', 'r')
    r_b = open('la_diffs.txt', 'r')
    r_c = open('before_diffs.txt', 'r')

    a = []
    lines = r_a.readlines()
    for l in lines:
        a.append(float(l.split(',')[-1].strip('\n')))
    r_a.close()

    b = []
    lines = r_b.readlines()
    for l in lines:
        b.append(float(l.split(',')[-1].strip('\n')))
    r_b.close()

    c = []
    lines = r_c.readlines()
    for l in lines:
        c.append(float(l.split(',')[-1].strip('\n')))
    r_c.close()

    sns.kdeplot(a, shade=True, label='Uniform: M= ' + str(round(np.mean(a), 2)) + ', std= ' + str(round(np.std(a), 2)))
    sns.kdeplot(b, shade=True, label='LA: M= ' + str(round(np.mean(b), 2)) + ', std= ' + str(round(np.std(b), 2)))
    sns.kdeplot(c, shade=True, label='Before: M= ' + str(round(np.mean(c), 2)) + ', std= ' + str(round(np.std(c), 2)))

    plt.xlabel('Norm vs Mis Exposure Diffs')

    plt.legend()
    plt.savefig('interv_results/' + 'interv' + '.png')
    plt.close()


def create_3d_random_walk(user_i_id, user_j_id, user_k_id, user_i_x, user_j_x, user_k_x):
    """visualize three automata joint random walk example"""

    common_length = min([len(user_i_x), len(user_j_x), len(user_k_x)])

    x = np.array(user_i_x[:common_length])
    y = np.array(user_j_x[:common_length])
    z = np.array(user_k_x[:common_length])

    x = np.cumsum(x)  # The cumsum() function is used to get cumulative sum over a DataFrame or Series axis i.e.
    # it sums the steps across for eachaxis of the plane.
    y = np.cumsum(y)
    z = np.cumsum(z)
    plt.figure()
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.plot(x, y, z, alpha=0.9)  # alpha sets the darkness of the path.
    ax.scatter(x[-1], y[-1], z[-1])
    plt.savefig('./random_walks_figs/'+str(user_i_id) + '_'+str(user_j_id) +'_'+ str(user_k_id) + '.png')
    plt.cla()


def create_synthetic_network(skewness_level, users_count, end_time, synthetic_labels_path):
    """This method creates a synthetic misinformation, normal, influence data so we can control the misinformation
    skewness level in the synthesized data to test the performance of our algorithm"""

    s_mis_dict = dict()
    s_norm_dict = dict()
    tweeting_times = dict()
    loop_change_value = skewness_level * users_count
    counter = 0
    synthetic_file = open(synthetic_labels_path, 'a')

    norm_content_times = random.sample(range(0, end_time), random.randint(4, 8))

    influence_relations = dict()

    for user_id in range(users_count):
        if counter < loop_change_value:
            mis_content_times = random.sample(range(0, end_time), random.randint(9, 15))
        else:
            mis_content_times = random.sample(range(0, end_time), random.randint(1, 3))

        r_x = random.random()
        s_mis_dict.update({user_id: np.array(sorted(mis_content_times), dtype=float)})
        tweeting_times.update({user_id: [(str(user_id)+str(r_x)+str(tw_time), tw_time) for tw_time in sorted(mis_content_times)]})
        synthetic_tw_ids = [str(user_id)+str(r_x)+str(tw_time) for tw_time in sorted(mis_content_times)]
        for s_tw_id in synthetic_tw_ids:
            synthetic_file.write('false' + ':' +  s_tw_id  + ' \n')

        r_x = random.random()
        s_norm_dict.update({user_id: np.array(sorted(norm_content_times), dtype=float)})
        tweeting_times[user_id] += [(str(user_id) + str(r_x) + str(tw_time), tw_time) for tw_time in sorted(norm_content_times)]
        synthetic_tw_ids = [str(user_id) + str(r_x) + str(tw_time) for tw_time in sorted(norm_content_times)]
        for s_tw_id in synthetic_tw_ids:
            synthetic_file.write('true' + ':' + s_tw_id + '\n')

        counter += 1

    for user_id in range(users_count):
        influence_relations.update({user_id: [(sub_id, s_mis_dict[sub_id][0]) for sub_id in s_mis_dict.keys() if user_id != sub_id]})

    return s_mis_dict, s_norm_dict, tweeting_times, influence_relations


def predict_events(incentives, original_MU, A):
    """this method predicts the timestamps for an event when given the incentives that were learned by the Automata Network"""

def calc_pegypt_realizations_event_changes(original_timestamps, predicted_timestamps):
    """this method shows the trajectories of an event over the different time realizations.
     Once, when using intervention, and another when not"""


