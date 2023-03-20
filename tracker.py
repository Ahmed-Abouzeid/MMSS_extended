import numpy as np
from utils import chunk_timestamps


class Tracker(object):
    def __init__(self, user_ids, realization_bounds, bias_info_org, social_circles_info_org, fixed_preds):

        self.users = user_ids
        self.realization_bounds = realization_bounds
        self.bias_info = bias_info_org
        self.circles_info_org = social_circles_info_org
        self.fixed_preds = fixed_preds
        self.bias_users_probs = dict()
        self.social_circles_users_probs = dict()


        self.initialize_probs()


    def initialize_probs(self):
        for e, id in enumerate(self.users):
            ev_count_against = np.sum([len(l) for l in chunk_timestamps(self.bias_info[0], self.realization_bounds)[e][:-1]])
            ev_count_for = np.sum([len(l) for l in chunk_timestamps(self.bias_info[1], self.realization_bounds)[e][:-1]])
            ev_count_for += len(self.fixed_preds[0][e])
            totals = ev_count_against + ev_count_for

            self.bias_users_probs.update({id:[ev_count_against, totals]})

            ev_count_A = np.sum([len(l) for l in chunk_timestamps(self.circles_info_org[0], self.realization_bounds)[e][:-1]])
            ev_count_A += len(self.fixed_preds[1][e])
            ev_count_B = np.sum([len(l) for l in chunk_timestamps(self.circles_info_org[1], self.realization_bounds)[e][:-1]])
            ev_count_C = np.sum([len(l) for l in chunk_timestamps(self.circles_info_org[2], self.realization_bounds)[e][:-1]])
            ev_count_C += len(self.fixed_preds[2][e])
            ev_count_E = np.sum([len(l) for l in chunk_timestamps(self.circles_info_org[3], self.realization_bounds)[e][:-1]])
            ev_count_E += len(self.fixed_preds[3][e])
            ev_count_F = np.sum([len(l) for l in chunk_timestamps(self.circles_info_org[4], self.realization_bounds)[e][:-1]])
            ev_count_F += len(self.fixed_preds[4][e])

            c = ev_count_B
            totals = ev_count_A +  ev_count_B + ev_count_C + ev_count_E +  ev_count_F

            self.social_circles_users_probs.update({id: [c, totals]})
