'''
Evolutionary K means for Hessian matrix
Created By Jiajia Li
01/10/2018
'''

import numpy as np
from sklearn.cluster import KMeans
from numpy import linalg as LA
import random


class hessian_parition(object):

    def __init__(self, alpha, eta, K, m):
        self.alpha = alpha
        self.eta = eta
        self.K = K
        self.m = m
        self.smooth_hessian = np.zeros((m, m))
        self.label_pred = []
        self.centroids = []
        self.t = 0


    def clustering_match(self, set1, smooth_hessian_raw, label_pred_new, index_set_new):

        disc = []
        for item in index_set_new:
            set2 = smooth_hessian_raw[label_pred_new == item, :]
            discrepancy_m = np.tile(set1[:, np.newaxis,:], [1, set2.shape[0], 1]) - \
                          np.tile(set2[np.newaxis, :, :], [set1.shape[0], 1, 1])
            discrepancy = np.sum(LA.norm(discrepancy_m, axis=2))
            disc.append(discrepancy)
        index = index_set_new[np.argmin(np.array(disc))]
        return index

    def greedy_approximation(self, smooth_hessian_new, label_pred_new):

        index_set = random.sample(range(self.K), self.K)
        index_set_new = random.sample(range(self.K), self.K)
        matching_arr = np.zeros([self.m, 2], dtype=np.int32)
        i = 0
        for item in index_set:
            set1 = self.smooth_hessian[self.label_pred == item, :]  # [none, m]
            index = self.clustering_match(set1, smooth_hessian_new, label_pred_new, index_set_new)
            matching_arr[i, :] = np.array([item, index], dtype=np.int32)
            index_set_new.remove(index)
            i = i + 1
        return matching_arr

    def one_step_kmeans(self, centroids_new, affinity_m):
        disc_m = np.tile(centroids_new[:, :, np.newaxis], [1, 1, self.m]) - \
                 np.tile(affinity_m[np.newaxis, :, :], [self.K, 1, 1])
        label_pred_new = np.argmin(LA.norm(disc_m, axis=2), axis=0)
        return label_pred_new

    def evolutionary_kmeans(self, hessian):

        if self.t == 0:
            affinity_m = abs(hessian)
            estimator.fit(affinity_m)
            self.label_pred = estimator.labels_
            self.centroids = estimator.cluster_centers_
            self.smooth_hessian = affinity_m
            self.t = self.t + 1

        else:
            smooth_hessian_new = (1 - self.alpha) * abs(hessian) + self.alpha * self.smooth_hessian
            affinity_m = smooth_hessian_new
            estimator.fit(affinity_m)
            label_pred_new = estimator.labels_
            centroids_new = estimator.cluster_centers_
            matching_arr = self.greedy_approximation(smooth_hessian_new, label_pred_new)
            self.t = self.t + 1
            self.smooth_hessian = smooth_hessian_new
            for i in range(self.K):
                n_new = np.sum([label_pred_new == matching_arr[i, :][1]])
                n = np.sum([self.label_pred == matching_arr[i, :][0]])
                gamma_t = n_new / (n_new + n)
                centroids_new[matching_arr[i, :][1]] = (1 - gamma_t) * eta * self.centroids[matching_arr[i, :][0]] + \
                                                       (1 - eta) * gamma_t * centroids_new[matching_arr[i, :][1]]
                label_pred_new = self.one_step_kmeans(centroids_new, affinity_m)
            self.label_pred = label_pred_new
            self.centroids = centroids_new

### Test Data ################
if __name__=='__main__':
    '''
    hyperparameter tuning and default value 
    '''
    alpha = 0.
    eta = 0.1
    m = 6
    K = 2
    estimator = KMeans(n_clusters=K,
                       init='k-means++',
                       n_init=10,
                       max_iter=300,
                       tol=0.0001,
                       precompute_distances='auto',
                       verbose=0,
                       random_state=None,
                       copy_x=True,
                       n_jobs=1,
                       algorithm='auto')
    simulation_data = np.load("doublehopper01_6.npy")
    iteration = simulation_data.shape[0]
    hessian_para = hessian_parition(alpha, eta, K, m)
    for i in range(iteration):
        hessian_para.evolutionary_kmeans(simulation_data[i, :, :])
        print('Iteration' + str(i) + ":",hessian_para.label_pred)
