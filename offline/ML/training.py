"""
Created on Sun Mar 17 09:03:30 2019
@author: marley
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
from sklearn.utils import shuffle
# from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
# from sklearn import model_selection
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
# import seaborn as sn
# from sklearn.utils.multiclass import unique_labels
# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy.fft as fft
import numpy as np
from scipy import stats
from scipy import signal

import pickle

import sys
sys.path.append('../utils')
from metadata import MARKER_DATA, LABELS, FILES_BY_SUBJECT, ALL_FILES, ELECTRODE_C3, ELECTRODE_C4
import file_utils

warnings.filterwarnings("ignore")

def epoch_data(data, window_length, shift, maxlen=2500):
    arr = []
    start = 0
    i = start
    maxlen = min(len(data), maxlen)
    while i + window_length < start + maxlen:
        arr.append(data[i:i+window_length])
        i += shift
    return np.array(arr)


def extract_features(all_data, window_s, shift, plot_psd=False, separate_trials=False, scale_by=None):
    all_psds = {label: [] for label in LABELS}
    all_features = {label: [] for label in LABELS}

    idx = 1
    for direction, data in all_data.items():
        for trial in data:
            epochs = epoch_data(trial, int(250 * window_s), int(shift*250))    # shape n x 500 x 2
            trial_features = []
            for epoch in epochs:
                features, freqs, psds_per_channel = get_features(epoch.T, scale_by=scale_by)
                psd_c3, psd_c4 = psds_per_channel[0] , psds_per_channel[-1]
                all_psds[direction].append([psd_c3, psd_c4])
                trial_features.append(features)

                # Sanity check: plot the psd
                if plot_psd:
                    plt.figure("psd")
                    plt.subplot(3, 2, idx)
                    plt.plot(freqs, psd_c3)
                    plt.ylim([0, 25])
                    plt.xlim([6, 20])
                    plt.subplot(3, 2, idx+1)
                    plt.plot(freqs, psd_c4)
                    plt.ylim([0, 25])
                    plt.xlim([6, 20])
            if trial_features:
                if separate_trials:
                    all_features[direction].append(np.array(trial_features))
                else:
                    all_features[direction].extend(trial_features)
        idx += 2
    return all_psds, all_features, freqs


def to_feature_vec(all_features, rest=False):
    feature_arr = []
    for direction, features in all_features.items():
        features = np.array(features)
        arr = np.hstack((features, np.full([features.shape[0], 1], LABELS.index(direction))))
        feature_arr.append(arr)
    if not rest or not len(features):
        feature_arr = feature_arr[:-1]
    return np.vstack(feature_arr)


def normalize(features_dict):
    """ Divide features by mean per channel """
    all_features = to_feature_vec(features_dict, rest=True)
    av = list(np.mean(all_features, axis=0))
    mean_coeff = np.array([el/sum(av[:-1]) for el in av[:-1]])
    for direction, features in features_dict.items():
        features = [np.divide(example, mean_coeff) for example in features]
        features_dict[direction] = features

def running_mean(x, N):
   cumsum = np.cumsum(np.insert(x, 0, 0))
   return (cumsum[N:] - cumsum[:-N]) / N


def evaluate_models(X, Y, X_test, Y_test, models):
    """ Evaluate test accuracy of all models in a list of models

    Args:
        X : array of features (train)
        Y : array of labels (train)
        X_test : array of features (test)
        Y_test : array of labels (test)
        models : list of (name, model) tuples

    Returns:
        val_results : array of accuracies, shape num_models
    """
    test_results = []
    for name, model in models.items():
        model.fit(X, Y)
        score = model.score(X_test, Y_test)
        test_results.append(score)
        msg = "%s: %f" % (name, score)
        print(msg)
    return np.array(test_results)

def get_features(arr, channels=[ELECTRODE_C3, ELECTRODE_C4], scale_by=None):
    """ Get features from single window of EEG data

    Args:
        arr : data of shape num_channels x timepoints

    Returns:
        features : array with feature values
        freqs : array of frequencies in Hz
        psds_per_channel : array with full psd spectrum, shape num_channels x num_freqs
    """

    psds_per_channel = []
    nfft = 500
    if arr.shape[-1] < 500:
        nfft = 250
    for ch in arr[channels]:
        #freqs,psd = signal.periodogram(np.squeeze(ch),fs=250, nfft=500, detrend='constant')
        psd, freqs = mlab.psd(np.squeeze(ch),NFFT=nfft,Fs=250)
        psds_per_channel.append(psd)
    psds_per_channel = np.array(psds_per_channel)
    mu_indices = np.where(np.logical_and(freqs >= 10, freqs <= 12))

    #features = np.amax(psds_per_channel[:,mu_indices], axis=-1).flatten()   # max of 10-12hz as feature
    features = np.mean(psds_per_channel[:, mu_indices], axis=-1).flatten()   # mean of 10-12hz as feature
    if scale_by:
        scale_indices = np.where(np.logical_and(freqs >= scale_by[0], freqs <= scale_by[-1]))
        scales = np.mean(psds_per_channel[:,scale_indices],axis=-1).flatten()
        temp.append(scales)
        features = np.divide(features, scales)
    #features = np.array([features[:2].mean(), features[2:].mean()])
    # features = psds_per_channel[:,mu_indices].flatten()                     # all of 10-12hz as feature
    return features, freqs, psds_per_channel


if __name__ == "__main__":
    shift = 0.25
    plot_psd = True
    tmin, tmax = 0, 0
    normalize_spectra = True
    run_pca = False
    scale_by = None

    # Load data
    dataset = file_utils.load_all()
    subjects = [i for i in range(len(FILES_BY_SUBJECT))]             # index of the test files we want to use

    # Test options and evaluation metric
    scoring = 'accuracy'
    validation = True
    test = True
    seed = 7
    models = {}
    # models.append(('LR', LogisticRegression(solver='lbfgs')))
    models['LDA'] = LinearDiscriminantAnalysis()
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB(var_smoothing=0.001)))
    # models.append(('SVM', SVC(gamma='scale')))

    # Perform leave-one-subject-out cross-validation for each subject
    # Delivers accuracy by subject, then by window size, then by session, then by model
    for subj in subjects:
        test_csvs = FILES_BY_SUBJECT[subj]
        print(test_csvs)
        train_csvs = test_csvs # [el for el in ALL_FILES if el not in test_csvs]
        print(train_csvs)
        train_data = file_utils.merge_all_dols([dataset[csv] for csv in train_csvs])
        print(train_data)

        # Print subject name
        print(test_csvs[0].split('/')[1])

        window_s = 2
        train_psds, train_features, freqs = extract_features(train_data, window_s, shift, plot_psd, scale_by=scale_by)
        data = to_feature_vec(train_features, rest=False)

        # X, Y for training
        # For testing: X_test, Y_test
        X = data[:, :-1]
        Y = data[:, -1]
        X, Y = shuffle(X, Y, random_state=seed)

        # if run_pca:
        #     pca = PCA(n_components=2, svd_solver='full')
        #     pca.fit(X)
        #     X = pca.transform(X)

        for csv in test_csvs:
            test_data = dataset[csv]
            _, test_features, _ = extract_features(test_data, window_s, shift, plot_psd, scale_by=scale_by)

            if normalize_spectra:
                normalize(test_features)

            test_data = to_feature_vec(test_features)
            X_test = test_data[:, :-1]
            Y_test = test_data[:, -1]

            # if run_pca:
            #     X_test = pca.transform(X_test)

            test_results = evaluate_models(X, Y, X_test, Y_test, models)
            print("average accuracy: {:2.1f}".format(test_results.mean() * 100))

    with open("/Users/hal/Desktop/model_lda.pkl", 'wb') as pickled_file:
        pickle.dump(models['LDA'], pickled_file)
