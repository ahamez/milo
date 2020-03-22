#!/usr/bin/env python3
"""
Created on Mon Aug  5 18:30:48 2019

@author: marley
"""
import pandas as pd
import numpy as np

from metadata import LABELS

def merge_dols(dol1, dol2):
    keys = set(dol1).union(dol2)
    no = []
    return dict((k, dol1.get(k, no) + dol2.get(k, no)) for k in keys)

def merge_all_dols(arr):
    all_data = {'Right': [], 'Left': [], 'Rest': []}
    for dol in arr:
        all_data = merge_dols(all_data, dol)
    return all_data

def load_openbci_raw(path):
    DATA_COLUMNS = (1, 2, 3, 4, 5, 6, 7, 8, 13)

    data = np.loadtxt(path,
                      delimiter=',',
                      skiprows=7, # skip headers
                      usecols=DATA_COLUMNS)
    eeg = data[:, :-1]
    timestamps = data[:, -1]
    return eeg, timestamps


def load_data(csv, raw):
    print("loading " + csv)

    data = {label: [] for label in LABELS}

    df = pd.read_csv(csv)
    eeg, timestamps = load_openbci_raw(raw)

    prev = 0
    prev_direction = df['Direction'][prev]
    for idx, el in enumerate(df['Direction']):
        if el != prev_direction or idx == len(df.index) - 1:
            start = df['Time'][prev]
            end = df['Time'][idx]
            indices = np.where(
                np.logical_and(
                    timestamps >= start,
                    timestamps <= end))
            trial = eeg[indices]
            data[prev_direction].append(trial)
            prev = idx
            prev_direction = el

    return data


def load_all(all_files):
    return {csv: load_data(csv, raw) for csv, raw in all_files.items()}
