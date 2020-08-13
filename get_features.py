import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks, peak_widths, argrelextrema
from datetime import datetime
import statistics
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.signal import correlate
import json

RAW_DATA_DIRECTORY= "./raw_data"


 # function that get the mag distance / step
def get_distance_per_step(df):
    velocity = simps(df['signal'])
    time = df['time'].max() - df['time'].min()
    time = time.total_seconds()
    distance = velocity * time
    return distance

# funstion that get the magnitude velocity / step
def get_velocity_per_step(df):
    return simps(df['signal'])

# normalizes the x-axis from 0-1 and makes it so there are 50 evenly spaced points for a given peak (to make comparison of peaks easier with DWS distance algorithm)
def normalize_and_iterpolate(peak):
    normal_time = (peak['time']-peak['time'].min())/(peak['time'].max()-peak['time'].min())
    new_f = interp1d(normal_time, peak['signal'])
    new_x = np.linspace(0, 1, num=50, endpoint=True)
    new_y = new_f(new_x)
    return new_x, new_y

# returns the peaks with lowest DWS distance to every other peak (ie. the peak more like all the other peaks)
def filter_peaks(segPeaks):

    
    average_distances = []
    
    for i in range(len(segPeaks)):
        distances = []

        # normalize time from 0 - 1 for each peak and linear interpolation to have equal points and distance between points for good comparison
        new_x_i, new_y_i = normalize_and_iterpolate(segPeaks[i])

        for j in range(len(segPeaks)):
            if(i != j):

                # normalize time from 0 - 1 for each peak and linear interpolation to have equal points and distance between points for good comparison
                new_x_j, new_y_j = normalize_and_iterpolate(segPeaks[j])

                distance, path = fastdtw(new_y_i, new_y_j, dist=euclidean)
                distances.append(distance)

        avg = sum(distances) / len(distances)
        average_distances.append((i, avg))

    return segPeaks[min(average_distances, key = lambda t: t[1])[0]]

# function to return local minimum points for a given spectrum
def get_min(df):
    df['min'] = df.iloc[argrelextrema(df.a_mag.values, np.less_equal, order=5)[0]]['a_mag']

    # returns indices of min values
    return df[df['min'].notnull()].index.tolist()

# function to converted timestamps to elapse time
def get_elapse_time(timeSeries):
    if(timeSeries.size > 0):
        min_time = timeSeries.iloc[0]
        timeSeries = timeSeries.apply(lambda x: x - min_time)
    return timeSeries

# function to get features the measure the asymmetry between legs
def fet_combined_leg_features(lPeak, rPeak):
    right_x, right_y = normalize_and_iterpolate(rPeak)
    left_x, left_y = normalize_and_iterpolate(lPeak)


    distance, path = fastdtw(right_y, left_y, dist=euclidean)
    xcorr = correlate(right_y, left_y)


    feature_values = [distance]
    feature_labels = ['dtw_distance']
    
    # return features and coloumn labels
    return feature_values, feature_labels

def get_single_leg_features(df, name):

    # get the local minimum points
    min_indices = get_min(df)
    
    # initalize data structures
    segmented_peaks = []
    
    # set n to the number of min indices found and iterate through the min_indices
    n = len(min_indices)
    for i in range(n):

        # takes 2 min indices adjacent to each other
        # start_number is the start index and end_number is the index directly
        # to the right of start index in the min_indices list
        # the values between start number and end number represent 1 peak
        single_peak_indices = []
        start_number = min_indices[i]

        # error condition: when start_number is at last indices
        # there is no next indice to its right to compare with
        if(i != n-1):
            end_number = min_indices[i+1]

        # gets the indices of values between start_number and end_number
        # and saves it into single_peak_indices
        while (start_number <= end_number):
            single_peak_indices.append(start_number)
            start_number = start_number + 1
        
        # get the signal and normalize the name (convention over configuration)
        single_peak = df.loc[single_peak_indices][["time", name]]
        single_peak = single_peak.rename(columns={name:'signal'})

        # filters out really small peaks
        if (single_peak['signal'].mean() > 0.1):

            # conver time to elapse time 
            single_peak['time'] = get_elapse_time(single_peak['time'])

            # add to our collection of peaks in this signal
            segmented_peaks.append(single_peak)

    # return the best peak of this signal
    best_peak = filter_peaks(segmented_peaks)
    # plt.plot(best_peak['time'], best_peak['a_mag'])
    # plt.show()

    # create features from best peak
    step_time = best_peak["time"].max() - best_peak["time"].min()
    step_time = step_time.total_seconds()
    step_velo = get_velocity_per_step(best_peak)
    step_dist = get_distance_per_step(best_peak)
    step_min  = best_peak['signal'].min()
    step_max  = best_peak['signal'].max()
    step_mean = best_peak['signal'].mean()
    step_var  = best_peak['signal'].var()
    step_std  = best_peak['signal'].std()
    step_skew = best_peak['signal'].skew()

    # TODO: get more features if time

    # return features and coloumn labels
    feature_values = [step_time, step_velo, step_dist, step_min, step_max, step_mean, step_var, step_std, step_skew]
    feature_labels = ['step_time', 'step_vel', 'step_dis', 'step_min', 'step_max', 'step_mean', 'step_var', 'step_std', 'step_skew']
  
    return feature_values, feature_labels, best_peak
    
def process_filtered_data(path):

    # import the filtered data set
    df_left_leg = pd.read_csv(RAW_DATA_DIRECTORY + "/" + path + "/filtered_left_leg.csv", parse_dates=['time'])
    df_right_leg = pd.read_csv(RAW_DATA_DIRECTORY + "/" + path + "/filtered_right_leg.csv", parse_dates=['time'])
    
    # Get features for all signals
    columns = list(df_left_leg.columns)

    all_features = []
    all_col_labels = []
    # for every signal (ax, ay, az, a_mag)
    for c in columns:

        # lets only consider magnitude of acceleration for simplicity
        if (c == "a_mag"):
            # get features for right leg
            right_features, right_labels, right_best_peak_right_leg = get_single_leg_features(df_right_leg, c)
            right_labels = list(map(lambda x: x + "_" + c + "_right_leg", right_labels))
            
            # get features for left leg
            left_features, left_labels, best_peak_left_leg, = get_single_leg_features(df_left_leg, c)
            left_labels = list(map(lambda x: x + "_" + c + "_right_leg", left_labels))

            # get features for that measure the asymmetry of the legs
            combined_features, combined_labels = fet_combined_leg_features(best_peak_left_leg, right_best_peak_right_leg)
            combined_labels = list(map(lambda x: x + "_" + c + "_asymm", combined_labels))

            all_features = all_features + right_features + left_features + combined_features
            all_col_labels = all_col_labels + right_labels + left_labels + combined_labels
            
    return all_features, all_col_labels

# get the injury status for the person
def get_person_data(path):
    person_id = int(path.replace("test-subject-", ""))
    with open(RAW_DATA_DIRECTORY + "/" + path + "/person.json") as json_file:
        data = json.load(json_file)
        person_injury = data['has_injury']
        if (person_injury == 'false'):
            person_injury = 0
        else:
            person_injury = 1
        return person_id, person_injury

# start the feature extraction
def start(path):

    final_df = pd.DataFrame()

    # every directory equates one person
    for subdirs, dirs, files in os.walk(path):
        for d in dirs:

            # get person data and features
            person_id, has_injury = get_person_data(d)
            features, col_labels = process_filtered_data(d)

            features.append(has_injury)
            col_labels.append("has_injury")

            if(len(final_df.columns) == 0):
                final_df = pd.concat([final_df, pd.DataFrame(columns=col_labels)])

            final_df.loc[person_id] = features 
        final_df.to_csv("features.csv", index=False)

def main():
    start(RAW_DATA_DIRECTORY)


if __name__=='__main__':
    main()