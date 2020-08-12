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

RAW_DATA_DIRECTORY= "./raw_data"


 # function that get the mag distance / step
def get_distance_per_step(df):
    velocity = simps(df['signal'])
    time = df['time'].max() - df['time'].min() 
    distance = velocity * time
    distance = distance.astype(int)
    return distance

# funstion that get the magnitude velocity / step
def get_velocity_per_step(df):
    return simps(df['signal'])

# normalizes the x-axis from 0-1 and makes it so there are 50 evenly spaced points for a given peak (to make comparison of peaks easier with DWS distance algorithm)
def normalize_and_iterpolate():
    return


# returns the peaks with lowest DWS distance to every other peak (ie. the peak more like all the other peaks)
def filter_peaks(segPeaks, name):

    
    average_distances = []
    
    for i in range(len(segPeaks)):
        distances = []

        # normalize time from 0 - 1 for each peak and linear interpolation to have equal points and distance between points for good comparison
        normal_time_i = (segPeaks[i]['time']-segPeaks[i]['time'].min())/(segPeaks[i]['time'].max()-segPeaks[i]['time'].min())
        new_f_i = interp1d(normal_time_i, segPeaks[i][name])
        new_x_i = np.linspace(0, 1, num=50, endpoint=True)
        new_y_i = new_f_i(new_x_i)

        for j in range(len(segPeaks)):
            if(i != j):

                # normalize time from 0 - 1 for each peak and linear interpolation to have equal points and distance between points for good comparison
                normal_time_j = (segPeaks[j]['time']-segPeaks[j]['time'].min())/(segPeaks[j]['time'].max()-segPeaks[j]['time'].min())
                new_f_j = interp1d(normal_time_j, segPeaks[j][name])
                new_x_j = np.linspace(0, 1, num=50, endpoint=True)
                new_y_j = new_f_j(new_x_j)

                distance, path = fastdtw(new_y_i, new_y_j, dist=euclidean)
                distances.append(distance)

        avg = sum(distances) / len(distances)
        average_distances.append((i, avg))

    print(average_distances)
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

def get_single_leg_features(df, path):

    # get the local minimum points
    min_indices = get_min(df)
    
    # initalize data structures
    segmented_peaks = []
    
    n = len(min_indices)
    for i in range(n):

        single_peak_indices = []
        start_number = min_indices[i]

        if(i != n-1):
            end_number = min_indices[i+1]

        while (start_number <= end_number):
            single_peak_indices.append(start_number)
            start_number = start_number + 1
        
        # get the signal and normalize the name (convention over configuration)
        single_peak = df.loc[single_peak_indices][["time", name]]
        single_peak = df.rename(columns={name:'signal'})

        # filters out really small peaks
        if (single_peak[name].mean() > 0.1):

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
    step_velo = get_velocity_per_step(best_peak)
    step_dist = get_distance_per_step(best_peak)
    # TODO: get more features if time

    # convert list of means into pandas DataFrame
    df_list_mean = pd.DataFrame(data=list_of_means)

    # append the min, max, mean, median, variance, std to features['values']
    feature_values = [step_time, step_velo, step_dist]
    feature_labels = ['step_time', 'step_vel', 'ste_dis']
  
    return feature_values, feature_labels, best_peak
    
def process_filtered_data(path):
    
    # Create dataframe for features accross all people
    df_features = pd.DataFrame()

    # import the filtered data set
    df_left_leg = pd.read_csv(RAW_DATA_DIRECTORY + "/" + path + "/filtered_left_leg.csv", parse_dates=['time'])
    df_right_leg = pd.read_csv(RAW_DATA_DIRECTORY + "/" + path + "/filtered_right_leg.csv", parse_dates=['time'])
    
    # Get features for all signals
    columns = list(df_left_leg.columns)

    # for every signal (ax, ay, az, a_mag)
    for c in columns:
        if (c != "time"):
            # get features for right leg
            right_features, right_labels, right_best_peak_right_leg = get_single_leg_features(df_right_leg, path)
            right_labels = map(right_labels, lambda x: x + "_" + c + "_right_leg")
            
            # get features for left leg
            left_features, left_labels, best_peak_left_leg, = get_single_leg_features(df_left_leg, path)
            left_labels = map(left_labels, lambda x: x + "_" + c + "_right_leg")

            right_features.extend(left_features)
            right_labels.extent(left_labels)

            combined_features, combined_labels = fet_combined_leg_features(best_peak_left_leg, right_best_peak_right_leg)
            # TODO: remove me
            break
        

# start the feature extraction
def start(path):

    # every directory equates one person
    for subdirs, dirs, files in os.walk(path):
        for d in dirs:
            process_filtered_data(d)
            
            # TODO: remove me
            break
        break
    

def main():
    start(RAW_DATA_DIRECTORY)


if __name__=='__main__':
    main()