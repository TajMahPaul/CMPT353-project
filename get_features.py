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

def get_distance_per_step(df):
    velocity = simps(df['a_mag'])
    time = df['time'].max() - df['time'].min() 
    distance = velocity * time
    distance = distance.astype(int)
    return distance
    

def get_velocity_per_step(df):
    return simps(df['a_mag'])

def filter_peaks(segPeaks, name):

    # returns the peaks with lowest DWS distance to every other peak (ie. the peak more like all the other peaks)
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

def get_peaks(x, y):
    peaks, _ = find_peaks(y, height=.5)
    # plt.plot(x,y)
    # plt.plot(peaks, y[peaks], "x")
    return peaks

def get_width(y, peaks):
    width = peak_widths(y, peaks, rel_height=0.9)
    # plt.hlines(*width[1:], color="C2")
    return width

def get_min(df):
    df['min'] = df.iloc[argrelextrema(df.a_mag.values, np.less_equal, order=5)[0]]['a_mag']
    # plt.scatter(df.index,df['min'], c='r')

    # returns indices of min values
    return df[df['min'].notnull()].index.tolist()

def get_features(df, name, path):
    peaks = get_peaks(df['time'],df[name])
    width = get_width(df[name], peaks)
    min_indices = get_min(df)
    
    segmented_peaks = []

    list_of_means = []
    features = {
        "labels": [],
        "values": []
    }
    
    n = len(min_indices)
    single_peak_indices = []
    for i in range(n):
        start_number = min_indices[i]
        if(i != n-1):
            end_number = min_indices[i+1]

        while (start_number <= end_number):
            single_peak_indices.append(start_number)
            start_number = start_number + 1
        
        single_peak = df.loc[single_peak_indices]
        
        # filters out small peaks
        if (single_peak[name].mean() > 0.1):

            # get elapse time 
            single_peak['time'] = get_elapse_time(single_peak['time'])

            segmented_peaks.append(single_peak)

            mean = single_peak[name].mean()
            list_of_means.append(mean)

        single_peak_indices.clear()

    best_peak = filter_peaks(segmented_peaks, name)
    plt.plot(best_peak['time'], best_peak['a_mag'])
    plt.show()

    # convert list of means into pandas DataFrame
    df_list_mean = pd.DataFrame(data=list_of_means)

    # append to features the min, max, mean, median, variance, std
    features.values.append(df_list_mean.min())
    features.values.append(df_list_mean.max())
    features.values.append(df_list_mean.mean())
    features.values.append(df_list_mean.median())
    features.values.append(df_list_mean.var())
    features.values.append(df_list_mean.std())

     # append to features the min, max, mean, median, variance, std
    features.labels.append("min")
    features.labels.append("max")
    features.labels.append("mean")
    features.labels.append("median")
    features.labels.append("var")
    features.labels.append("std")

    # convert feature list to np array to transpose
    features_np = np.array(features)
    features_np = features_np.transpose()
 
    return features_np
    

def get_elapse_time(timeSeries):
    if(timeSeries.size > 0):
        min_time = timeSeries.iloc[0]
        timeSeries = timeSeries.apply(lambda x: x - min_time)
    return timeSeries

def process_data(path):
    df_features = pd.DataFrame()

    df_left_leg = pd.read_csv(RAW_DATA_DIRECTORY + "/" + path + "/filtered_left_leg.csv", parse_dates=['time'])
    df_right_leg = pd.read_csv(RAW_DATA_DIRECTORY + "/" + path + "/filtered_right_leg.csv", parse_dates=['time'])
    
    # # Convert time to elapsed time
    # df_left_leg['time'] = get_elapse_time(df_left_leg['time'])
    # df_right_leg['time'] = get_elapse_time(df_right_leg['time'])
    
    columns = ['ax', 'ay', 'az', 'a_mag']
    for column in columns:
        features = get_features(df_right_leg, 'a_mag', path)
        # get_features(df_left_leg, 'a_mag')

        
        df_right_leg_feature = pd.DataFrame(data=features.values, columns= features.labels)
        print(df_right_leg_feature)

        break

    
    

def get_people_data(path):
    for subdirs, dirs, files in os.walk(path):
        for d in dirs:
            process_data(d)
            break
        break
    

def main():
    get_people_data(RAW_DATA_DIRECTORY)


if __name__=='__main__':
    main()