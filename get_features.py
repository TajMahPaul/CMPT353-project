import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks, peak_widths, argrelextrema
from datetime import datetime
import statistics

RAW_DATA_DIRECTORY= "./raw_data"

# def get_distance_per_step():


# def get_velocity_per_step():

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
    # plt.show()

    list_of_means = []
    features = []
   
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
            single_peak['time'] = get_elapse_time(single_peak['time'])
            plt.plot(single_peak['time'], single_peak[name])

            mean = single_peak[name].mean()
            list_of_means.append(mean)

        single_peak_indices.clear()

    plt.show()

    # convert list of means into pandas DataFrame
    df_list_mean = pd.DataFrame(data=list_of_means)

    # append to features the min, max, mean, median, variance, std
    features.append(df_list_mean.min())
    features.append(df_list_mean.max())
    features.append(df_list_mean.mean())
    features.append(df_list_mean.median())
    features.append(df_list_mean.var())
    features.append(df_list_mean.std())

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

        
        df_right_leg_feature = pd.DataFrame(data=features, columns=['min', 'max', 'mean', 'median', 'variance', 'std'])
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