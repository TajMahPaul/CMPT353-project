import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks, peak_widths, argrelextrema
from datetime import datetime
import statistics

RAW_DATA_DIRECTORY= "./raw_data"

list_of_means = []

def get_peaks(x, y):
    peaks, _ = find_peaks(y, height=.5)
    plt.plot(x,y)
    plt.plot(peaks, y[peaks], "x")
    return peaks

def get_width(y, peaks):
    width = peak_widths(y, peaks, rel_height=0.9)
    plt.hlines(*width[1:], color="C2")
    return width

def get_min(df):
    df['min'] = df.iloc[argrelextrema(df.a_mag.values, np.less_equal, order=5)[0]]['a_mag']
    plt.scatter(df.index,df['min'], c='r')

    # returns indices of min values
    return df[df['min'].notnull()].index.tolist()

def get_features(df, name):
    peaks = get_peaks(df['time'],df[name])
    width = get_width(df[name], peaks)
    min_indices = get_min(df)
    plt.show()
   
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
            x = list(range(len(single_peak)))
            plt.plot(x, single_peak[name])

            mean = single_peak[name].mean()
            list_of_means.append(mean)

        single_peak_indices.clear()

    plt.show()
   
    final_mean = statistics.mean(list_of_means)
    print("Average magnitude:" , final_mean)
    list_of_means.clear()
    


def process_data(path):
    df_features = pd.DataFrame()

    df_left_leg = pd.read_csv(RAW_DATA_DIRECTORY + "/" + path + "/filtered_left_leg.csv")
    df_right_leg = pd.read_csv(RAW_DATA_DIRECTORY + "/" + path + "/filtered_right_leg.csv")
    
    columns = ['ax', 'ay', 'az', 'a_mag']
    for column in columns:
        get_features(df_right_leg, 'a_mag')
        get_features(df_left_leg, 'a_mag')
        
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