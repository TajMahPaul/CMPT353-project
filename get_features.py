import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks, peak_widths

RAW_DATA_DIRECTORY= "./raw_data"

def get_peaks(x, y):
    peaks, _ = find_peaks(y, height=.5)
    plt.plot(x,y)
    plt.plot(peaks, y[peaks], "x")
    return peaks
    

def get_features(df, name):
    peaks = get_peaks(df['time'],df['a_mag'])
    
    # Filters the df to show only data where the peaks are
    df_peaks = df.loc[peaks]
    print(df_peaks)

    # should be the value from the next row, then drops last value because after it's shifted it will be null
    df_shifted = df[name].shift(periods=-1) 
    df_shifted = df_shifted[df_shifted.notnull()]

    # TODO: Filters peaks that are close together in a_mag
    

    # TODO: find the distance between 2 points in a_mag


def process_data(path):
    df_features = pd.DataFrame()

    df_left_leg = pd.read_csv(RAW_DATA_DIRECTORY + "/" + path + "/filtered_left_leg.csv")
    df_right_leg = pd.read_csv(RAW_DATA_DIRECTORY + "/" + path + "/filtered_right_leg.csv")
    
    columns = ['ax', 'ay', 'az', 'a_mag']
    for column in columns:
        get_features(df_right_leg, 'a_mag')
        
        # get_features(df_right_leg['time'],df_right_leg['a_mag'], 'a_mag')
        # get_features(df_right_leg['time'],df_right_leg['az'], 'az')
        # get_features(df_right_leg['time'],df_right_leg['ay'], 'ay')
        # get_features(df_right_leg['time'],df_right_leg['ax'], 'ax')
        plt.show()
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