import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks, peak_widths

RAW_DATA_DIRECTORY= "./raw_data"

def get_peaks(x, y):
    peaks, _ = find_peaks(y, height=.1)
    plt.plot(x,y)
    plt.plot(peaks, y[peaks], "x")
    

def get_features(x, y, name):
    peaks = get_peaks(x,y)


def process_data(path):
    df_features = pd.DataFrame()

    df_left_leg = pd.read_csv(RAW_DATA_DIRECTORY + "/" + path + "/filtered_left_leg.csv")
    df_right_leg = pd.read_csv(RAW_DATA_DIRECTORY + "/" + path + "/filtered_right_leg.csv")
    
    columns = ['ax', 'ay', 'az', 'a_mag']
    for column in columns:
        get_features(df_right_leg['time'],df_right_leg['a_mag'], 'a_mag')
        get_features(df_right_leg['time'],df_right_leg['az'], 'az')
        get_features(df_right_leg['time'],df_right_leg['ay'], 'ay')
        get_features(df_right_leg['time'],df_right_leg['ax'], 'ax')
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