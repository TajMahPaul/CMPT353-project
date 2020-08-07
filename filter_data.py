import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.signal import chirp, find_peaks, peak_widths
    
RAW_DATA_DIRECTORY= "./raw_data"

def low_pass_fliter(noisy_signal):
    b, a = signal.butter(3, .4, btype='lowpass', analog=False)
    return signal.filtfilt(b, a, noisy_signal)
    
def visualize(df_left, df_right):
    plt.plot(df_right['time'], df_right['a_mag'], df_right['time'], df_right['ax'],  df_right['time'], df_right['ay'], df_right['time'], df_right['az'])
    plt.grid(True)
    plt.show()
    
def get_magnitude_acc(df):
    return np.sqrt( np.square(df['ax'] )+ np.square(df['ay']) + np.square(df['az']))

def process_signal(df):
    df['a_mag'] = get_magnitude_acc(df)
    df['a_mag'] = low_pass_fliter(df['a_mag'])
    df['ax'] = low_pass_fliter(df['ax'])
    df['ay'] = low_pass_fliter(df['ay'])
    df['az'] = low_pass_fliter(df['az'])

    return df

def process_data(path):

    df_left_leg = pd.read_csv(RAW_DATA_DIRECTORY + "/" + path + "/left_leg.csv")
    df_right_leg = pd.read_csv(RAW_DATA_DIRECTORY + "/" + path + "/right_leg.csv")

    df_left_leg = process_signal(df_left_leg)
    df_right_leg = process_signal(df_right_leg)

    # visualize(df_left_leg, df_right_leg)
    df_left_leg.to_csv(RAW_DATA_DIRECTORY + "/" + path + "/filtered_left_leg.csv", index=False)
    df_right_leg.to_csv(RAW_DATA_DIRECTORY + "/" + path + "/filtered_right_leg.csv", index=False)

def create_data_frame(path):
    for subdirs, dirs, files in os.walk(path):
        for d in dirs:
            process_data(d)

def main():
    create_data_frame(RAW_DATA_DIRECTORY)


if __name__=='__main__':
    main()