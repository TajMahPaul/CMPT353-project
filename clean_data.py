import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.signal import chirp, find_peaks, peak_widths

RAW_DATA_DIRECTORY= "./raw_data"

def guassian_filter(noisy_signal):
    return gaussian_filter(noisy_signal, sigma=3)

def low_pass_fliter(noisy_signal):
    b, a = signal.butter(3, .2, btype='lowpass', analog=False)
    return signal.filtfilt(b, a, noisy_signal)
    
def visualize(df):
    plt.plot(df_right['time'], df_right['a_mag'], df_left['time'], df_left['a_mag'])
    plt.grid(True)
    plt.show()
    
def get_magnitude_acc(df):
    return np.sqrt(np.square(df['ax']))

def process_signal(df):
    df['a_mag'] = get_magnitude_acc(df)
    df['a_mag'] = low_pass_fliter(df['a_mag'])
    df['a_mag'] = guassian_filter(df['a_mag'])
    peaks = find_peaks(df['a_mag'])
    results_full = peak_widths(df['a_mag'], peaks, rel_height=1)
    plt.hlines(*results_full[1:], color="C3")
    return df

def process_data(path):

    df_left_leg = pd.read_csv(RAW_DATA_DIRECTORY + "/" + path + "/left_leg.csv")
    df_right_leg = pd.read_csv(RAW_DATA_DIRECTORY + "/" + path + "/right_leg.csv")

    df_left_leg= process_signal(df_left_leg)
    # df_right_leg, peaks_right_leg = process_signal(df_right_leg)

    visualize(df_left_leg)

def create_data_frame(path):
    for subdirs, dirs, files in os.walk(path):
        for d in dirs:
            process_data(d)
            break
        break

def main():
    create_data_frame(RAW_DATA_DIRECTORY)


if __name__=='__main__':
    main()