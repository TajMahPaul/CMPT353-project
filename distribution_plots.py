import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.signal import chirp, find_peaks, peak_widths
import json
    
RAW_DATA_DIRECTORY= "./raw_data"
FIG_DATA_DIRECOTRY="./figures"

json_list = []

def create_data_frame(path):
    for subdirs, dirs, files in os.walk(path):
        for d in dirs:
            read_json_file(d)

def read_json_file(path):
    with open(RAW_DATA_DIRECTORY + "/" + path + "/person.json") as json_file:
        data = json.load(json_file)
    json_list.append(data)

def create_distribution(df):

    df['age'] = df['age'].astype(int)
    df['weight'] = df['weight'].astype(int)
    
    plt.hist(df['age'], bins=15)
    plt.title("Age Distribution")
    plt.xlabel("Age (Years)")
    plt.ylabel("Number")
    plt.savefig(FIG_DATA_DIRECOTRY + 'age_distribution.png') 
    plt.clf()


    plt.hist(df['gender'], bins=2)
    plt.title("Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Number")
    plt.savefig(FIG_DATA_DIRECOTRY + 'gender_distribution.png') 
    plt.clf()

    plt.hist(df['has_injury'], bins=2)
    plt.title("Injury Distribution")
    plt.xlabel("Injury")
    plt.ylabel("Number")
    plt.savefig(FIG_DATA_DIRECOTRY + 'injury_distribution.png') 
    plt.clf()

    plt.hist(df['height'], bins=9)
    plt.title("Height Distribution")
    plt.xlabel("Height (Feet)")
    plt.ylabel("Number")
    plt.savefig(FIG_DATA_DIRECOTRY + 'height_distribution.png') 
    plt.clf()

    plt.hist(df['weight'], bins=20)
    plt.title("Weight Distribution")
    plt.xlabel("Weight (lbs)")
    plt.ylabel("Number")
    plt.xticks(rotation=-90)
    plt.savefig(FIG_DATA_DIRECOTRY + 'weight_distribution.png') 
    plt.clf()
    

def main():
    create_data_frame(RAW_DATA_DIRECTORY)

    df = pd.DataFrame(data=json_list)
    create_distribution(df)


if __name__=='__main__':
    main()