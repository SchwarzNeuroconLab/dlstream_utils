"""
DeepLabStream utility code is meant to be used with DLStream
© J.Schweihoff
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/dlstream_utils
Licensed under GNU General Public License v3.0
"""


import pandas as pd
import numpy as np
from utils.dataframe_helper import load_dataset, get_unique_levels_clm, get_bodyparts



def convert_to_dlc_output(path, include_frames = False, number_of_animals = 1):
    df = load_dataset(path, header = [0,1,2], index_col = 0, sep = ';')

    for animal in range(1, number_of_animals+1):
        df_dlc = pd.DataFrame()
        df_to_dlc = df['Animal{}'.format(animal)]
        #get levels from df that contain bodyparts and get unique values:
        bodyparts = get_unique_levels_clm(df_to_dlc)
        index_start = df_to_dlc.first_valid_index()
        headline = 'DLStream_idx'+ str(index_start)
        for  bp in bodyparts:
            df_dlc[(headline, bp, 'x')]  =  df_to_dlc[(bp , 'x')]
            df_dlc[(headline, bp, 'y')] = df_to_dlc[(bp, 'y')]
            df_dlc[(headline, bp, 'likelihood')] = 1
        #set to normal index from 0
        df_dlc.reset_index(inplace= True)
        if include_frames:
            df_dlc.rename(columns={'Frame': ('DLStream', 'Frame', '')}, inplace= True)
        else:
            df_dlc.drop(columns=['Frame'], inplace=True)
            print('Dropped frames')

        df_dlc.columns = pd.MultiIndex.from_tuples(df_dlc.columns, names = ['scorer', 'bodyparts', 'coords'])
        return df_dlc

def batch_convert_to_DLC(folder_path):
    """Converts DLStream output file to DLC format csv file for convenient post analysis with already established scripts"""

    files = glob.glob(folder_path + '/' + '*.csv')

    print("Found {} files".format(len(files)))

    for num,file in enumerate(files):
        filename = os.path.basename(file)
        filename = filename.split('.')[0]
        output_filepath, _ = file.split('.')
        print('Processing file {}'.format(num + 1),'\n' + filename)
        df_dlc = convert_to_dlc_output(file)
        df_dlc.to_csv(output_filepath + '_dlc.csv', sep=",")



if __name__ == '__main__':

    path = r"PATH_TO_DLSTREAMOUTPUT.csv"
    df_dlc = convert_to_dlc_output(path)
    output_filepath,_ = path.split('.')
    df_dlc.to_csv(output_filepath + '_dlc.csv', sep=",")