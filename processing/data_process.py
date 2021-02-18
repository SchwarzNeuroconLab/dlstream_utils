"""
DeepLabStream utility code is meant to be used with DLStream
© J.Schweihoff
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/dlstream_utils
Licensed under GNU General Public License v3.0
"""


import pandas as pd
import numpy as np
from utils.dataframe_helper import remove_unnamed_lvl_multiindex, flatten_multiindex, load_dataset, label_block,\
    convert_bodyparts_to_column_multiindex, get_bodyparts



def adjust_dtype_according_to_column_list(df: pd.DataFrame,dtype_str,column_list: list):
    """ Adjusts data type of dataframe columns according to dtype_str. Takes list of column names as input"""
    for col in column_list:
        df[col] = df[col].astype(dtype_str)


def adjust_dtype_according_to_dict(df: pd.DataFrame,dtype_dict: dict, report = False):
    """ Adjusts data type of dataframe columns according to dtype_str. Takes dict with column names as keys as input"""
    if report:
        print('Evaluating dtypes now...')
    for col,dtype_str in dtype_dict.items():
        if df[col].dtype != dtype_str:
            df[col] = df[col].astype(dtype_str)
            if report:
                print('Converted {} to type {}.'.format(col,dtype_str))
        else:
            if report:
                print('{} was already type {}.'.format(col,dtype_str))


def convert_empty_to_nan(df,column_list=None, report = False):
    """Fixes import issue where empty entries are not considered NaN in columns given by column_list.
    If column_list is None, takes whole df."""
    if column_list is not None:
        if report:
            print('Found {} empty entries in the columns {}. Converting to NaN...'.format
                  (df[column_list].isnull().sum().sum(),', '.join(map(str,column_list))))
        df = df[column_list].replace('',np.nan)
    else:
        if report:
            print('Found {} empty entries. Converting to NaN...'.format(df.isnull().sum().sum()))
        df = df.replace('',np.nan)
    return df


def interpol_lost_tracking(df: pd.DataFrame,column_list: list,method='linear',multiindex=True,round_to_full=False, report = False):
    """
    WARNING: CANNOT WORK WITH INT64
    Interpolates (forwards) lost tracking (NaN) in all part columns given by column_list
    :param df: dataframe containing dlc analysed frames
    :param parts: list of strings for all bodyparts that should be interpolated
    :param method: method of interpolation passed to df.interpolate, default is linear
    :return: cleaned df with added columns of True/false if row was interpolated
    """

    df_clean = df.copy()

    if multiindex:
        column_list = convert_bodyparts_to_column_multiindex(column_list,number_of_animals=1)
    if report:
        print('Interpolating Data in columns {} using method: {}. \n Detected NaN values: \n'.format
              (', '.join(map(str,column_list)),method),df_clean[column_list].isna().sum())
        if round_to_full:
            print('All values in (inlc. interpolated) will be rounded to whole numbers.')
    for col in column_list:
        # create column strings for better overview
        clm_clean = list(col)
        clm_clean[0] = clm_clean[0] + '_clean'
        clm_clean = tuple(clm_clean)
        clm_interpol = 'track_interpol'
        # interpolate missing data, default method is linear
        df_clean[clm_clean] = df_clean[col].interpolate(method=method,inplace=False,limit_direction='forward')
        # look for equal values and add column with False if not interpolated or True if it is
        # sets NaN as True (not equal) as np.nan != np.nan !
        df_clean[clm_interpol] = np.where(df_clean[clm_clean] == df_clean[col],False,True)
        # set all remaining NaN as False to correct last step (all NaN values were not interpolated)
        df_clean[clm_interpol] = np.where(df_clean[clm_clean].isna(),False,df_clean[clm_interpol])
        if round_to_full:
            # rounding numbers to whole numbers (to match pixels), but keeping them as float
            df_clean[clm_clean] = df_clean[clm_clean].round(0)
        # exchange original with cleaned column and drop cleaned
        df_clean[col] = df_clean[clm_clean]
        df_clean.drop([clm_clean],inplace=True,axis=1)
        #

    if report:
        print('After interpolation of {} entries/rows, the following NaN entries remain:\n'.format
              (df_clean.track_interpol.value_counts().loc[True]),df_clean[column_list].isna().sum())

    return df_clean


def reduce_to_experiment(df: pd.DataFrame,clm_exp=('Experiment','Status',''),clm_time=('Time','',''), report = False):
    """
    Filter data for experiment status True and reset Time according to it.
    ignores data where no experiment was started. Takes multiindex standard from DlStream.

    :param df: Dataframe to work with (Will be copied)
    :param clm_exp: column that contains bool entries for start of experiments
    :param clm_time: column containing timing from start of tracking. will be adjusted to start of experiment
    :return: reduced Dataframe with df[clm_exp] == True and reset time or just copy of df
    """
    if report:
        print('Reducing dataframe to Experiment... ')
    df_exp = df.copy()
    if any(df_exp[clm_exp]):
        if report:
            print('Reduced Dataframe by {} rows.'.format(df_exp[clm_exp].value_counts().loc[False]))
        df_exp = df_exp[df_exp[clm_exp] == True]
        df_exp[clm_time] = df_exp[clm_time] - df_exp[clm_time].iloc[0]

    else:
        if report:
            print('No Experiment was started.')

    return df_exp


def label_trials( dataframe, clm_trial = ('Experiment', 'Trial', ''), clm_label = ('Experiment', 'Trial', 'Label'), report = False):
    """
    Groups dataframe to blocks of same entry in block_clm and returns grouped dataframe
    :param df:
    :param block_clm: which column to look for blocks
    :param label_clm: column in which the labels are written
    :return:
    """
    df_labelled = label_block(dataframe, clm_block = clm_trial, clm_label = clm_label)
    if report:
        print('Found {} trials.'.format(df_labelled[clm_label].nunique()))
    return df_labelled

def convert_2_cm( df, bodyparts, px_cm_factor):
    """Converts Px data into cm by factor given to function"""
    for bp in bodyparts:
        df['Animal1', bp] = df['Animal1', bp].div(px_cm_factor)
    return df


def clean_data(path_or_df,
               number_of_animals = 1,
               report = False,
               reduce_2_experiment = True,
               px_factor = None):
    """
    Takes path to raw experiment csv file from DlStream, loads it as panda dataframe and cleans dataframe (converts empty to NaN, adjust dtype, interpolates lost tracking) and returns processed dataframe
    :param path_or_df: str, path to csv file (instead of Dataframe) or pd.Dataframe
    :param number_of_animals: number of animals that are in csv file. Corresponds to Header row 1 : Animal1 etc. Default: 1
    :param report: bool, if True prints out report of steps and found data. Default: False
    :param reduce_2_experiment: bool, if True: reduces dataframe to experiment (experiment column "True"). Default: True
    :return: cleaned dataframe
    """
    df = load_dataset(path = path_or_df, header = [0,1,2])
    bodyparts = get_bodyparts(df)

    """Create cleaning dictionary for type conversion:"""
    cleaning_dict = {('Experiment','Status', ''): 'bool',
                        ('Experiment', 'Trial', ''): 'bool',
                        ('Time', '',''): 'float64'}
    for col in convert_bodyparts_to_column_multiindex(bodyparts, number_of_animals= number_of_animals):
        cleaning_dict[col] = 'float64'

    """Let's get rid of those empty cells..."""
    df = convert_empty_to_nan(df)

    """Let's do this!"""
    adjust_dtype_according_to_dict(df= df, dtype_dict= cleaning_dict)
    df = interpol_lost_tracking(df, bodyparts, multiindex = True, round_to_full= True)
    if reduce_2_experiment:
        #reduce to experiment Status True and label events/trials (False.....True....False) then sort new column into old framework
        df = reduce_to_experiment(df)
        df = label_trials(df, clm_trial=('Experiment', 'Trial', ''))
        df.sort_index(axis=1, inplace=True)

    if px_factor is not None:
        df = convert_2_cm(df, bodyparts, px_cm_factor= px_factor)

    """Final report"""

    if report:
        print(df.describe(include = 'all'))
        print(df.head())

    return df


def batch_process(folder_path, px_factor):
    """Processes DLStream output file batch"""

    files = glob.glob(folder_path + '/' + '*.csv')

    print("Found {} files".format(len(files)))

    for num,file in enumerate(files):
        filename = os.path.basename(file)
        filename = filename.split('.')[0]
        output_filepath, _ = file.split('.')
        print('Processing file {}'.format(num + 1),'\n' + filename)
        df = clean_data(path,number_of_animals=1,report=False,reduce_2_experiment=True,px_factor=px_factor)
        df.to_csv(output_filepath + '_clean.csv',sep=",")


if __name__ == '__main__':
    #Enter px to cm ratio or set px_factor to None
    px_factor = 7.54
    path = r'FULLPATH_TO_DLSTREAM_OUTPUT.csv'
    df = clean_data(path, number_of_animals=1, report=True, reduce_2_experiment=True, px_factor = px_factor)
    output_filepath,_ = path.split('.')
    df_dlc.to_csv(output_filepath + '_clean.csv', sep=",")
