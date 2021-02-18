"""
DeepLabStream utility code is meant to be used with DLStream
© J.Schweihoff
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/dlstream_utils
Licensed under GNU General Public License v3.0
"""


import pandas as pd
import numpy as np


def insert_new_level_multiindex(df,lvl_name,lvl_value,lvl):
    # Convert index to dataframe
    old_idx = df.index.to_frame()
    print('\n\n\n\n',old_idx,'\n\n\n\n')

    # Insert new level at specified location
    old_idx.insert(lvl,lvl_name,lvl_value)
    print('\n\n\n\n',old_idx,'\n\n\n\n')

    # Convert back to MultiIndex
    df.index = pd.MultiIndex.from_frame(old_idx)
    return df


def flatten_multiindex(df):
    """flattens multiindex from ('Header1",'Header2', 'Header3', ...) to 'Header1_Header2_Header3...'"""
    df_flatten = df.copy()
    df_flatten.columns = ['_'.join(col).strip('_') for col in df_flatten.columns.values]
    return df_flatten


def remove_unnamed_lvl_multiindex(df: pd.DataFrame):
    """ takes dataframe (usually after reimport from csv) with false multiindex column and resets unnamed levels to ''
    Rename unnamed columns name for Pandas DataFrame
    See https://stackoverflow.com/questions/41221079/rename-multiindex-columns-in-pandas

    Parameters
    ----------
    df : pd.DataFrame object
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Output dataframe

    """
    for i,columns_old in enumerate(df.columns.levels):
        columns_new = np.where(columns_old.str.contains('Unnamed'),'',columns_old)
        df.rename(columns=dict(zip(columns_old,columns_new)),level=i,inplace=True)
    return df


def check_if_path(input_param):
    if isinstance(input_param, str):
        return True
    else:
        return False


def check_if_df(input_param):
    if isinstance(input_param, pd.DataFrame) or isinstance(input_param, pd.Series):
        return True
    else:
        return False


def load_dataset(path: str,header = [0,1,2],index_col: int = 0,sep=';'):
    """
    Loads csv as panda dataframe and converts empty multiindex naming if applicable
    If passed a dataframe it returns it (for compatability in batch processing)
    :param path: str, path to csv file
    :param header: int, number of rows to take as header
    :param index_col: int or column name, column that should be considered as index
    :param sep: str, delimiter that seperates entries. standard are ',' ';'.
    :return: dataframe
    """
    if check_if_path(path):
        df = pd.read_csv(path,header=header,sep=sep,index_col=index_col)
        df = remove_unnamed_lvl_multiindex(df)
        return df
    elif check_if_df(path):
        return path

    else:
        raise ValueError('Incorrect input. Path must be pd.Dataframe/Series or str.')


def convert_nested_dic2df(dictionary,orient='columns'):
    """ Taken from https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    A pandas MultiIndex consists of a list of tuples. So the most natural approach would be to reshape your input dict
    so that its keys are tuples corresponding to the multi-index values you require. Then you can just construct your
    dataframe using pd.DataFrame.from_dict, using the option orient='index': """

    collection_df = pd.DataFrame.from_dict({(i,j): dictionary[i][j]
                                            for i in dictionary.keys()
                                            for j in dictionary[i].keys()},
                                           orient=orient)
    return collection_df


def label_block(df,clm_block,clm_label):
    """
    Groups dataframe to blocks of same entry in block_clm and returns grouped dataframe
    :param df:
    :param block_clm: which column to look for blocks
    :param label_clm: column in which the labels are written
    :return:
    """
    df_labelled = df.copy()
    df_labelled[clm_label] = (df_labelled[clm_block].shift(1) != df_labelled[clm_block]).astype(int).cumsum()
    return df_labelled



def get_unique_levels_clm(df,level: int = 0):
    unique_levels = list(set(df.columns.get_level_values(level)))
    return unique_levels


def get_bodyparts(df,animal_clm='Animal1'):
    bodyparts = get_unique_levels_clm(df[animal_clm])
    return bodyparts


def convert_bodyparts_to_column_multiindex(bodyparts,number_of_animals=1):

    """Takes list of str as input and converts it to DLstreams multiindexing with form ('Animal X', 'bodypart', 'X'/'Y')"""

    bp_columns = []
    for number in range(1,number_of_animals + 1):
        animal_str = 'Animal{}'.format(number)
        for bp in bodyparts:
            bp_columns.append((animal_str,bp,'x'))
            bp_columns.append((animal_str,bp,'y'))
    return bp_columns




