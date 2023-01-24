import numpy as np
import pandas as pd
import utils
import itertools

def example():
    data = utils.import_csv("all.csv")
    df = utils.preproces_df(data)
    df = utils.value_trans(df)
    df = utils.t_trans(df)
    df, num_e = utils.event_trans(df)
    
    # Create combinations of s_v, s_t, s_e0, s_e1, s_e2
    # ==========
    s_v_list = [0, 1]
    s_t_list = [0, 1]
    s_e_list = [0]
    s_e_list = [s_e_list for i in range(num_e)]

    # column names starting with 's_e'
    s_e_columns = ["s_e{}".format(i) for i in range(num_e)]

    s_combinations = list(itertools.product(s_v_list, s_t_list, *s_e_list))
    s_combinations_df = pd.DataFrame(s_combinations, columns=['s_v', 's_t', *s_e_columns])
    # ==========

    step_size_t, step_size_v, step_size_e = (3, 1.5, 1)
    point_rep_df = point_rep(df, step_size_t, step_size_v, step_size_e, s_combinations_df, num_e)
    bin_rep_df = sum_pooling(point_rep_df)
    laplace_rep_df = laplace_pooling(bin_rep_df)

    return laplace_rep_df

if __name__ == '__main__':
    example()

def point_rep(df: pd.DataFrame, 
            step_size_t: float, step_size_v: float, step_size_e: float, 
            s_combinations_df: pd.DataFrame, 
            dim_e: float):
    '''
    Create point representations of data.
    
    Parameters:
        step_size_t, step_size_v: the initial bin width for bin_t and bin_v,
        s_combinations: the dataframe consisting of all possible combinations of s_v, s_t, and s_e{},
        s_e_columns: the list consisting of the names of all event columns,
        dim_e: the number of event columns.
    '''
    # Subset df and keep the following columns: pt, value_num, t_hr, e_0, e_1, e_2
    e_columns = ['e{}'.format(i) for i in range(dim_e)]
    df = df.loc[:, ['pt', 't_hr'] + e_columns + ['value_num']]
    # Rename value_num and t_hr
    df.rename(columns={'t_hr': 't', 'value_num': 'v'}, inplace=True)

    # Initialize bin_{}, step_size_{}
    df = initialize_bin_and_step(df, 't', step_size_t)
    df = initialize_bin_and_step(df, 'v', step_size_v)
    for i in range(dim_e):
        df = initialize_bin_and_step(df, 'e{}'.format(i), step_size_e)
    
    # Insert columns s_v, s_t, s_e1, s_e2, s_e3
    df = df.merge(s_combinations_df, 'cross')

    # Insert a column Fs
    df['Fs'] = calculate_F(df, dim_e)

    # Drop unnecessary columns
    if dim_e == 0:
        df.drop(columns=['t', 'v'], inplace=True)
    else:
        df.drop(columns=['t', 'v', *e_columns], inplace=True)
    
    return df

def initialize_bin_and_step(df: pd.DataFrame, col_suffix: str, step_size: float):
    df = df.copy()

    bin_name = 'bin_{}'.format(col_suffix)
    if bin_name not in df.columns:
        df[bin_name] = df[col_suffix] // step_size

    step_size_name = 'step_size_{}'.format(col_suffix)
    if step_size_name not in df.columns:
        df[step_size_name] = step_size
    
    return df

def calculate_F(df: pd.DataFrame, num_e: float):
    '''
    Calculate the Fs column.
    '''
    df = df.copy()

    # Initialize Fs
    result = 1

    # Calculate Fs contributed from events
    e_columns = ['e{}'.format(i) for i in range(num_e)]
    s_e_columns = ['s_e{}'.format(i) for i in range(num_e)]
    bin_e_columns = ['bin_e{}'.format(i) for i in range(num_e)]
    step_size_e_columns = ['step_size_e{}'.format(i) for i in range(num_e)]
    for i in range(num_e):
        result *= __calculate_F(df, e_columns[i], s_e_columns[i], bin_e_columns[i], step_size_e_columns[i])
    
    # Calculate F contributed from t and v
    result *= __calculate_F(df, 't', 's_t', 'bin_t', 'step_size_t')
    result *= __calculate_F(df, 'v', 's_v', 'bin_v', 'step_size_v')
    
    return result

def __calculate_F(df: pd.DataFrame, value_col: str, s_col: str, bin_col: str, step_size_col: str):
    '''
    The helper function for calculating the Fs column.
    '''
    return np.exp(-df[s_col] * (df[value_col] - df[bin_col] * df[step_size_col]))

def sum_pooling(df: pd.DataFrame):
    df = df.copy()
    
    groupby_columns = list(df.columns)
    groupby_columns.remove('Fs')
    df = pd.DataFrame(df.groupby(by=groupby_columns).sum().loc[:, 'Fs']).reset_index()

    return df

def laplace_pooling(df: pd.DataFrame, col_suffix: str, step_size: float):
    '''
    Parameters:
        df: the dataframe on which laplace pooling is performed
        col_suffix: the bin_{col_suffix} to pool across
        step_size: the step_size of pooling
            e.g. If the previous step_size in df is 2 and the step_size passed through is 3,
                then the new step_size will be 6.
    '''
    df = df.copy()

    bin_col_name = "bin_{}".format(col_suffix)
    step_size_col_name = "step_size_{}".format(col_suffix)
    new_bin = df[bin_col_name] // step_size
    new_step_size = df[step_size_col_name] * step_size

    # Laplace tranformation on Fs based on bin_{col_suffix}
    s_col = "s_{}".format(col_suffix)
    df['Fs'] = df['Fs'] * np.exp( -df[s_col] * (df[bin_col_name] * df[step_size_col_name] - new_bin * new_step_size) )

    # Update the columns bin_{col_suffix} and step_size_{col_suffix}
    df[bin_col_name] = new_bin
    df[step_size_col_name] = new_step_size

    # Pooling
    df = sum_pooling(df)

    return df