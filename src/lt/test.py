from .src.utils import *
from .src.main import *
import pandas as pd


def test_point_rep(point_rep_df: pd.DataFrame):
    true_df = true_df_for_test_point_rep()

    true_df_sorted = true_df.sort_values(by=list(true_df.columns))
    point_rep_df_sorted = point_rep_df.sort_values(by=list(true_df.columns))
    
    assert(sorted(true_df_sorted) == sorted(point_rep_df_sorted))
    print("Pass test_point_rep. Successfully generating a point representation.")

def true_df_for_test_point_rep():
    true_df = pd.DataFrame({'pt': [1]*8,
                            'bin_t': [0,1,2,2]*2,
                            'bin_v': [2]*8,
                            'step_size_t': [2]*8,
                            'step_size_v': [5]*8,
                            's_t': [0]*8,
                            's_v': [0,1]*4,
                            'Fs': [1, 0.367879441, 1, 1, 1, 0.367879441, 1, 1]
                            })
    return true_df

def create_point_rep(step_size_t, step_size_v, s_v_list, s_t_list, step_size_e=0, s_e_list=None):
    df, num_e = initialize_test_df()
    s_combinations_df = initialize_s_combinations_df(s_v_list, s_t_list, s_e_list)

    point_rep_df = point_rep(df, step_size_t, step_size_v, step_size_e, s_combinations_df, num_e)
    
    ## Create the true dataframe, compare it with the one above.
    # assert(point_rep.loc[])
    # print(point_rep_df)
    return point_rep_df

def initialize_test_df():
    pt = [1] * 4
    t_hr = [1,2,4,5]
    value_num = [10,11,12,10]
    num_e = 0

    df = pd.DataFrame({'pt': pt, 't_hr': t_hr, 'value_num': value_num})
    return df, num_e

def initialize_s_combinations_df(s_v_list: list, s_t_list: list, s_e_list: list): # Move to main.py
    # Create a dataframe for s_combinations
    s_v_list = s_v_list
    s_t_list = s_t_list
    s_e_list = s_e_list

    s_combinations = list(itertools.product(s_v_list, s_t_list))
    s_combinations_df = pd.DataFrame(s_combinations, columns=['s_v', 's_t'])
    # if not s_e_list:
    #     s_combinations = list(itertools.product(s_v_list, s_t_list))
    #     s_combinations_df = pd.DataFrame(s_combinations, columns=['s_v', 's_t'])
    # else:
    #     s_e_list = [s_e_list for i in range(num_e)]
    #     s_combinations = list(itertools.product(s_v_list, s_t_list))
    #     s_
    return s_combinations_df

# point_rep = create_point_rep(2, 5, s_v_list=[0], s_t_list=[0,1])
# print(point_rep)
# test_point_rep(point_rep)
# df = sum_pooling(point_rep)
# print(df)
# df = laplace_pooling(df, 't', 3)
# print(df)

# print(true_df_for_test_point_rep())

# data = import_csv("all.csv")
# df = preproces_df(data)
# df = value_trans(df)
# df = t_trans(df)
# df = df.drop('event', axis=1)
# num_e = 0

# s_v_list = [0, 1]
# s_t_list = [0, 1]
# s_e_list = [0]
# s_e_list = [s_e_list for i in range(num_e)]

# # column names starting with 's_e'
# s_e_columns = ["s_e{}".format(i) for i in range(num_e)]

# s_combinations = list(itertools.product(s_v_list, s_t_list, *s_e_list))
# s_combinations_df = pd.DataFrame(s_combinations, columns=['s_v', 's_t', *s_e_columns])
# # ==========

# step_size_t, step_size_v, step_size_e = (3, 1.5, 1)
# point_rep_df = point_rep(df, step_size_t, step_size_v, step_size_e, s_combinations_df, num_e)
# bin_rep_df = sum_pooling(point_rep_df)
# laplace_rep_df = laplace_pooling(bin_rep_df, 't', 2)

# print(laplace_rep_df)





