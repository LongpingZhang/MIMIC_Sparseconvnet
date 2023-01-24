import src.transform as transform
import pandas as pd
import itertools

def create_point_rep(step_size_t, step_size_v, s_v_list, s_t_list, step_size_e=0, s_e_list=None):
    df, num_e = initialize_test_df()

    s_combinations_df = initialize_s_combinations_df(s_v_list, s_t_list, s_e_list)

    point_rep_df = transform.point_rep(df, step_size_t, step_size_v, step_size_e, s_combinations_df, num_e)
    print("===Point representation===")
    print(point_rep_df)
    return point_rep_df

def initialize_test_df():
    pt = [1] * 4 + [2] * 4
    t_hr = [1,2,4,5] * 2
    value_num = [10,11,12,10] * 2
    num_e = 0

    df = pd.DataFrame({'pt': pt, 't_hr': t_hr, 'value_num': value_num})
    print("===Originial df===")
    print(df)
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

if __name__ == '__main__':
    df = create_point_rep(2, 5, s_v_list=[0], s_t_list=[0,1])

    df = transform.sum_pooling(df)
    print("===Sum pooling===")
    print(df)

    df = transform.laplace_pooling(df, 't', 3)
    print("===Laplace pooling===")
    print(df)