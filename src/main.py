import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils import *

def main():
    data = import_csv()
    df = preproces_df(data)
    df = value_trans(df)
    df = t_trans(df)
    df = event_trans(df)
    return df

def point_rep(df: pd.DataFrame):
    pass

if __name__ == '__main__':
    df = main()
    print(df)