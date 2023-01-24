'''
utils.py is used in particular for processing the "data/all.csv".
'''


import pandas as pd
# from pathlib import Path
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn

def import_csv(csv_file_name):
    '''
    Import the clinical data from the directory 'data/'.
    '''
    base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../../")
    datafile_path = os.path.join(base_dir, "data/", csv_file_name)

    return pd.read_csv(datafile_path, delimiter="|")

def preproces_df(df: pd.DataFrame):
    '''
    Drop rows with the 'event' or the 't' columns equal to NaN. 
    Drop dupicated rows.
    '''
    df = df.copy()

    # Drop rows with the 'event' or the 't' columns equal to NaN.
    df = df.dropna(subset = ['t', 'event'])
    # Drop duplicated rows.
    df = df.drop_duplicates()

    return df

def value_trans(df: pd.DataFrame):
    '''
    Transform the 'value' column.
    '''
    df = df.copy()

    # Obtain unique 'event'-'value' combinations where value is not NaN,
    # because value = NaN will be unchanged.
    ev_df = df.loc[:,['event', 'value']].drop_duplicates().dropna(subset=['value'])
    
    # Tranform numbers of the format like 1,000 to 1000.
    r_indices = ev_df[ev_df['value'].str.contains(",") & 
                    ev_df['value'].str.match('^[0-9]{1,3}(,[0-9]{3})*(\.[0-9]+)?$')].index
    ev_df.loc[r_indices, 'value'] = ev_df.loc[r_indices, 'value'].apply(lambda value: value.replace(',', '') 
                                                            if isinstance(value, str) else value)
    ev_df = ev_df.drop_duplicates()

    ev_df = label_encoding(ev_df)

    # Once we have all possible unique combinations of 'event'-'value' in ev_df,
    # we can join two dataframe to encode categorical variables.
    df = df.merge(ev_df, how='left', left_on=['event', 'value'], right_on=['event', 'value'])

    return df

def label_encoding(ev_df: pd.DataFrame):
    '''
    Given a 'event'-'value' dataframe, perform label encoding on the value column,
    i.e. n levels -> [0, n-1].
    '''
    # Get row indices of numerical and categorical values respectively
    num_value = pd.to_numeric(ev_df['value'], 'coerce')
    cat_indices = num_value[np.isnan(num_value)].index
    num_indices = num_value[~np.isnan(num_value)].index

    # Obtain unique 'event'-['value1', 'value2', ...] combinations for categorical data
    ec_df = ev_df.loc[cat_indices, :]
    ec_df = pd.DataFrame(ec_df.groupby('event')['value'].unique()).reset_index()

    # Generate a dictionary of the format {'event1': {'value1': 0, 'value2': 1, 
    #                                                   ...}, 
    #                                       ...}.
    e_c2n = {}
    for _, r in ec_df.iterrows():
        event = r['event']
        values = r['value']
        c2n = dict(zip(values, range(len(values))))
        e_c2n[event] = c2n
    
    # Create a new column 'value_num' in ev_df to store the numerical values and label encodings of categorical values
    ev_df['value_num'] = ""
    # Categorical values
    for i in cat_indices:
        event = ev_df.loc[i, 'event']
        value = ev_df.loc[i, 'value']
        ev_df.loc[i, 'value_num'] = e_c2n[event][value]
    # Numerical values
    ev_df.loc[num_indices, 'value_num'] = ev_df.loc[num_indices, 'value']
    
    ev_df['value_num'] = ev_df['value_num'].astype('float')

    return ev_df

def t_trans(df: pd.DataFrame):
    '''
    Transform the 't' column.
    '''
    df = df.copy()

    # Remove the rows with the following events: birthed, gender, ethnicity, 
    # because these are not events that happen at hospital
    df = df[~df['event'].str.startswith('birthed')
      & ~df['event'].str.startswith('gender')
      & ~df['event'].str.startswith('ethnicity')]

    # Get the min_t, the time that the patient has any medical event at hospitals, for each patient.
    min_t_df = pd.DataFrame(df.loc[:, ['pt', 't']].groupby('pt')['t'].min())
    min_t_df.rename(columns={'t': 'min_t'}, inplace=True)

    df = df.merge(min_t_df, how='left', left_on=['pt'], right_index=True)
    # Subtract t by min_t and apply the helpler function, 
    # so that t=0 corresponds to the first time the patient has any event at hospital
    df['t-t_min'] = df['t'] - df['min_t']
    df['t_hr'] = list(map(seconds_to_hours, df['t-t_min']))

    return df
    
def seconds_to_hours(time):
  hour = time / 3600

  return int(hour) + 1

def event_trans(df: pd.DataFrame):
    '''
    Transform the 'event' column.

    Return the dataframe and the number of columns of event embeddings
    '''
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", output_hidden_states=True)

    # Obtain the unique events and tokenize all elements in the list.
    events_list = list(df['event'].unique())
    tokens_dict = tokenizer(events_list, padding=True)

    tokens_tensor = torch.tensor(tokens_dict['input_ids'])
    type_tensor = torch.tensor(tokens_dict['token_type_ids'])
    attention_tensor = torch.tensor(tokens_dict['attention_mask'])
    with torch.no_grad():
        output = model(tokens_tensor, attention_tensor, type_tensor)
    
    batch_size = len(output[0])
    hidden_states = torch.stack(output['hidden_states'][1:], dim=0) # layer_num * batch_size * token_len * hidden_units

    # Take the last hidden layer as embeddings
    token_embeddings = hidden_states[-1]
    token_embeddings = token_embeddings.reshape(batch_size, -1)  # shape: batch_size * (token_len * hidden_units)
    embeddings_pca = pca_lowrank(token_embeddings)
    
    # Use nn.Embedding to match each event in the df to its embedding
    eventidx_embed = nn.Embedding(*embeddings_pca.shape)
    eventidx_embed.weight = nn.Parameter(embeddings_pca, requires_grad=False)

    event_eventidx_df = pd.DataFrame(data=range(len(events_list)), index=events_list, columns=['eventidx'])
    event_indices = df.merge(event_eventidx_df, how='left', left_on='event', right_index=True)['eventidx'].tolist()
    events_embeds = eventidx_embed(torch.tensor(event_indices))
    
    # Insert embeddings into the dataframe
    tmp_df = pd.DataFrame(events_embeds.numpy(), columns=['e{}'.format(i) for i in range(events_embeds.shape[1])])
    tmp_df.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = pd.concat((df, tmp_df), axis=1)

    return df, tmp_df.shape[1]

def pca_lowrank(input: torch.tensor):
    '''
    Perform dimension reduction.
    '''
    input_centered = input - input.mean(dim=0)
    _, S, V = torch.pca_lowrank(input, center=False)
    variance = S**2 / torch.sum(S**2, dim=0)
    explained_var = torch.cumsum(variance, dim=0)

    num_pcs = torch.where(explained_var > 0.9)[0][0] + 1 # Number of pcs needed to explain 90% variance
    subspace = V[:,0:num_pcs+1]
    new_data = input @ subspace
    
    return new_data