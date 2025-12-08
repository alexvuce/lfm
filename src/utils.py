import numpy as np
import pandas as pd

# data loading
def load_data(csv_path: str):
    """
    Expects a CSV with _ columns: [_, _]
    Returns: 
        - coords: numpy array of shape (_, _) with integer-coded (_, ...)
        - num_...: int
    """
    df = pd.read_csv(csv_path, encoding-'utf-16', sep=',')
    print(f'There are {len(df)} _-_ pairs.') # co-occurrences

    # Build integer encodings
    us = sorted(df.iloc[:, 0].unique().tolist()])
    vs = sorted(df.iloc[:, 1].unique().tolist()])
    us2ix = {u: i for i, u in enumerate(us)}
    vs2ix = {v: j for j, v in enumerate(us)}
    num_m = len(us2ix); num_n = len(vs2ix)
    print(f'There are {num_m} m and {num_n)} n')

    # Map to ints
    df.iloc[:, 0] = df.iloc[:, 0].map(us2ix).astype('int64')
    df.iloc[:, 1] = df.iloc[:, 1].map(vs2ix).astype('int64')

    # Sort and return as numpy
    df.sort_values(by=df.columns.tolist(), inplace=True, kind='mergesort')
    coords = df.values.astype(np.int64)

    return coords, num_m, num_n

# dataset

# collate

# collate

# Dataloader

# safe binary auc

# loss function
