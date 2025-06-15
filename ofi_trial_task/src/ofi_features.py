import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def compute_best_level_ofi(df):
    """
    Computes Best-Level Order Flow Imbalance (OFI) using level 1 bid/ask changes.
    Returns a Python list of length=len(df), one OFI per row.
    """
    ofi = [0]
    for i in range(1, len(df)):
        # Bid‐side
        if df.loc[i, 'bid_price1'] > df.loc[i-1, 'bid_price1']:
            delta_bid = df.loc[i, 'bid_size1']
        elif df.loc[i, 'bid_price1'] == df.loc[i-1, 'bid_price1']:
            delta_bid = df.loc[i, 'bid_size1'] - df.loc[i-1, 'bid_size1']
        else:
            delta_bid = -df.loc[i-1, 'bid_size1']

        # Ask‐side
        if df.loc[i, 'ask_price1'] < df.loc[i-1, 'ask_price1']:
            delta_ask = df.loc[i, 'ask_size1']
        elif df.loc[i, 'ask_price1'] == df.loc[i-1, 'ask_price1']:
            delta_ask = df.loc[i, 'ask_size1'] - df.loc[i-1, 'ask_size1']
        else:
            delta_ask = -df.loc[i-1, 'ask_size1']

        ofi.append(delta_bid - delta_ask)
    return ofi


def compute_multi_level_ofi(df, levels=10):
    """
    Computes OFI at each of the first `levels` depths (1..levels).
    Returns a DataFrame with columns: ofi_1, ofi_2, ..., ofi_{levels}.
    """
    records = []
    for i in range(len(df)):
        row = {}
        for lvl in range(1, levels+1):
            b_p, b_s = f'bid_price{lvl}', f'bid_size{lvl}'
            a_p, a_s = f'ask_price{lvl}', f'ask_size{lvl}'

            if i == 0:
                row[f'ofi_{lvl}'] = 0
                continue

            # bid‐side change
            if df.loc[i, b_p] > df.loc[i-1, b_p]:
                delta_bid = df.loc[i, b_s]
            elif df.loc[i, b_p] == df.loc[i-1, b_p]:
                delta_bid = df.loc[i, b_s] - df.loc[i-1, b_s]
            else:
                delta_bid = -df.loc[i-1, b_s]

            # ask‐side change
            if df.loc[i, a_p] < df.loc[i-1, a_p]:
                delta_ask = df.loc[i, a_s]
            elif df.loc[i, a_p] == df.loc[i-1, a_p]:
                delta_ask = df.loc[i, a_s] - df.loc[i-1, a_s]
            else:
                delta_ask = -df.loc[i-1, a_s]

            row[f'ofi_{lvl}'] = delta_bid - delta_ask

        records.append(row)

    return pd.DataFrame.from_records(records)


def compute_integrated_ofi(multi_ofi_df):
    """
    Takes a DataFrame of multi-level OFIs (columns ofi_1..ofi_10), 
    applies PCA, and returns the first principal component (array).
    """
    X = multi_ofi_df.values
    # Normalize each row by its L1 norm to stabilize scale
    norms = np.linalg.norm(X, ord=1, axis=1, keepdims=True)
    norms[norms == 0] = 1
    Xn = X / norms

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(Xn).flatten()
    return pc1
