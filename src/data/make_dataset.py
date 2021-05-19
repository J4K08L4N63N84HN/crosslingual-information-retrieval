import numpy as np
import pandas as pd


def get_sample(df, n):
    """ Function to generate a sample set of 2n with n correct examples and n wrong examples. """
    # get n entries with a correct translation marked with a 1 in the translation column
    df_right = pd.concat([df.loc[0:(n - 1)], pd.DataFrame(np.ones((n, 1)), columns=['Translation'])], axis=1)
    # get n entries with a wrong translation marked with a 0 in the translation column, shift of 1000 to prevent
    # correlation
    df_wrong = pd.concat([pd.DataFrame(df.loc[n:(2 * n - 1)]['English']).reset_index(drop=True),
                          pd.DataFrame(df.loc[(n + 1000):2 * n+1001]['German']).reset_index(drop=True)], axis=1)
    df_wrong = pd.concat([df_wrong, pd.DataFrame(np.zeros((n, 1)), columns=['Translation'])], axis=1)
    # combine the two tables
    df_dataset = pd.concat([df_right, df_wrong]).reset_index(drop=True)

    return df_dataset
