import pandas as pd
import numpy as np


def preprocessing(data, types, convert):
    df = pd.DataFrame(data)
    drops = []
    result = df.iloc[:, -1]
    df = df.iloc[:, :-1]

    for name, tp in zip(df, types):
        new_column = df[name].copy()
        if tp == 'numeric':
            # if all the values are NaN (representing missing value), remove the variable from the dataset
            if df[name].isnull().all():
                drops.append(name)
            else:
                # fill missing values with the mean
                df[name] = df[name].replace(np.nan, df[name].mean())

                # normalization
                df[name] = (df[name] - df[name].min()) / (df[name].max() - df[name].min())
        else:
            # find majority for later replacement of missing values
            counts = df[name].value_counts()
            majority = counts.idxmax().decode("utf-8")
            if majority != "?" and majority != np.nan:
                # iterate over the values in the variable and replace the missing values by the majority found
                for idx, val in df[name].items():
                    if val.decode("utf-8") == "?" or val.decode("utf-8") == np.nan:
                        new_column[idx] = majority.encode()
                df[name] = new_column
            else:
                # if majority is "?" (representing missing value), remove the variable from the dataset
                drops.append(name)

            if convert:
                dummy = pd.get_dummies(df[name], prefix=name)
                df = pd.merge(left=df, right=dummy, left_index=True, right_index=True)
                drops.append(name)

    for item in drops:
        del df[item]

    df['clase'] = result

    return df
