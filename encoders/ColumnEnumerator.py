import numpy as np
import pandas as pd

class ColumnEnumerator:


    '''ColumnEnumerator class is designed to perform transformations on columns of a DataFrame. Its purpose is to preprocess
    and manipulate data in order to enhance its quality and make it more suitable for analysis or modeling tasks.
    By applying these transformations, the ColumnEnumerator class aims to improve the quality and usability of the data,
    making it more amenable to various data analysis techniques, statistical modeling, or machine learning algorithms.

    Value Enumeration: For columns containing categorical or discrete data, the class assigns unique indices or numerical
    representations to each distinct value. This enumeration process facilitates subsequent analysis or modeling by
    transforming categorical data into a numerical format that can be processed by algorithms.
    '''

    def __init__(self):
        self.enumerated_cols = []
        self.sorted_values = {}

    def fit_transform(self, df):
        tuples_once_represented_values = []
        updated_df = df.copy()

        for col in df.columns:
            unique_values = df[col].dropna().unique()

            if df[col].isna().mean() > 0.6:
                continue

            if len(unique_values) <= 30 and df[col].dtype != 'float64':
                unique_value_counts = df[col].value_counts()
                once_represented_values = unique_value_counts[unique_value_counts == 1].index

                indices = df[df[col].isin(once_represented_values)].index
                updated_df.loc[indices, col] = df[col].mode().iloc[0]
                tuples_once_represented_values.extend([(col, index, value) for index, value in zip(indices, df.loc[indices, col])])

                try:
                    unique_values_after_replacement = updated_df[col].dropna().unique()
                    sorted_non_nan_values = np.sort(unique_values_after_replacement)
                    enumerated_sorted_non_nan_values = {val: i for i, val in enumerate(sorted_non_nan_values)}
                    self.sorted_values[col] = {k: v for k, v in enumerated_sorted_non_nan_values.items() if not pd.isna(k)}
                    self.enumerated_cols.append(col)
                except TypeError as e:
                    print(f"Error sorting column '{col}': {e}")

            elif len(unique_values) <= 30:
                if not all(str(val).endswith('.0') for val in unique_values):
                    continue

                unique_value_counts = df[col].value_counts()
                once_represented_values = unique_value_counts[unique_value_counts == 1].index

                indices = df[df[col].isin(once_represented_values)].index
                updated_df.loc[indices, col] = df[col].mode().iloc[0]
                tuples_once_represented_values.extend([(col, index, value) for index, value in zip(indices, df.loc[indices, col])])

                try:
                    unique_values_after_replacement = updated_df[col].dropna().unique()
                    sorted_non_nan_values = np.sort(unique_values_after_replacement)
                    enumerated_sorted_non_nan_values = {val: i for i, val in enumerate(sorted_non_nan_values, start=0)}
                    self.sorted_values[col] = {k: v for k, v in enumerated_sorted_non_nan_values.items() if not pd.isna(k)}
                except TypeError as e:
                    print(f"Error sorting column '{col}': {e}")

        df_copy = updated_df.copy()
        for col, values_dict in self.sorted_values.items():
            df_copy[col] = df_copy[col].map(values_dict).fillna(df_copy[col])

        return df_copy, self.sorted_values, tuples_once_represented_values