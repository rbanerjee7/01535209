"""Python File for helper functions."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Function to filter data based on ProductID
def filter_product(product: int, dataframe: pd.DataFrame, sort: bool = True):
    """
    Filter the given dataframe based on the provided ProductID.

    Parameters:
    - product (int): The ID of the product to filter the dataframe by.
    - dataframe (pd.DataFrame): The dataframe to be filtered.
    - sort (bool, optional): Indicates whether the filtered dataframe should be sorted. Defaults to True.

    Returns: - pd.DataFrame: The filtered dataframe. If sort is True, the dataframe is sorted by 'year', 'month',
    and 'day' in ascending order.
    """

    filtered = dataframe.loc[dataframe['ProductID'] == product]

    if sort:
        return filtered.sort_values(by=['date'], ascending=True)

    return filtered


# Function to filter data based on StoreID
def filter_store(StoreID: int, dataframe: pd.DataFrame, sort: bool = True):
    """
    Filter the given dataframe based on the given StoreID.

    Parameters:
    - store (int): The ID of the store to filter the dataframe on.
    - dataframe (pd.DataFrame): The dataframe to be filtered.
    - sort (bool, optional): Whether to sort the filtered dataframe.
      Defaults to True.

    Returns:
    - pd.DataFrame: The filtered dataframe. If sort is True, it is sorted
      in ascending order based on the 'year', 'month', and 'day' columns.
    """

    filtered = dataframe.loc[dataframe['StoreID'] == StoreID]

    if sort:
        return filtered.sort_values(by=['date'], ascending=True)

    return filtered


# Function to filter data based on ProductID and StoreID
def filter_product_store(product: int, store: int, dataframe: pd.DataFrame, sort: bool = True):
    """
    Filter the given dataframe based on the provided `ProductID` and `StoreID`.

    Parameters:
    - product (int): The ID of the product to filter.
    - store (int): The ID of the store to filter.
    - dataframe (pd.DataFrame): The dataframe to filter.
    - sort (bool, optional): Whether to sort the filtered dataframe. Default is True.

    Returns:
    - `pd.DataFrame`: The filtered dataframe.
    """

    filtered = dataframe.loc[(dataframe['ProductID'] == product) & (dataframe['StoreID'] == store)]

    if sort:
        return filtered.sort_values(by=['date'], ascending=True)

    return filtered


def week_agg(product: int, store: int, df: pd.DataFrame):
    """
    Function to aggregate data by week for a specific product and store.

    Parameters:
    - product (int): The ID of the product.
    - store (int): The ID of the store.
    - df (pd.DataFrame): The input dataframe containing the data.

    Returns:
    - filtered (pd.DataFrame): The aggregated dataframe filtered by week.
    """

    filtered = filter_product_store(product, store, df)
    filtered = filtered.drop(columns=['ProductID', 'StoreID'])
    filtered.set_index('date', inplace=True)
    filtered = filtered.resample('W').sum()

    return filtered


def weekly_sales_forecast(product: int, store: int, df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Function to forecast sales for a specific product and store.

    Parameters:
    - store (int): The ID of the store.
    - product (int): The ID of the product.
    - df1 (pd.DataFrame)
    - df2 (pd.DataFrame)

    Returns:
    - weekly (pd.DataFrame): Weekly aggregated dataframe.
    """

    filtered_df1 = filter_product_store(product, store, df1)
    filtered_df2 = filter_product_store(product, store, df2)

    filtered_df1 = filtered_df1.drop(columns=['ProductID', 'StoreID'])
    filtered_df2 = filtered_df2.drop(columns=['ProductID', 'StoreID'])

    filtered_df1.set_index('date', inplace=True)
    filtered_df2.set_index('date', inplace=True)

    filtered_df1 = filtered_df1.resample('W').sum()
    filtered_df2 = filtered_df2.resample('W').sum()

    weekly = pd.merge(filtered_df1, filtered_df2, left_index=True, right_index=True, how='inner')

    return weekly

def generate_filename(hyperparameters: dict,file_type:str,model_type:str):

    import datetime

    # Generate a filename based on hyperparameters
    filename = file_type + "_" + model_type + "_"

    for key, value in hyperparameters.items():
        filename += f'{value}_'
    filename += f'{datetime.datetime.now().strftime("t_%H_%M_d_%d_%m_%Y")}'
    return filename