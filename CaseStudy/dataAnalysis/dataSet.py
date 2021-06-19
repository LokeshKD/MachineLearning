
import pandas as pd

# Read the CSV data files into DataFrames
def read_dataframes():
    # CODE HERE
    train_df = pd.read_csv('weekly_sales.csv')
    features_df = pd.read_csv('features.csv')
    stores_df = pd.read_csv('stores.csv')
    return (train_df, features_df, stores_df)

