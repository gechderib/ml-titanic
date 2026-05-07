# split raw data into train and test sets 80-20 ratio and save in data folder
import pandas as pd
from sklearn.model_selection import train_test_split

def split_raw_data(input_file, train_file, test_file, test_size=0.2, random_state=42):
    df = pd.read_csv(input_file)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

