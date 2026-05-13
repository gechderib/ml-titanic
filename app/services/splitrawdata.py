from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.model_selection import train_test_split

load_dotenv()

RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')
RAW_DATA_FILENAME = os.getenv('RAW_DATA_FILENAME')
RAW_DATA_TRAIN_FILENAME = os.getenv('RAW_DATA_TRAIN_FILENAME')
RAW_DATA_TEST_FILENAME = os.getenv('RAW_DATA_TEST_FILENAME')


def split_raw_data(input_file, train_file, test_file, test_size=0.2, random_state=42):
    df = pd.read_csv(input_file)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)


if __name__ == "__main__":
    print(os.path.join(RAW_DATA_PATH, RAW_DATA_FILENAME))
    split_raw_data(
        input_file=os.path.join(RAW_DATA_PATH, RAW_DATA_FILENAME),
        train_file=os.path.join(RAW_DATA_PATH, RAW_DATA_TRAIN_FILENAME),
        test_file=os.path.join(RAW_DATA_PATH, RAW_DATA_TEST_FILENAME)
    )