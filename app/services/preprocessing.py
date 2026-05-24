
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')
RAW_DATA_TRAIN_FILENAME = os.getenv('RAW_DATA_TRAIN_FILENAME')
RAW_DATA_TEST_FILENAME = os.getenv('RAW_DATA_TEST_FILENAME')
PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH')
PROCESSED_DATA_TRAIN_FILENAME = os.getenv('PROCESSED_DATA_TRAIN_FILENAME')
PROCESSED_DATA_TEST_FILENAME = os.getenv('PROCESSED_DATA_TEST_FILENAME')

def preprocess_data(df):
 
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Encode categorical variables
    sex_map = {
        "male":0,
        "female":1
    }
    df['Sex'] = df['Sex'].map(sex_map).astype(int)
    
    # embarked_map = {
    #     "S": 0,
    #     "C": 1,
    #     "Q": 2
    # }
    # df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    # df['Embarked'] = df['Embarked'].map(embarked_map).astype(int)
    
    # df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # # from name extract title and encode it Mr, Mrs, Miss, Master,sir also the title has to be in [Mr, Mrs, Miss, Master, Sir] if not use mode of title
    # title_map = {
    #     "Mr": 0,
    #     "Miss": 1,
    #     "Mrs": 2,
    #     "Master": 3,
    #     "Sir": 4,
    #     "Other": 5
    # }
    # df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # df['Title'].fillna(df['Title'].mode()[0], inplace=True)
    # df['Title'] = df['Title'].map(title_map).fillna(title_map['Other']).astype(int)
    
    # Drop unnecessary columns
    df.drop(['Name', 'Ticket', 'Cabin','PassengerId', 'Embarked'], axis=1, inplace=True)
    return df

def process_and_save_data(input_file, output_file):
    df = pd.read_csv(input_file)
    processed_df = preprocess_data(df)
    processed_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    process_and_save_data(
        input_file=os.path.join(RAW_DATA_PATH, RAW_DATA_TRAIN_FILENAME),
        output_file=os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_TRAIN_FILENAME)
    )
    process_and_save_data(
        input_file=os.path.join(RAW_DATA_PATH, RAW_DATA_TEST_FILENAME),
        output_file=os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_TEST_FILENAME)
    )