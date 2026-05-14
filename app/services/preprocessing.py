
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
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
    
    # add family size feature FamilySize = SibSp + Parch + 1
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # from name extract title and encode it Mr, Mrs, Miss, Master,sir also the title has to be in [Mr, Mrs, Miss, Master, Sir] if not use mode of title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # df['Title'].fillna(df['Title'].mode()[0], inplace=True)
    # df['Title'] = df['Title'].apply(lambda x: x if x in ['Mr', 'Mrs', 'Miss', 'Master', 'Sir'] else 'Other')
    # df['Title'] = label_encoder.fit_transform(df['Title'])
    # Drop unnecessary columns
    df.drop(['Name', 'Ticket', 'Cabin','PassengerId'], axis=1, inplace=True)
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