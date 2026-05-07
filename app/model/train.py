from dotenv import load_dotenv
import os
import sys
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from app.services.splitrawdata import split_raw_data
from app.services.preprocessing import process_and_save_data

load_dotenv()

RAW_DATA_PATH = os.getenv('RAW_DATA_PATH')
RAW_DATA_FILENAME = os.getenv('RAW_DATA_FILENAME')
RAW_DATA_TRAIN_FILENAME = os.getenv('RAW_DATA_TRAIN_FILENAME')
RAW_DATA_TEST_FILENAME = os.getenv('RAW_DATA_TEST_FILENAME')

PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH')
PROCESSED_DATA_TRAIN_FILENAME = os.getenv('PROCESSED_DATA_TRAIN_FILENAME')
PROCESSED_DATA_TEST_FILENAME = os.getenv('PROCESSED_DATA_TEST_FILENAME')

MODEL_PATH = os.getenv('MODEL_PATH', './models')
MODEL_FILENAME = os.getenv('MODEL_FILENAME', 'titanic_model.pkl')


def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def split_and_process_data():
    split_raw_data(
        input_file=os.path.join(RAW_DATA_PATH, RAW_DATA_FILENAME),
        train_file=os.path.join(RAW_DATA_PATH, RAW_DATA_TRAIN_FILENAME),
        test_file=os.path.join(RAW_DATA_PATH, RAW_DATA_TEST_FILENAME)
    )


def process_data():
    ensure_directory(Path(PROCESSED_DATA_PATH))
    process_and_save_data(
        input_file=os.path.join(RAW_DATA_PATH, RAW_DATA_TRAIN_FILENAME),
        output_file=os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_TRAIN_FILENAME)
    )
    process_and_save_data(
        input_file=os.path.join(RAW_DATA_PATH, RAW_DATA_TEST_FILENAME),
        output_file=os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_TEST_FILENAME)
    )


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def train_model(train_file: str, test_file: str, model_file: str):
    train_df = load_data(train_file)
    test_df = load_data(test_file)

    if 'Survived' not in train_df.columns:
        raise ValueError('Processed training data must include a Survived column')

    X_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']
    X_test = test_df.drop('Survived', axis=1)
    y_test = test_df['Survived']

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Test accuracy: {accuracy:.4f}')

    model_path = Path(model_file)
    ensure_directory(model_path.parent)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f'Model saved to {model_path}')
    return model


if __name__ == '__main__':
    split_and_process_data()
    process_data()

    processed_train = os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_TRAIN_FILENAME)
    processed_test = os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_TEST_FILENAME)
    model_file = os.path.join(MODEL_PATH, MODEL_FILENAME)

    train_model(processed_train, processed_test, model_file)
