from dotenv import load_dotenv
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

load_dotenv()

PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH')
PROCESSED_DATA_TRAIN_FILENAME = os.getenv('PROCESSED_DATA_TRAIN_FILENAME')
PROCESSED_DATA_TEST_FILENAME = os.getenv('PROCESSED_DATA_TEST_FILENAME')

MODEL_PATH = os.getenv('MODEL_PATH', './models')
MODEL_FILENAME = os.getenv('MODEL_FILENAME', 'titanic_model.pkl')


def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)

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

    model = LogisticRegression(max_iter=1500, random_state=42)
    model.fit(X_train, y_train)
    print("x_test columns:", X_test)
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

    processed_train = os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_TRAIN_FILENAME)
    processed_test = os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_TEST_FILENAME)
    model_file = os.path.join(MODEL_PATH, MODEL_FILENAME)

    train_model(processed_train, processed_test, model_file)
