from dotenv import load_dotenv
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

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

    # ### Train a logistic regression model on the training data
    # model = LogisticRegression(max_iter=1000, random_state=42)
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    
    # ### Train using decision tree
    # model = DecisionTreeClassifier(random_state=42)
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    
    # ### Train using random forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Out of all predictions, how many were correct?
    # Formula:
    #     Accuracy = Correct Predictions / Total Predictions
    accuracy = accuracy_score(y_test, predictions)  
    

    # When the model says “survived,” how often is it correct?
    # Formula:
    #     Precision = True Positives / (True Positives + False Positives)
    precision = precision_score(y_test, predictions)
    
    
    recall = recall_score(y_test, predictions)

    # Out of all actual survivors, how many did the model successfully find?
    # Formula:
    #     Recall = True Positives / (True Positives + False Negatives)
    f1 = f1_score(y_test, predictions)
    
    print(f'Test accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    
    # confusion matric
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)
    
    model_path = Path(model_file)
    ensure_directory(model_path.parent)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    # confusion_matrix_map(cm)
    print(f'Model saved to {model_path}')
    return model


def confusion_matrix_map(cm):
    # plt.figure(figsize=(6, 4))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Died', 'Survived'],
        yticklabels=['Died', 'Survived']
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# X_train = train_df.drop('Survived', axis=1)
# y_train = train_df['Survived']
# graph using x, y
# def x_y_graph():
#     processed_train = os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_TRAIN_FILENAME)
#     train_df = load_data(processed_train)
#     X_train = train_df.drop('Survived', axis=1)
#     y_train = train_df['Survived']
    
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x=X_train['Age'], y=X_train['Fare'], hue=y_train, palette='Set1')
#     plt.title('Age vs Fare colored by Survival')
#     plt.xlabel('Age')
#     plt.ylabel('Fare')
#     plt.legend(title='Survived', loc='upper right')
#     plt.show()
    
def x_y_graph():
    processed_train = os.path.join(
        PROCESSED_DATA_PATH,
        PROCESSED_DATA_TRAIN_FILENAME
    )

    train_df = load_data(processed_train)

    X_train = train_df.drop('Survived', axis=1)
    y_train = train_df['Survived']

    # Add target back for visualization
    df = X_train.copy()
    df['Survived'] = y_train

    # -----------------------------
    # Pair Plot
    # -----------------------------
    sns.pairplot(
        df,
        hue='Survived',
        vars=['Age', 'Fare', 'Pclass', 'Sex'],
        palette='Set1'
    )

    plt.show()

    # -----------------------------
    # Individual Scatter Plots
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.scatterplot(
        ax=axes[0, 0],
        data=df,
        x='Age',
        y='Fare',
        hue='Survived',
        palette='Set1'
    )
    axes[0, 0].set_title('Age vs Fare')

    sns.scatterplot(
        ax=axes[0, 1],
        data=df,
        x='Age',
        y='Pclass',
        hue='Survived',
        palette='Set1'
    )
    axes[0, 1].set_title('Age vs Pclass')

    sns.scatterplot(
        ax=axes[1, 0],
        data=df,
        x='Fare',
        y='Pclass',
        hue='Survived',
        palette='Set1'
    )
    axes[1, 0].set_title('Fare vs Pclass')

    sns.scatterplot(
        ax=axes[1, 1],
        data=df,
        x='Age',
        y='Sex',
        hue='Survived',
        palette='Set1'
    )
    axes[1, 1].set_title('Age vs Sex')

    plt.tight_layout()
    plt.show()
    


def x_y_graph1():

    processed_train = os.path.join(
        PROCESSED_DATA_PATH,
        PROCESSED_DATA_TRAIN_FILENAME
    )

    train_df = load_data(processed_train)

    X_train = train_df.drop('Survived', axis=1)

    # Normalize features between 0 and 1
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(X_train)

    scaled_df = pd.DataFrame(
        scaled_data,
        columns=X_train.columns
    )

    plt.figure(figsize=(14, 7))

    # Draw one line per feature
    for column in scaled_df.columns:
        plt.plot(
            scaled_df.index,
            scaled_df[column],
            label=column
        )

    plt.title("Feature Comparison Graph")
    plt.xlabel("Passenger Index")
    plt.ylabel("Normalized Value")
    plt.legend()
    plt.show()
  
if __name__ == '__main__':

    processed_train = os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_TRAIN_FILENAME)
    processed_test = os.path.join(PROCESSED_DATA_PATH, PROCESSED_DATA_TEST_FILENAME)
    model_file = os.path.join(MODEL_PATH, MODEL_FILENAME)
    # x_y_graph()
    # x_y_graph1()
    train_model(processed_train, processed_test, model_file)
