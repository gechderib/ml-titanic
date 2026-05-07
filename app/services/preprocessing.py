
# process titanic data for training and testing
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
 
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    
    # Drop unnecessary columns
    df.drop(['Name', 'Ticket', 'Cabin','PassengerId', "Embarked"], axis=1, inplace=True)
    return df

def process_and_save_data(input_file, output_file):
    df = pd.read_csv(input_file)
    processed_df = preprocess_data(df)
    processed_df.to_csv(output_file, index=False)
