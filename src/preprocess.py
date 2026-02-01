import pandas as pd
import os

# Define paths
INPUT_PATH = "data/Titanic-Dataset.csv"
OUTPUT_PATH = "data/processed/titanic_processed.csv"

def preprocess():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Dataset not found at {INPUT_PATH}")

    # Load data
    df = pd.read_csv(INPUT_PATH)
    
    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Drop columns that won't be used
    df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)
    
    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Save processed data
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Preprocessed data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess()
