import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

def load_config(config_path="../config/config.yml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_data(config):
    data_path = config["data"]["data_path"]
#     data_path = "../data/credit_card_behaviour_score.csv.gz"
    df = pd.read_csv(data_path, compression='gzip')
    return df

# Function to handle missing values
def preprocess_data(df):
    
    # Calculate missing percentages
    missing_pct = (df.isnull().sum() / len(df)) * 100
    
    # Drop columns with > 70% missing values
    cols_to_drop = missing_pct[missing_pct > 70].index
    df = df.drop(columns=cols_to_drop)
    
    # Fill remaining missing values with median (more robust than mean)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

def split_data(df, config):
    if 'bad_flag' in df.columns:
        target_col = "bad_flag"
    else:
        raise ValueError("Target column (Fraud- bad_flag) not found in dataset.")
    
    col_to_be_dropped = ["account_number", "bad_flag"]

    X = df.drop(col_to_be_dropped, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y)  
    
    return X_train, X_test, y_train, y_test