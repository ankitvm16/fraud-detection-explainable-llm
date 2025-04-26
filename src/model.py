from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def build_model(config):
    params = config["model"]["parameters"]
    model_config = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", None),
        random_state=config["data"]["random_state"]
    )
    return model_config

def train_and_evaluate_model(model_config, X_train, y_train, X_test, y_test):
    #     Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    model = model_config.fit(X_train_balanced, y_train_balanced)
    score = model.score(X_test_scaled, y_test)
    return model, score, X_train_scaled, X_test_scaled