from sklearn.metrics import classification_report, roc_auc_score

def evaluate_performance(model, X_test, y_test):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print("Classification Report:\n", report)
    print("ROC AUC Score:", auc)
    return report, auc