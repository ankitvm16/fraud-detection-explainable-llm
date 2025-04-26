import shap
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def shap_explanation(model, X_train, X_test, sample_index=0):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # For binary classification, use shap_values[1]
    shap.force_plot(explainer.expected_value[1], shap_values[1][sample_index,:], X_test.iloc[sample_index,:], matplotlib=True, show=True)
    plt.show()

def lime_explanation(model, X_train, X_test, sample_index=0):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=["Normal", "Fraud"],
        mode='classification'
    )
    exp = explainer.explain_instance(
        data_row=X_test.iloc[sample_index],
        predict_fn=model.predict_proba
    )
    exp.show_in_notebook(show_table=True)
    
def plot_pdp(model, X_train, features):
    display = PartialDependenceDisplay.from_estimator(model, X_train, features)
    display.figure_.suptitle("Partial Dependence Plot")
    plt.show()