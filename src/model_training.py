import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os




X = pd.read_csv("./data/processed/X_processed.csv")
y = pd.read_csv("./data/processed/y_processed.csv")

y = y.values.ravel()

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1_Score": f1_score(y_test, y_pred, average="weighted")
    }

log_reg = LogisticRegression(max_iter=1000)


log_reg.fit(X_train, y_train)

log_reg_results = evaluate_model(log_reg, X_test, y_test)

print("Logistic Regression Results:")
for metric, value in log_reg_results.items():
    print(f"{metric}: {value:.4f}")

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_results = evaluate_model(dt_model, X_test, y_test)

print("\nDecision Tree Results:")
for metric, value in dt_results.items():
    print(f"{metric}: {value:.4f}")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_results = evaluate_model(rf_model, X_test, y_test)

print("\nRandom Forest Results:")
for metric, value in rf_results.items():
    print(f"{metric}: {value:.4f}")

knn = KNeighborsClassifier(n_neighbors=60, metric="euclidean")
knn.fit(X_train, y_train)


knn_results = evaluate_model(knn, X_test, y_test)

print("\nKNN Results:")
for metric, value in knn_results.items():
    print(f"{metric}: {value:.4f}")

svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

svm_model.fit(X_train, y_train)

svm_results = evaluate_model(svm_model, X_test, y_test)

print("\nSVM Results:")
for metric, value in svm_results.items():
    print(f"{metric}: {value:.4f}")

gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_results = evaluate_model(gb_model, X_test, y_test)

print("\nGradient Boosting Results:")
for metric, value in gb_results.items():
    print(f"{metric}: {value:.4f}")


xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(np.unique(y)),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

xgb_results = evaluate_model(xgb_model, X_test, y_test)

print("\nXGBoost Results:")
for metric, value in xgb_results.items():
    print(f"{metric}: {value:.4f}")

def plot_confusion_matrix(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    os.makedirs("./visuals/confusion_matrices", exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, cmap="Blues", annot=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"./visuals/confusion_matrices/{model_name}_cm.png")
    plt.show()

plot_confusion_matrix(log_reg, X_test, y_test, "Logistic Regression")
plot_confusion_matrix(dt_model, X_test, y_test, "Decision Tree")
plot_confusion_matrix(rf_model, X_test, y_test, "Random Forest")
plot_confusion_matrix(knn, X_test, y_test, "KNN")
plot_confusion_matrix(svm_model, X_test, y_test, "SVM")
plot_confusion_matrix(gb_model, X_test, y_test, "Gradient Boosting")
plot_confusion_matrix(xgb_model, X_test, y_test, "XGBoost")

model_names = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "KNN",
    "SVM",
    "Gradient Boosting",
    "XGBoost"
]

f1_scores = [
    log_reg_results["F1_Score"],
    dt_results["F1_Score"],
    rf_results["F1_Score"],
    knn_results["F1_Score"],
    svm_results["F1_Score"],
    gb_results["F1_Score"],
    xgb_results["F1_Score"]
]

plt.figure(figsize=(10, 6))
plt.barh(model_names, f1_scores)
plt.xlabel("F1 Score")
plt.title("Model Performance Comparison (F1-score)")
plt.tight_layout()
plt.savefig("./visuals/model_comparison_f1.png")
plt.show()

importances = gb_model.feature_importances_

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances)
plt.title("Feature Importance – Gradient Boosting")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("./visuals/feature_importance_gb.png")
plt.show()
