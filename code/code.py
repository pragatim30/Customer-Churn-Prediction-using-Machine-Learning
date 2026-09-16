import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

df = pd.read_csv("data/customer_churn.csv")

print("\n========== DATASET ==========\n")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


print("\n========== MISSING VALUES ==========\n")

print(df.isnull().sum())

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing values after converting TotalCharges:")

print(df.isnull().sum())

df = df.dropna()

print("\nShape after removing missing values:")
print(df.shape)

df = df.drop("customerID", axis=1)

print("\n========== CHURN DISTRIBUTION ==========\n")

print(df["Churn"].value_counts())

print("\nChurn Percentage:")

print(
    df["Churn"].value_counts(normalize=True) * 100
)

plt.figure(figsize=(6, 5))

sns.countplot(
    x="Churn",
    data=df
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

sns.countplot(
    x="Contract",
    hue="Churn",
    data=df
)

plt.title("Churn by Contract Type")
plt.xlabel("Contract")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

sns.countplot(
    x="InternetService",
    hue="Churn",
    data=df
)

plt.title("Churn by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))

sns.countplot(
    x="PaymentMethod",
    hue="Churn",
    data=df
)

plt.title("Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="MonthlyCharges",
    hue="Churn",
    kde=True
)

plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    kde=True
)

plt.title("Tenure Distribution by Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

numeric_df = df.select_dtypes(
    include=np.number
)

plt.figure(figsize=(12, 8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()

X = df.drop("Churn", axis=1)

y = df["Churn"]


print("\n========== FEATURES AND TARGET ==========\n")

print("Feature shape:", X.shape)

print("Target shape:", y.shape)

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

numerical_columns = X.select_dtypes(
    include=np.number
).columns

print("\nCategorical Columns:")

print(categorical_columns.tolist())

print("\nNumerical Columns:")

print(numerical_columns.tolist())

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

print("\nShape after encoding:")

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN TEST SPLIT ==========\n")

print("Training features:", X_train.shape)

print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)

print("Testing target:", y_test.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\n========== LOGISTIC REGRESSION ==========\n")

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_pred = logistic_model.predict(
    X_test_scaled
)

logistic_probability = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]


print(
    classification_report(
        y_test,
        logistic_pred
    )
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        logistic_probability
    )
)

print("\n========== KNN ==========\n")

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(
    X_train_scaled,
    y_train
)

knn_pred = knn_model.predict(
    X_test_scaled
)


print(
    classification_report(
        y_test,
        knn_pred
    )
)

print("\n========== DECISION TREE ==========\n")

tree_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

tree_model.fit(
    X_train,
    y_train
)

tree_pred = tree_model.predict(
    X_test
)


print(
    classification_report(
        y_test,
        tree_pred
    )
)

print("\n========== RANDOM FOREST ==========\n")

forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

forest_model.fit(
    X_train,
    y_train
)

forest_pred = forest_model.predict(
    X_test
)


print(
    classification_report(
        y_test,
        forest_pred
    )
)

results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        accuracy_score(y_test, logistic_pred),
        accuracy_score(y_test, knn_pred),
        accuracy_score(y_test, tree_pred),
        accuracy_score(y_test, forest_pred)
    ],

    "Precision": [
        precision_score(y_test, logistic_pred),
        precision_score(y_test, knn_pred),
        precision_score(y_test, tree_pred),
        precision_score(y_test, forest_pred)
    ],

    "Recall": [
        recall_score(y_test, logistic_pred),
        recall_score(y_test, knn_pred),
        recall_score(y_test, tree_pred),
        recall_score(y_test, forest_pred)
    ],

    "F1 Score": [
        f1_score(y_test, logistic_pred),
        f1_score(y_test, knn_pred),
        f1_score(y_test, tree_pred),
        f1_score(y_test, forest_pred)
    ]
})


print("\n========== MODEL COMPARISON ==========\n")

print(
    results.to_string(
        index=False
    )
)

results_for_plot = results.set_index(
    "Model"
)

results_for_plot[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Machine Learning Model Comparison")

plt.ylabel("Score")

plt.ylim(0, 1)

plt.xticks(rotation=0)

plt.legend()

plt.tight_layout()

plt.show()

models = {
    "Logistic Regression": logistic_pred,
    "KNN": knn_pred,
    "Decision Tree": tree_pred,
    "Random Forest": forest_pred
}


for model_name, predictions in models.items():

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d"
    )

    plt.title(
        model_name + " Confusion Matrix"
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.show()

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
        forest_model.feature_importances_

})


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\n========== TOP FEATURES ==========\n")

print(
    feature_importance.head(10).to_string(
        index=False
    )
)


# Plot top 10 features

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 10 Important Features - Random Forest"
)

plt.tight_layout()

plt.show()

sample_customer = X.iloc[[0]]

sample_customer_scaled = scaler.transform(
    sample_customer
)


sample_prediction = logistic_model.predict(
    sample_customer_scaled
)


sample_probability = logistic_model.predict_proba(
    sample_customer_scaled
)[0][1]


print("\n========== CUSTOMER PREDICTION ==========\n")


if sample_prediction[0] == 1:

    print(
        "Prediction: Customer is likely to churn."
    )

else:

    print(
        "Prediction: Customer is likely to stay."
    )


print(
    "Churn Probability:",
    round(sample_probability * 100, 2),
    "%"
)

print("\n========================================")
print("CUSTOMER CHURN PROJECT COMPLETED")
print("========================================")