import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

train_df=pd.read_csv("data/customer_churn_dataset-training-master.csv")
test_df=pd.read_csv("data/customer_churn_dataset-testing-master.csv")
print("\n TRAINING DATASET \n")

print("First 5 rows:")
print(train_df.head())

print("\nDataset Shape:")
print(train_df.shape)

print("\nColumn Names:")
print(train_df.columns.tolist())

print("\nDataset Information:")
train_df.info()

print("\nStatistical Summary:")
print(train_df.describe())

print("\n TESTING DATASET \n")

print("First 5 rows:")
print(test_df.head())

print("\nDataset Shape:")
print(test_df.shape)

print("\nColumn Names:")
print(test_df.columns.tolist())

print("\n MISSING VALUES \n")

print("Training Data:")
print(train_df.isnull().sum())

print("\nTesting Data:")
print(test_df.isnull().sum())

train_df=train_df.dropna()

print("\nTraining shape after removing missing values:")
print(train_df.shape)

print("\n CHURN DISTRIBUTION \n")

print("Training data:")
print(train_df["Churn"].value_counts())

print("\nTraining churn percentage:")
print(
    train_df["Churn"].value_counts(normalize=True) * 100
)

print("\nTesting data:")
print(test_df["Churn"].value_counts())

print("\nTesting churn percentage:")
print(test_df["Churn"].value_counts(normalize=True) * 100)

plt.figure(figsize=(6, 5))
sns.countplot(x="Churn",data=train_df)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.countplot(x="Gender",hue="Churn",data=train_df)
plt.title("Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="Subscription Type",hue="Churn",data=train_df)

plt.title("Churn by Subscription Type")
plt.xlabel("Subscription Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="Contract Length",hue="Churn",data=train_df)
plt.title("Churn by Contract Length")
plt.xlabel("Contract Length")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data=train_df,x="Age",hue="Churn",kde=True)
plt.title("Age Distribution by Churn")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data=train_df,x="Tenure",hue="Churn",kde=True)
plt.title("Tenure Distribution by Churn")
plt.xlabel("Tenure")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data=train_df,x="Support Calls",hue="Churn",kde=True)
plt.title("Support Calls by Churn")
plt.xlabel("Number of Support Calls")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data=train_df,x="Payment Delay",hue="Churn",kde=True)
plt.title("Payment Delay by Churn")
plt.xlabel("Payment Delay")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

train_df["Churn"]=train_df["Churn"].astype(int)
test_df["Churn"]=test_df["Churn"].astype(int)
numeric_df=train_df.select_dtypes(include=np.number)

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(),annot=True,cmap="coolwarm",fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

X_train=train_df.drop(["Churn", "CustomerID"],axis=1)
y_train=train_df["Churn"]
X_test=test_df.drop(["Churn", "CustomerID"],axis=1)
y_test=test_df["Churn"]

print("\n FEATURES AND TARGET \n")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

categorical_columns=X_train.select_dtypes(include=["object"]).columns
numerical_columns=X_train.select_dtypes(include=np.number).columns

print("\nCategorical Columns:")
print(categorical_columns.tolist())

print("\nNumerical Columns:")
print(numerical_columns.tolist())

X_train=pd.get_dummies(X_train,columns=categorical_columns,drop_first=True)
X_test=pd.get_dummies(X_test,columns=categorical_columns,drop_first=True)
X_test=X_test.reindex(columns=X_train.columns,fill_value=0)

print("\nShape after encoding:")

print("Training:", X_train.shape)
print("Testing:", X_test.shape)

scaler = StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

print("\n LOGISTIC REGRESSION \n")

logistic_model=LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_scaled,y_train)
logistic_pred=logistic_model.predict( X_test_scaled)
logistic_probability=logistic_model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test,logistic_pred))

print("ROC-AUC:",round(roc_auc_score(y_test,logistic_probability),4))

print("\n KNN \n")
knn_model=KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled,y_train)
knn_pred=knn_model.predict(X_test_scaled)
knn_probability=knn_model.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test,knn_pred))
print("ROC-AUC:",round(roc_auc_score(y_test,knn_probability),4))

print("\n DECISION TREE \n")

tree_model=DecisionTreeClassifier(random_state=42,max_depth=5)
tree_model.fit(X_train,y_train)
tree_pred=tree_model.predict( X_test)
tree_probability=tree_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test,tree_pred))
print("ROC-AUC:",round(roc_auc_score(y_test,tree_probability),4))

print("\n RANDOM FOREST \n")

forest_model=RandomForestClassifier(n_estimators=200,max_depth=8,random_state=42)
forest_model.fit(X_train,y_train)
forest_pred=forest_model.predict(X_test)
forest_probability=forest_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test,forest_pred))
print("ROC-AUC:",round(roc_auc_score(y_test,forest_probability),4))

results=pd.DataFrame({
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
    ],
    "ROC-AUC": [
        roc_auc_score(y_test, logistic_probability),
        roc_auc_score(y_test, knn_probability),
        roc_auc_score(y_test, tree_probability),
        roc_auc_score(y_test, forest_probability)
    ]
})

print("\n MODEL COMPARISON \n")

print(results.to_string(index=False))

results_for_plot=results.set_index("Model")
results_for_plot[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
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
    cm=confusion_matrix(y_test,predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm,annot=True,fmt="d")
    plt.title(model_name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

feature_importance=pd.DataFrame({
    "Feature":X_train.columns,
    "Importance":forest_model.feature_importances_
})

feature_importance=feature_importance.sort_values(by="Importance",ascending=False)

print("\n TOP FEATURES \n")

print(feature_importance.head(10).to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10),x="Importance",y="Feature")
plt.title("Top 10 Important Features - Random Forest")
plt.tight_layout()
plt.show()

sample_customer=X_test.iloc[[0]]
sample_customer_scaled=scaler.transform(sample_customer)
sample_prediction=logistic_model.predict(sample_customer_scaled)
sample_probability=logistic_model.predict_proba(sample_customer_scaled)[0][1]

print("\n CUSTOMER PREDICTION \n")

if sample_prediction[0]==1:
    print("Prediction: Customer is likely to churn.")
else:
    print("Prediction: Customer is likely to stay.")

print("Churn Probability:",round(sample_probability * 100,2),"%")

print("CUSTOMER CHURN PROJECT COMPLETED")
