# -----------------------------
# Importing all the dependencies
# -----------------------------
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
form sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Loading the Data
# -----------------------------
data = pd.read_csv("train.csv")

# -----------------------------
# Data Preprocessing
# -----------------------------
# Finding Missing Values
print(data.isna().sum())

# Filling Missing Values
data["Age"] = data["Age"].fillna(data["Age"].mean())
data = data.drop("Cabin", axis=1)
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    
# Feature Engineering
data["FamilySize"] = data["SibSp"] + data["Parch"]
data["IsAlone"] = data["FamilySize"] == 0

def age_group(age):
    if age<18:
        return 0
    else:
        return 1
    
data["Age_group"] = data["Age"].apply(age_group)

# Coverting categorical to numeric
data = data.replace({"Sex": {"male" : 0, "female" : 1}, "Embarked" : {"S":0, "C":1, "Q":2}, "IsAlone" : {True :1, False :0}})


# -----------------------------
# Features & Target
# -----------------------------
X = data.drop(columns= ["PassengerId", "Ticket", "Name", "Survived"])
Y = data["Survived"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# -----------------------------
# Logistic Regression
# -----------------------------
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, Y_train)

# -----------------------------
# Decision Tree
# -----------------------------
model_dt = DecisionTreeClassifier(max_depth = 3)
model_dt.fit(X_train, Y_train)

# -----------------------------
# Random Forest
# -----------------------------
model_rf = RandomForestClassifier(max_depth = 3)
model_rf.fit(X_train, Y_train)

# -----------------------------
# Evaluation
# -----------------------------
# Linear Regression 
# Training Prediction and Accuracy
lr_train_prediction = model_lr.predict(X_train)
lr_train_accuracy = accuracy_score(Y_train, lr_train_prediction )

# Testing Prediction and Accuracy
lr_test_Prediction = model_lr.predict(X_test)
lr_test_accuracy = accuracy_score(Y_test, lr_test_Prediction)


# Decision Tree 
# Training Prediction and Accuracy
dt_train_prediction = model_dt.predict(X_train)
dt_train_accuracy = accuracy_score(Y_train, dt_train_prediction )

# Testing Prediction and Accuracy
dt_test_Prediction = model_dt.predict(X_test)
dt_test_accuracy = accuracy_score(Y_test, dt_test_Prediction)


# Random Forest
# Training Prediction and Accuracy
rf_train_prediction = model_rf.predict(X_train)
rf_train_accuracy = accuracy_score(Y_train, rf_train_prediction )

# Testing Prediction and Accuracy
rf_test_Prediction = model_rf.predict(X_test)
rf_test_accuracy = accuracy_score(Y_test, rf_test_Prediction)

# -----------------------------
# Results
# -----------------------------
print("Logistic Regression:")
print("Train Accuracy : ", lr_train_accuracy )
print("Test Accuracy : ", lr_test_accuracy)

print("Decision Tree:")
print("Train Accuracy : ", dt_train_accuracy )
print("Test Accuracy : ", dt_test_accuracy)

print("Random Forest:")
print("Train Accuracy : ", rf_train_accuracy )
print("Test Accuracy : ", rf_test_accuracy)
