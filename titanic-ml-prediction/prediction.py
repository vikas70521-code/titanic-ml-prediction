import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression

# ---------------------------------
# Loading the data
# ---------------------------------
data = pd.read_csv("train.csv")

# ---------------------------------
# Data Preprocessing
# ---------------------------------
data["Age"] = data["Age"].fillna(data["Age"].mean())
data = data.drop("Cabin", axis=1)
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# Coverting categorical to numeric
data = data.replace({"Sex": {"male" : 0, "female" : 1}, "Embarked" : {"S":0, "C":1, "Q":2}, "IsAlone" : {True :1, False :0}})

# -----------------------------
# Features & Target
# -----------------------------
x = data.drop(columns= ["PassengerId", "Ticket", "Name", "Survived", "Embarked"])
y = data["Survived"]
    
# -----------------------------
# Training the Model
# -----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=2)

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, Y_train)

def Survival_predict():
    name = input("Enter your Name:")
    pclass = int(input("Enter your passenger class : "))
    gender = int(input("Enter your Gender if male(0), if female(1): "))
    age = int(input("Enter your Age :"))
    sibsp = int(input("Enter Number of Siblings and Spouses you have :"))
    parch = int(input("Enter Number of Parents and children you have :"))
    fare = int(input("Enter your fare :"))

    features = pd.DataFrame([[pclass, gender, age, sibsp, parch, fare]], columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])
    return name, features

name, features = Survival_predict()
prediction = model_lr.predict(features)

if prediction[0] == 1:
    print(f"{name} you would have Survived")
else:
    print(f"{name} you would Not have Survived")



