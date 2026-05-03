# Titanic Survival Prediction (Machine Learning)

## Objective
Build a machine learning model to predict survival on the Titanic.

## Steps Performed
- Data cleaning and Preprocessing
- Feature Engineering (FamilySize, IsAlone, Age_group)
- Converted categorical data (Sex, Embarked)
- Train-test split

## Models Used
- Logistic Regression 
- Decision Tree 

## Results
- Logistic Regression:
  - Train Accuracy : 0.81
  - Test Accuracy : 0.78

- Decision Tree:
  - Before Tuning:
    - Train Accuracy : 0.98
    - Test Accuracy : 0.76

  - After Tuning (max_depth = 3):
    - Train Accuracy : 0.83
    - Test Accuracy : 0.78

## Key Learnings
- Logistic Regression generalizes better with smaller gap between train and test accuracy
- Decision Tree initially overfitted (larger gap between train and test accuracy)
- max_depth helps reduce overfitting in Decision Tree
- Features like Sex and Pclass have strong impact on Survival

## Tools Used
- Python
- Pandas
- Scikit_learn (sklearn)

## Dataset
- Source : Kaggle
- Name : Titanic Dataset

## Conclusion
Logistic Regression performed better due to better generalization, while Decision Tree required tuning to control overfitting