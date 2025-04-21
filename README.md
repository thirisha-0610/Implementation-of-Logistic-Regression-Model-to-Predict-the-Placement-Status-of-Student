# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: THIRISHA A
RegisterNumber: 212223040228
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data=pd.read_csv("/content/Placement_Data.csv") 
data.head()
```
![432115131-b1c2a378-6483-4335-8845-e134727f7aeb](https://github.com/user-attachments/assets/3ad32241-2811-4083-a90d-ce579bdedd5a)

```
data1=data.copy() 
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```

![432115290-f1cd4aea-f490-475d-9ee0-709e833b9dc1](https://github.com/user-attachments/assets/bb659689-9d53-43bd-aa9c-a72409b552df)

```
data1.isnull()
```
![432115596-1465be90-73f0-4fe3-b109-2b749e928768](https://github.com/user-attachments/assets/b17fe346-9103-4159-928e-aba3e6cd3a6d)

```
data1.duplicated().sum()
```
![432115639-db4985dc-f2d2-4039-b1e3-58561e795df8](https://github.com/user-attachments/assets/56354795-d3b1-4e75-bcda-4639b6688174)

```
le = LabelEncoder()
cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]
for col in cols:
    data1[col] = le.fit_transform(data1[col])
data1
```

![432115677-0621305c-498a-4b80-9073-cd118f071e5e](https://github.com/user-attachments/assets/9f8abaf7-a4f1-4f0c-9079-a7a219535dc7)

```
x = data1.iloc[:, :-1]
x
y = data1["status"]
y
```
![432116012-c79e2235-e526-4d3a-9fb0-28de323eacbd](https://github.com/user-attachments/assets/8929a7d8-bfb9-493d-8e5c-45ec8ce8a39e)

```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
```
![432116047-72291306-0315-400b-9e4e-8377c2fe3309](https://github.com/user-attachments/assets/e9d4cddc-49c4-4078-97e8-e93e6a71a821)

```
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
```
![432116062-9765d6c4-3c97-453d-9441-8c125af8ed32](https://github.com/user-attachments/assets/b70fdb50-23aa-4a1e-b504-b3fb78e26324)
```
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
```
![432116093-4cc75bca-4f2a-4e4f-82e2-10701d12b1fb](https://github.com/user-attachments/assets/6d4b4ec4-ec4a-4479-8990-c8f49c6f5ae7)

```
lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
```
![432116120-a1b83c6a-7cd9-4d79-b868-6865626d4ad6](https://github.com/user-attachments/assets/22ea9a5f-1fec-4918-b65f-486c2f455c1b)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
