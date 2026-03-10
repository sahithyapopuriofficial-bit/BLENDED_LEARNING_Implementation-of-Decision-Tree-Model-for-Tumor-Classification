# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries such as pandas, sklearn, seaborn, and matplotlib.
2. Load the dataset tumor.csv and display the data and Separate the input features and target variable (Class).
3. Split the dataset into training and testing sets and Create a Decision Tree Classifier model.
4. Train the model using the training dataset and Predict the output for the test dataset.
5. Calculate the accuracy score and generate a classification report and Compute the confusion matrix.
6. Visualize the confusion matrix using a heatmap.

## Program:
```
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: POPURI SAHITHYA
RegisterNumber:  212225240106
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('tumor.csv')
print(data.head())
print(data.columns)
X=data.drop(['Class'], axis=1)
y=data['Class']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Name:POPURI SAHITHYA")
print("Register number: 212225240106")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))
conf_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="799" height="352" alt="image" src="https://github.com/user-attachments/assets/4e444c8e-2f42-408d-a452-234d95e95e7a" />

<img width="607" height="289" alt="image" src="https://github.com/user-attachments/assets/ad7a6bb1-ed4a-45f7-9aa1-4779e0ffcdef" />

<img width="775" height="579" alt="image" src="https://github.com/user-attachments/assets/46808d2b-896e-4c23-b035-a9e89df1154d" />

## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
