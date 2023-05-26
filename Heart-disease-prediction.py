import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix ,mean_squared_error
from sklearn.preprocessing import LabelEncoder

Df = pd.read_csv("/content/Heart_Disease.csv")
#checking nulls
Df.isna().sum()

Df["Gender"].fillna(Df["Gender"].mode()[0], inplace=True)
Df["smoking_status"].fillna(Df["smoking_status"].mode()[0], inplace=True)
Df["work_type"].fillna(Df["work_type"].mode()[0], inplace=True)
imputer = KNNImputer(n_neighbors=5)  # Adjust the number of neighbors as per your preference
Df['Age'] = imputer.fit_transform(Df[['Age']])

Df.isna().sum()

dummy = pd.get_dummies(Df['Gender'])
Df = pd.concat((Df,dummy),axis=1)
Df = Df.drop(['Gender' , 'Male'] , axis=1)

dummy = pd.get_dummies(Df['Heart Disease'])
Df = pd.concat((dummy,Df) , axis=1 )
Df = Df.drop(['Heart Disease','No'],axis=1)
Df = Df.rename(columns={"Yes":"Heart Disease"})

label_encoder=LabelEncoder()

Df["smoking_status"]=label_encoder.fit_transform(Df["smoking_status"])
Df["work_type"]=label_encoder.fit_transform(Df["work_type"])

Df.info()

# Get the number of rows and columns
num_rows, num_cols = Df.shape

print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_cols}")
Df.head()

Df.head()

plt.figure(figsize = (15,8))
sns.heatmap(Df.iloc[:,:].corr(), annot=True)

Df.head()

# Get the number of rows and columns
num_rows, num_cols = Df.shape

print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_cols}")
Df.head()

print(Df.columns)
Df.isna().sum()

# Split the data into features and target variable
X = Df.drop('Heart Disease' , axis=1)
y = Df['Heart Disease']

# Perform feature selection using information gain
k = 5 # Number of top features to select
selector = SelectKBest(score_func=mutual_info_classif, k=k)
X_new = selector.fit_transform(X, y)

# Get the selected feature indices
selected_indices = selector.get_support(indices=True)

# Get the names of the selected features
selected_features = X.columns[selected_indices]

# Print the selected features
print("Selected Features:")
print(selected_features)

# Specify the columns of interest
columns_of_interest = ['Number of vessels fluro', 'Chest pain type', 'Cholesterol', 'Max HR', 'ST depression']

# Create a figure with a specific size
plt.figure(figsize=(15, 8))

# Create the box plot using seaborn
sns.boxplot(data=Df[columns_of_interest])

# Set the x-axis labels
plt.xticks(range(len(columns_of_interest)), columns_of_interest)

# Set the plot title
plt.title('Box Plot of Selected Columns')

# Show the plot
plt.show()

for col in Df.columns:
    q25, q75 = np.percentile(Df[col], [25, 75])
    iqr = q75 - q25
    lower, upper = q25 - (iqr * 1.5), q75 + (iqr * 1.5)
    outliers = (Df[col] < lower) | (Df[col] > upper)
    
    if outliers.any():
        mean_value = Df[~outliers][col].mean()
        Df.loc[outliers, col] = mean_value
        print(f'Replaced {sum(outliers)} outliers in {col} with the mean value: {mean_value}')
    else:
        print(f'No outliers found in {col}')



# Specify the columns of interest
columns_of_interest = ['Number of vessels fluro', 'Chest pain type', 'Cholesterol', 'Max HR', 'ST depression']

# Create a figure with a specific size
plt.figure(figsize=(15, 8))

# Create the box plot using seaborn
sns.boxplot(data=Df[columns_of_interest])

# Set the x-axis labels
plt.xticks(range(len(columns_of_interest)), columns_of_interest)

# Set the plot title
plt.title('Box Plot of Selected Columns')

# Show the plot
plt.show()

Df.describe()

# Split the data into features and target variable
X = Df.drop('Heart Disease' , axis=1)
y = Df['Heart Disease']

# Create a new dataset (X_selected) by selecting only the columns corresponding to the selected features
selected_features = ['Chest pain type', 'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium']
X_selected = Df[selected_features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=41)

# Create a Decision Tree classifier
dt_model = DecisionTreeClassifier()

# Train the model on the training set
dt_model.fit(X_train, y_train)

# Predict the classes of the testing set
y_pred = dt_model.predict(X_test)

# Evaluate the model's accuracy on testing set
aaccuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {aaccuracy}")

# Predict the classes of the training set
y_train_pred = dt_model.predict(X_train)

# Calculate the training accuracy
atraining_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {atraining_accuracy}")
#############
cr=classification_report(y_test, y_pred)
print(cr)
################

cm = confusion_matrix(y_test, y_pred)
cm

#############
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error

# Split the data into features and target variable
X = Df.drop('Heart Disease' , axis=1)
y = Df['Heart Disease']

# Create a new dataset (X_selected) by selecting only the columns corresponding to the selected features
selected_features = ['Chest pain type', 'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium']
X_selected = Df[selected_features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=41)

# Create a Decision Tree classifier with max_depth hyperparameter
dt_model = DecisionTreeClassifier(max_depth=5)

# Train the model on the training set
dt_model.fit(X_train, y_train)

# Predict the classes of the testing set
y_pred = dt_model.predict(X_test)

# Evaluate the model's accuracy on the testing set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Predict the classes of the training set
y_train_pred = dt_model.predict(X_train)

# Calculate the training accuracy
training_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {training_accuracy}")

# Print classification report
cr = classification_report(y_test, y_pred)
print(cr)

# Calculate and print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Split the data into features (X) and the target variable (y)
X = Df.drop('Heart Disease' , axis=1)
y = Df['Heart Disease']

# Create a new dataset (X_selected) by selecting only the columns corresponding to the selected features
selected_features = ['Chest pain type', 'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium']
X_selected = Df[selected_features]

# Split the selected data and the target variable into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=33)

# Train the model on the training set
rf_model.fit(X_train, y_train)

# Predict the classes of the testing set
y_pred = rf_model.predict(X_test)

# Evaluate the model's accuracy on the testing set
baccuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {baccuracy}")

# Predict the classes of the training set
y_train_pred = rf_model.predict(X_train)

# Calculate the training accuracy
btraining_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {btraining_accuracy}")
#############
cr=classification_report(y_test, y_pred)
print(cr)
################

cm = confusion_matrix(y_test, y_pred)
cm

#############
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Split the data into features and target variable
X = Df.drop('Heart Disease' , axis=1)
y = Df['Heart Disease']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=39)

# Create the SVM model with a linear kernel
svm_model = SVC(kernel='linear')
 


# Train the model on the training set
svm_model.fit(X_train, y_train)

# Predict the classes of the testing set
y_pred = svm_model.predict(X_test)

# Evaluate the model's accuracy on testing set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Predict the classes of the training set
y_train_pred = svm_model.predict(X_train)

# Calculate the training accuracy
training_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {training_accuracy}")

cr=classification_report(y_test, y_pred)
print(cr)


cm = confusion_matrix(y_test, y_pred)
cm


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Split the data into features and target
X = Df.drop('Heart Disease' , axis=1)
y = Df['Heart Disease']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=30)
# Initialize the model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
print(f"Training Accuracy: {train_score}")
# Evaluate the model on the testing data
score = model.score(X_test, y_test)
print(f"Accuracy: {score}")

cr=classification_report(y_test, y_pred)
print(cr)


cm = confusion_matrix(y_test, y_pred)
cm


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Split the data into features and target
X = Df.drop('Heart Disease' , axis=1)
y = Df['Heart Disease']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Initialize the model with a regularization parameter (C) of 0.1
model = LogisticRegression(C=0.1)

# Train the model on the training data
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
print(f"Training Accuracy: {train_score}")

# Evaluate the model on the testing data
score = model.score(X_test, y_test)
print(f"Accuracy: {score}")

# Predict the classes of the testing set
y_pred = model.predict(X_test)

# Print the classification report
cr = classification_report(y_test, y_pred)
print(cr)

# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)