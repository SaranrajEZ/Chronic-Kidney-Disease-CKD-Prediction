import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier

# Load dataset
dataset = pd.read_csv("cleaned_kidney_disease.csv")
dataset.head()

# One-hot encoding
df = pd.get_dummies(dataset, dtype=int, drop_first=True)
df.head()

# Separate independent and dependent variables
X = df.drop(['id', 'classification_notckd'], axis=1)
y = df['classification_notckd']

# Select top 5 features using chi2
kbest = SelectKBest(score_func=chi2, k=5)
X_kbest = kbest.fit_transform(X, y)
selected_features = X.columns[kbest.get_support()]
X_kbest = pd.DataFrame(X_kbest, columns=selected_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_kbest, y, test_size=0.25, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_features)
X_test = pd.DataFrame(scaler.transform(X_test), columns=selected_features)

# Initialize and train Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print(f"Decision Tree Selected Features: {selected_features}")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
print(classification_report(y_test, y_pred_dt))

# Initialize and train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print(f"SVM Selected Features: {selected_features}")
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")
print(classification_report(y_test, y_pred_svm))

# Initialize and train Logistic Regression model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print(f"Logistic Regression Selected Features: {selected_features}")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.2f}")
print(classification_report(y_test, y_pred_log))

# Initialize and train XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print(f"XGBoost Selected Features: {selected_features}")
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.2f}")
print(classification_report(y_test, y_pred_xgb))

# Initialize and train AdaBoost model
ada_model = AdaBoostClassifier()
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)
print(f"AdaBoost Selected Features: {selected_features}")
print(f"AdaBoost Accuracy: {accuracy_score(y_test, y_pred_ada):.2f}")
print(classification_report(y_test, y_pred_ada))

import pickle
# Save Random Forest model and scaler
with open('restaurant_Revenue_dt_model.pkl', 'wb') as model_file:
    pickle.dump(dt_model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Load the saved model and scaler
with open('restaurant_Revenue_dt_model.pkl', 'rb') as model_file:
    loaded_rf_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)


new_input = np.array([[0,140, 10, 1.20, 15]])  # Example: [Seating Capacity=100, Meal Price=20.5, Location_Rural=1, Cuisine_Japanese=0, Cuisine_Mexican=1]

# Scale the new input data
scaled_input = loaded_scaler.transform(new_input)

# Make the prediction using the trained model
prediction = loaded_rf_model.predict(scaled_input)

# Print the prediction
print("Prediction for the new input:", prediction)