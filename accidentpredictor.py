# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:22:27 2024
@author: DELL
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import cross_val_score


np.random.seed(42)


num_samples = 1000
data = {
    'Weather_conditions': np.random.choice(['Clear', 'Rainy', 'Foggy', 'Snowy'], num_samples),
    'Light_conditions': np.random.choice(['Daylight', 'Night'], num_samples),
    'Road_surface_conditions': np.random.choice(['Dry', 'Wet', 'Snowy', 'Icy'], num_samples),
    'Type_of_collision': np.random.choice(['Head-on', 'Rear-end', 'Side-swipe', 'Intersection'], num_samples),
    'Day_of_week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], num_samples),
    'Sex_of_driver': np.random.choice(['Male', 'Female'], num_samples),
    'Educational_level': np.random.choice(['No education', 'Primary', 'Secondary', 'Higher'], num_samples),
    'Vehicle_driver_relation': np.random.choice(['Owner', 'Employee', 'Family', 'Friend'], num_samples),
    'Type_of_vehicle': np.random.choice(['Car', 'Motorcycle', 'Bus', 'Truck'], num_samples),
    'Owner_of_vehicle': np.random.choice(['Individual', 'Company'], num_samples),
    'Number_of_vehicles_involved': np.random.randint(1, 5, num_samples),
    'Driving_experience': np.random.choice(['Below 1yr', '1-2yr', '2-5yr', '5-10yr', 'Above 10yr'], num_samples),
    'Age_band_of_driver': np.random.choice(['Under 18', '18-30', '31-50', 'Over 50'], num_samples),
    'Service_year_of_vehicle': np.random.randint(1, 15, num_samples),
    'Number_of_casualties': np.random.randint(0, 5, num_samples),
    'Accident_severity': np.random.choice(['Slight Injury', 'Serious Injury', 'Fatal'], num_samples)
}

df = pd.DataFrame(data)

# Preview the dataset
print("Dataset Preview:")
print(df.head())


categorical_cols = [
    'Weather_conditions', 'Light_conditions', 'Road_surface_conditions',
    'Type_of_collision', 'Day_of_week', 'Sex_of_driver', 'Educational_level',
    'Vehicle_driver_relation', 'Type_of_vehicle', 'Owner_of_vehicle'
]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


experience_mapping = {
    'Below 1yr': 0.5, '1-2yr': 1.5, '2-5yr': 3.5, '5-10yr': 7.5, 'Above 10yr': 10
}
df['Driving_experience'] = df['Driving_experience'].replace(experience_mapping)

age_band_mapping = {
    'Under 18': 17, '18-30': 24, '31-50': 40, 'Over 50': 55
}
df['Age_band_of_driver'] = df['Age_band_of_driver'].replace(age_band_mapping)


severity_mapping = {'Slight Injury': 0, 'Serious Injury': 1, 'Fatal': 2}
df['Accident_severity'] = df['Accident_severity'].map(severity_mapping)


X = df.drop('Accident_severity', axis=1)
y = df['Accident_severity']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


smote_enn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)


param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'scale_pos_weight': [1, 2, 5]
}

xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, random_state=42)
random_search.fit(X_train_res, y_train_res)


best_model = random_search.best_estimator_


best_model.fit(X_train_res, y_train_res)

# Save the model
joblib.dump(best_model, "improved_xgb_accident_severity_model.pkl")
print("Model saved as 'improved_xgb_accident_severity_model.pkl'")


y_pred = best_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Slight Injury', 'Serious Injury', 'Fatal']))


cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validated accuracy: {np.mean(cv_scores):.2f}")

hypothetical_input = X_test.iloc[0:1]  
predicted_severity = best_model.predict(hypothetical_input)
print(f"Predicted Accident Severity for hypothetical input: {predicted_severity[0]}")
