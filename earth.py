
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv('earthquake_alert_balanced_dataset.csv')


data.drop_duplicates(inplace=True)


numeric_cols = data.select_dtypes(include=np.number).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

categorical_cols = data.select_dtypes(include="object").columns
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])


data["mag_depth_interaction"] = data["magnitude"] * data["depth"]
data["energy_approx"] = 10 ** (1.5 * data["magnitude"])


label_encoder = LabelEncoder()
data["alert"] = label_encoder.fit_transform(data["alert"])

joblib.dump(label_encoder, "label_encoder.pkl")


X = data.drop("alert", axis=1)
y = data["alert"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


cv_scores = cross_val_score(rf, X_scaled, y, cv=5)
print("Cross Validation Score:", cv_scores.mean())

# HYPERPARAMETER TUNING
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)


# FINAL MODEL

best_rf = grid.best_estimator_

final_pred = best_rf.predict(X_test)
print("Final Accuracy:", accuracy_score(y_test, final_pred))


# SAVE MODEL
joblib.dump(best_rf, "rf_model.pkl")

print("✅ Model, scaler, and encoder saved successfully!")
