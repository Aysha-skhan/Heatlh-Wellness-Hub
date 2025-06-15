
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv("data.csv")

# Drop irrelevant or non-numeric columns
df.drop(columns=['Person ID', 'BMI Category'], inplace=True, errors='ignore')

# Convert categorical columns to numeric
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Occupation'] = df['Occupation'].astype('category').cat.codes

# Handle Blood Pressure (split into systolic and diastolic)
if 'Blood Pressure' in df.columns:
    df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df.drop(columns=['Blood Pressure'], inplace=True)
def is_healthy(row):
    if (
        6 <= row['Sleep Duration'] <= 9 and
        row['Quality of Sleep'] >= 3 and
        row['Physical Activity Level'] >= 2 and
        row['Stress Level'] <= 6 and
        60 <= row['Heart Rate'] <= 120 and
        row['Daily Steps'] >= 4000
    ):
        return "Healthy"
    else:
        return "Unhealthy"


# 👇 Apply that rule to create the new label
df['HealthLabel'] = df.apply(is_healthy, axis=1)

# Select features and target
X = df.drop(columns=['HealthLabel', 'Sleep Disorder'])
#print("Training features:", X.columns.tolist())
y = df['HealthLabel' ]
print("Training features:", X.columns.tolist())


#print(df['HealthLabel'].value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "lifestyle_model.pkl")

print("✅ Model trained and saved as lifestyle_model.pkl")
