import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import joblib

# 1. Load the dataset
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset file not found. Please download 'WA_Fn-UseC_-Telco-Customer-Churn.csv' and place it in the same directory.")
    exit()

# 2. Data Cleaning and Preprocessing
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

X = df.drop('Churn', axis=1)
y = df['Churn']

categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(exclude=['object']).columns

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# 3. Model Training with Imbalance Handling (SMOTE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a pipeline with SMOTE and XGBoost
model = ImbPipeline(steps=[('preprocessor', preprocessor),
                           ('smote', SMOTE(sampling_strategy='minority', random_state=42)),
                           ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))])

model.fit(X_train, y_train)
print("\nModel trained successfully with SMOTE and XGBoost.")

# 4. Save the trained model and features
joblib.dump(model, 'churn_predictor.joblib')
joblib.dump(list(X.columns), 'features.joblib')
print("Model and feature list saved as 'churn_predictor.joblib' and 'features.joblib'.")