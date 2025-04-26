import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_and_save_models():
    # Load dataset
    df = pd.read_csv('data1_gear.csv')  # Adjust path as needed
    df = df.dropna()

    # Split features and target
    X = df.drop(columns='faulty')
    y = df['faulty']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # Initialize models
    svm_model = SVC(kernel='linear', random_state=42)
    log_reg_model = LogisticRegression(random_state=42)

    # Train the models
    svm_model.fit(X_train_scaled, y_train)
    log_reg_model.fit(X_train_scaled, y_train)

    # Save models and preprocessing objects
    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(log_reg_model, 'lr_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(imputer, 'imputer.pkl')

    # Evaluation (optional)
    svm_pred = svm_model.predict(X_test_scaled)
    log_reg_pred = log_reg_model.predict(X_test_scaled)

    print(f"SVM Accuracy: {accuracy_score(y_test, svm_pred):.2f}")
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, log_reg_pred):.2f}")

if __name__ == "__main__":
    train_and_save_models()
