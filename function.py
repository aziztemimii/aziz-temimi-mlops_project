import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


#####################################################
# 1. PREPARE DATA
#####################################################
def prepare_data():
    print("Préparation des données...")

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Données préparées !")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


#####################################################
# 2. TRAIN MODEL
#####################################################
def train_model(model_name, X_train, y_train):
    print(f"Entraînement du modèle : {model_name} ...")

    if model_name == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    elif model_name == "ada":
        model = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)

    elif model_name == "xgb":
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42,
        )

    else:
        raise ValueError("Modèle inconnu ! Utilisez : rf | ada | xgb")

    model.fit(X_train, y_train)
    print("Modèle entraîné !")

    return model


#####################################################
# 3. EVALUATE MODEL
#####################################################
def evaluate_model(model, X_test, y_test):
    print("Évaluation du modèle...")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy :", acc)
    print("Matrice de confusion :\n", cm)
    print("Classification report :\n", report)

    return acc, cm, report


#####################################################
# 4. SAVE MODEL + SCALER
#####################################################
def save_model(model, scaler, model_path="model.pkl", scaler_path="scaler.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print("Modèle et scaler sauvegardés !")


#####################################################
# 5. LOAD MODEL + SCALER
#####################################################
def load_model(model_path="model.pkl", scaler_path="scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Modèle chargé avec succès !")
    return model, scaler
