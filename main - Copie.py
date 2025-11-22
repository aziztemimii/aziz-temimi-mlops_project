from function import prepare_data, train_model, evaluate_model, save_model, load_model

if __name__ == "__main__":

    print("\n=== Pipeline ML étape par étape ===")
    print("1: Préparer les données")
    print("2: Entraîner le modèle")
    print("3: Évaluer le modèle")
    print("4: Sauvegarder le modèle")
    print("5: Charger le modèle")

    step = input("Entrez le numéro de l'étape à tester (1-5) : ").strip()

    # -------------------------
    # Etape 1 : PREPARE
    # -------------------------
    if step == "1":
        X_train, X_test, y_train, y_test, scaler = prepare_data()
        save_model(None, scaler, scaler_path="scaler.pkl")  # sauvegarde scaler
        print("✔ Données préparées (scaler sauvegardé) !")

    # -------------------------
    # Etape 2 : TRAIN
    # -------------------------
    elif step == "2":
        X_train, X_test, y_train, y_test, scaler = prepare_data()

        print("Choisissez le modèle :")
        print("1 = RandomForest")
        print("2 = AdaBoost")
        print("3 = XGBoost")
        choice = input("Votre choix : ")

        model_name = "rf" if choice == "1" else "ada" if choice == "2" else "xgb"

        model = train_model(model_name, X_train, y_train)

        # 🔥 IMPORTANT : on sauvegarde automatiquement
        save_model(model, scaler)
        print("✔ Modèle entraîné ET sauvegardé automatiquement !")

    # -------------------------
    # Etape 3 : EVALUATE
    # -------------------------
    elif step == "3":
        try:
            model, scaler = load_model()
        except:
            print("❌ Aucun modèle sauvegardé. Faites l'étape 2 d'abord.")
            exit()

        X_train, X_test, y_train, y_test, scaler = prepare_data()
        evaluate_model(model, X_test, y_test)

    # -------------------------
    # Etape 4 : SAVE MANUEL
    # -------------------------
    elif step == "4":
        try:
            model, scaler = load_model()
            save_model(model, scaler)
            print("✔ Modèle sauvegardé !")
        except:
            print("❌ Aucun modèle chargé ou entraîné.")

    # -------------------------
    # Etape 5 : LOAD
    # -------------------------
    elif step == "5":
        try:
            model, scaler = load_model()
            print("✔ Modèle chargé et prêt !")
        except:
            print("❌ Aucun modèle sauvegardé n’a été trouvé.")

    else:
        print("❌ Étape invalide !")
from function import prepare_data, train_model, evaluate_model, save_model, load_model

if __name__ == "__main__":

    print("\n=== Pipeline ML étape par étape ===")
    print("1: Préparer les données")
    print("2: Entraîner le modèle")
    print("3: Évaluer le modèle")
    print("4: Sauvegarder le modèle")
    print("5: Charger le modèle")

    step = input("Entrez le numéro de l'étape à tester (1-5) : ").strip()

    # -------------------------
    # Etape 1 : PREPARE
    # -------------------------
    if step == "1":
        X_train, X_test, y_train, y_test, scaler = prepare_data()
        save_model(None, scaler, scaler_path="scaler.pkl")  # sauvegarde scaler
        print("✔ Données préparées (scaler sauvegardé) !")

    # -------------------------
    # Etape 2 : TRAIN
    # -------------------------
    elif step == "2":
        X_train, X_test, y_train, y_test, scaler = prepare_data()

        print("Choisissez le modèle :")
        print("1 = RandomForest")
        print("2 = AdaBoost")
        print("3 = XGBoost")
        choice = input("Votre choix : ")

        model_name = "rf" if choice == "1" else "ada" if choice == "2" else "xgb"

        model = train_model(model_name, X_train, y_train)

        # 🔥 IMPORTANT : on sauvegarde automatiquement
        save_model(model, scaler)
        print("✔ Modèle entraîné ET sauvegardé automatiquement !")

    # -------------------------
    # Etape 3 : EVALUATE
    # -------------------------
    elif step == "3":
        try:
            model, scaler = load_model()
        except:
            print("❌ Aucun modèle sauvegardé. Faites l'étape 2 d'abord.")
            exit()

        X_train, X_test, y_train, y_test, scaler = prepare_data()
        evaluate_model(model, X_test, y_test)

    # -------------------------
    # Etape 4 : SAVE MANUEL
    # -------------------------
    elif step == "4":
        try:
            model, scaler = load_model()
            save_model(model, scaler)
            print("✔ Modèle sauvegardé !")
        except:
            print("❌ Aucun modèle chargé ou entraîné.")

    # -------------------------
    # Etape 5 : LOAD
    # -------------------------
    elif step == "5":
        try:
            model, scaler = load_model()
            print("✔ Modèle chargé et prêt !")
        except:
            print("❌ Aucun modèle sauvegardé n’a été trouvé.")

    else:
        print("❌ Étape invalide !")

