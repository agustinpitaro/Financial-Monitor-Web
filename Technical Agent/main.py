import matplotlib.pyplot as plt
import seaborn as sns

# Importar las clases definidas
# (Asumiendo que cada archivo .py está en el mismo directorio o configurado en PYTHONPATH)

def main():
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    from feature_selection import FeatureSelector
    from model_tuning import ModelTuner
    from model_execution import ModelExecutor

    # 1. Cargar datos
    loader = DataLoader(ticker="AAPL", start_date="2015-01-01", end_date="2020-01-01")
    df = loader.load_data()

    # 2. Ingeniería de características (indicadores)
    fe = FeatureEngineer(df)
    df = fe.add_technical_indicators()

    # 3. Definir la lista de features
    features = [
        'rsi','macd','macd_signal',
        'bb_mavg','bb_hband','bb_lband',
        'atr','obv',
        'adx','adx_pos','adx_neg',
        'stoch_k','stoch_d'
    ]

    # 4. Visualizar correlación (opcional)
    corr_matrix = df[features].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de Correlación de Features")
    plt.show()

    # 5. Selección de features
    fs = FeatureSelector(df, features)
    selected_features, dropped_features = fs.remove_highly_correlated(threshold=0.9)
    print("Features eliminadas:", dropped_features)
    print("Features seleccionadas:", selected_features)

    # 6. Dividir en train/test
    train_data = df.loc[:'2018-12-31']
    test_data  = df.loc['2019-01-01':]

    X_train = train_data[selected_features]
    y_train = train_data['target']
    X_test  = test_data[selected_features]
    y_test  = test_data['target']

    # 7. Ajuste de hiperparámetros (ModelTuner)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    tuner = ModelTuner(param_grid=param_grid, cv_splits=3)
    best_model, best_params, best_score = tuner.tune(X_train, y_train)

    print("\nMejores parámetros:", best_params)
    print("Mejor score de validación:", best_score)

    # 8. Ejecutar y evaluar el modelo final
    executor = ModelExecutor(best_model)
    train_acc = executor.evaluate(X_train, y_train)
    test_acc  = executor.evaluate(X_test, y_test)
    print(f"Accuracy en entrenamiento: {train_acc:.2f}")
    print(f"Accuracy en test: {test_acc:.2f}")

    # 9. Validación walk-forward en el set de test
    scores_wf = executor.walk_forward_validation(X_test, y_test, initial_train_size=100, test_size=30)
    if scores_wf:
        print("Walk-Forward Scores:", scores_wf)
        print("Walk-Forward Mean Accuracy:", sum(scores_wf) / len(scores_wf))

    # 10. Visualizar importancia de variables
    importances = best_model.feature_importances_
    plt.figure(figsize=(10,5))
    plt.bar(selected_features, importances, color='skyblue')
    plt.title("Importancia de Features (RandomForest)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()