import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Import de las clases
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    from feature_selection import FeatureSelector
    from model_tuning import ModelTuner
    from model_execution import ModelExecutor

    # 1. Definir tickers y rango de fechas
    TICKERS = ["AAPL", "MSFT", "GOOG"]
    loader = DataLoader(tickers=TICKERS, start_date="2020-01-01", end_date="2025-01-01")

    # 2. Cargar datos en un solo DataFrame con columna 'Ticker'
    df_all = loader.load_data_multi()
    # df_all => columnas: [Date, Open, High, Low, Close, Adj Close, Volume, Ticker]

    # 3. Ingeniería de características (indicadores + target)
    fe = FeatureEngineer(df_all)
    df_feat = fe.add_technical_indicators()

    # 4. Convertir Ticker en categoría => luego a código numérico
    df_feat["Ticker"] = df_feat["Ticker"].astype("category")
    df_feat = df_feat.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    
    # (Opcional) Creas un Ticker_code para que el modelo sepa de cuál activo se trata
    df_feat["Ticker_code"] = df_feat["Ticker"].cat.codes

    # 5. Definir lista inicial de features
    base_features = [
        "rsi","macd","macd_signal",
        "bb_mavg","bb_hband","bb_lband",
        "atr","obv","adx","adx_pos","adx_neg",
        "stoch_k","stoch_d", 
        # Incluir Ticker_code
        "Ticker_code"
    ]

    # 6. Visualizar correlación (opcional)
    corr_matrix = df_feat[base_features].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
    plt.title("Matriz de Correlación de Features (Multiticker)")
    plt.show()

    # 7. Selección de features (colinealidad)
    fs = FeatureSelector(df_feat, base_features)
    selected_feats, dropped_feats = fs.remove_highly_correlated(threshold=0.9)
    print("Features eliminadas por correlación alta:", dropped_feats)
    print("Features finales:", selected_feats)

    # 8. Dividir en Train y Test por fecha (respetando la cronología)
    train_data = df_feat.loc[df_feat["Date"] <= "2022-12-31"]
    test_data  = df_feat.loc[df_feat["Date"] >  "2022-12-31"]

    X_train = train_data[selected_feats]
    y_train = train_data["target"]
    X_test  = test_data[selected_feats]
    y_test  = test_data["target"]

    # 9. Definir grilla de hiperparámetros
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    # 10. Ajuste de hiperparámetros con GridSearchCV y TimeSeriesSplit
    tuner = ModelTuner(param_grid=param_grid, cv_splits=3)
    best_model, best_params, best_score = tuner.tune(X_train, y_train)

    print("Mejores parámetros:", best_params)
    print("Mejor score (cv):", best_score)

    # 11. Evaluar modelo final
    executor = ModelExecutor(best_model)
    acc_train = executor.evaluate(X_train, y_train)
    acc_test  = executor.evaluate(X_test, y_test)
    print(f"Accuracy en entrenamiento: {acc_train:.2f}")
    print(f"Accuracy en test: {acc_test:.2f}")

    # 12. Validación Walk-Forward (Opcional)
    wf_scores = executor.walk_forward_validation(X_test, y_test, initial_train_size=100, test_size=30)
    if wf_scores:
        print("Walk-Forward scores:", wf_scores)
        print("Media WF accuracy:", sum(wf_scores)/len(wf_scores))

    # 13. (Opcional) Graficar importancia de features
    importances = best_model.feature_importances_
    plt.figure(figsize=(10,4))
    plt.bar(selected_feats, importances, color='skyblue')
    plt.title("Importancia de Features (RandomForest)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
