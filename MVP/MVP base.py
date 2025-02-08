# Filename: agente_tecnico.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta  # Paquete de análisis técnico
# from xgboost import XGBClassifier  # Descomentar si deseas usar XGBoost

# 1. Descargar datos de Apple desde yfinance
data = yf.download("AAPL", start="2015-01-01", end="2020-01-01")
close_data = data['Close'].squeeze()
# 2. Calcular indicadores con 'ta'
# Ejemplo: RSI y MACD
data['rsi'] = ta.momentum.rsi(close_data, window=14)
data['macd'] = ta.trend.macd(close_data, window_slow=26, window_fast=12)
data['macd_signal'] = ta.trend.macd_signal(close_data, window_slow=26, window_fast=12, window_sign=9)

# (Puedes añadir más indicadores según necesites)

# 3. Crear la etiqueta (target) para predecir si el precio de mañana sube(1) o no(0)
data['target'] = (close_data.shift(-1) > close_data).astype(int)

# 4. Eliminar filas con NaN generados por indicadores o shift
data.dropna(inplace=True)

# 5. Separar en train y test
train_data = data.loc[:'2018-12-31']
test_data  = data.loc['2019-01-01':]

# 6. Seleccionar las columnas que usarás como features
features = ['rsi', 'macd', 'macd_signal']
X_train = train_data[features]
y_train = train_data['target']
X_test  = test_data[features]
y_test  = test_data['target']

# 7. Entrenar un modelo simple, p.ej. RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 8. Evaluar el modelo
accuracy_train = clf.score(X_train, y_train)
accuracy_test = clf.score(X_test, y_test)
print(f"Accuracy Train: {accuracy_train:.2f}")
print(f"Accuracy Test:  {accuracy_test:.2f}")

# 9. (Opcional) Visualizar
plt.figure(figsize=(10,5))
plt.plot(test_data.index, test_data['Close'], label='Precio Close')
plt.title('Precio de Apple (Test Set)')
plt.legend()
plt.show()