import pandas as pd
import ta
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator

class FeatureEngineer:
    """
    Esta clase calcula indicadores técnicos y crea la columna 'target' 
    para un DataFrame que contiene múltiples tickers (columna 'Ticker').
    
    FUNCIONAMIENTO:
    1. Recibe en el constructor (self.data) un DataFrame con columnas:
       [Date, Open, High, Low, Close, Volume, Ticker, ...]
    2. El método `add_technical_indicators()` agrupa el DataFrame por
       la columna 'Ticker' para procesar cada activo por separado:
       - Calcula RSI, MACD, ATR, OBV, ADX, Stoch, etc.
       - Genera la columna 'target' = 1 si Close(t+1) > Close(t), sino 0.
    3. Retorna un DataFrame concatenado con todos los tickers, 
       pero cada uno procesado en su secuencia temporal individual.
    """

    def __init__(self, data: pd.DataFrame):
        # Guardamos una copia para evitar modificar el original
        self.data = data.copy()

    def add_technical_indicators(self) -> pd.DataFrame:
        """
        Aplica los indicadores técnicos a cada Ticker de manera independiente.
        Retorna un DataFrame con las columnas de indicadores y 'target'.
        """
        # Usamos groupby("Ticker") para que cada subset se procese sin mezclar datos de otros tickers
        df_processed = self.data.groupby("Ticker", group_keys=False)\
                                .apply(self._calc_indicators_for_group)
        return df_processed

    def _calc_indicators_for_group(self, df_subset: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores para un solo Ticker.
        Recibe un DF con filas correspondientes a un Ticker específico.
        Asegura el orden por fecha, genera la columna 'target'
        y elimina filas NaN producidas por los cálculos de indicadores.
        """
        # Ordenamos por fecha para no mezclar tiempos
        df_subset = df_subset.sort_values("Date")

        close_ = df_subset["Close"]
        high_  = df_subset["High"]
        low_   = df_subset["Low"]
        vol_   = df_subset["Volume"]

        # --- EJEMPLOS DE INDICADORES ---

        # 1. RSI
        df_subset["rsi"] = ta.momentum.rsi(close_, window=14)

        # 2. MACD y MACD signal
        df_subset["macd"] = ta.trend.macd(close_, window_slow=26, window_fast=12)
        df_subset["macd_signal"] = ta.trend.macd_signal(close_, window_slow=26, window_fast=12, window_sign=9)

        # 3. Bollinger Bands
        bb = BollingerBands(close=close_, window=20, window_dev=2)
        df_subset["bb_mavg"] = bb.bollinger_mavg()
        df_subset["bb_hband"] = bb.bollinger_hband()
        df_subset["bb_lband"] = bb.bollinger_lband()

        # 4. ATR (Average True Range)
        atr = AverageTrueRange(high_, low_, close_, window=14)
        df_subset["atr"] = atr.average_true_range()

        # 5. OBV (On-Balance Volume)
        obv = OnBalanceVolumeIndicator(close=close_, volume=vol_)
        df_subset["obv"] = obv.on_balance_volume()

        # 6. ADX (fuerza de la tendencia)
        adx = ADXIndicator(high_, low_, close_, window=14)
        df_subset["adx"] = adx.adx()
        df_subset["adx_pos"] = adx.adx_pos()
        df_subset["adx_neg"] = adx.adx_neg()

        # 7. Estocástico
        stoch = StochasticOscillator(high_, low_, close_, window=14, smooth_window=3)
        df_subset["stoch_k"] = stoch.stoch()
        df_subset["stoch_d"] = stoch.stoch_signal()

        # --- GENERACIÓN DE 'target' ---
        df_subset["target"] = (close_.shift(-1) > close_).astype(int)

        # Eliminamos filas que tengan NaN por cálculos de indicadores
        df_subset.dropna(inplace=True)

        return df_subset