import ta
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from ta.trend import ADXIndicator
from ta.momentum import StochasticOscillator
import pandas as pd

class FeatureEngineer:
    """
    Clase para encapsular la lógica de ingeniería de características,
    como la generación de indicadores técnicos.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()  # Para no mutar el original
    
    def add_technical_indicators(self) -> pd.DataFrame:
        """
        Añade indicadores técnicos y crea la columna 'target'.
        Retorna el DataFrame modificado.
        """
        close_data = self.data['Close'].squeeze()
        high_data  = self.data['High'].squeeze()
        low_data   = self.data['Low'].squeeze()
        volume_data = self.data['Volume'].squeeze()

        # Ejemplo de algunos indicadores
        self.data['rsi'] = ta.momentum.rsi(close_data, window=14)
        self.data['macd'] = ta.trend.macd(close_data, window_slow=26, window_fast=12)
        self.data['macd_signal'] = ta.trend.macd_signal(close_data, window_slow=26, window_fast=12, window_sign=9)

        bb = BollingerBands(close=close_data, window=20, window_dev=2)
        self.data['bb_mavg'] = bb.bollinger_mavg()
        self.data['bb_hband'] = bb.bollinger_hband()
        self.data['bb_lband'] = bb.bollinger_lband()

        atr = AverageTrueRange(high=high_data, low=low_data, close=close_data, window=14)
        self.data['atr'] = atr.average_true_range()

        obv = OnBalanceVolumeIndicator(close=close_data, volume=volume_data)
        self.data['obv'] = obv.on_balance_volume()

        adx = ADXIndicator(high=high_data, low=low_data, close=close_data, window=14)
        self.data['adx'] = adx.adx()
        self.data['adx_pos'] = adx.adx_pos()
        self.data['adx_neg'] = adx.adx_neg()

        stoch = StochasticOscillator(high=high_data, low=low_data, close=close_data, window=14, smooth_window=3)
        self.data['stoch_k'] = stoch.stoch()
        self.data['stoch_d'] = stoch.stoch_signal()

        # Generar variable target (sube el precio mañana respecto a hoy)
        self.data['target'] = (close_data.shift(-1) > close_data).astype(int)

        self.data.dropna(inplace=True)  # Quitar filas con NaN
        return self.data