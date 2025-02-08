import yfinance as yf
import pandas as pd

class DataLoader:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def load_data_multi(self) -> pd.DataFrame:
        """
        Descarga datos de cada ticker en su propio DataFrame,
        añade la columna 'Ticker', y concatena en formato largo.
        """
        all_data = []
        for t in self.tickers:
            print(f"Descargando {t} de {self.start_date} a {self.end_date}")
            # DESCARGA SOLO UN TICKER A LA VEZ
            df_t = yf.download(t, start=self.start_date, end=self.end_date, progress=False)
            # Añadimos la columna Ticker
            print(df_t.columns)
            df_t["Ticker"] = t
            print(df_t.columns)
            df_t.reset_index(inplace=True)
            all_data.append(df_t)
            print(all_data.columns)

        # Concatenamos en 'formato largo': un solo DataFrame
        df_all = pd.concat(all_data)
        print(df_all.columns)
        # Reseteamos índice para que 'Date' sea columna normal
        df_all.reset_index(inplace=True)  # Devuelve 'Date' como columna
        return df_all
