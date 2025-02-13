import yfinance as yf
import pandas as pd

class DataLoader:
    """
    Clase encargada de descargar y preparar datos del mercado 
    a partir de yfinance (u otra fuente).
    """
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def load_data(self) -> pd.DataFrame:
        """
        Descarga datos históricos de yfinance y los retorna como un DataFrame.
        """
        print(f"Descargando datos de {self.ticker} desde {self.start_date} hasta {self.end_date} ...")
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return data