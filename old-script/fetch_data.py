import os
import datetime as dt
import pandas as pd
import yfinance as yf
from typing import Optional, Union

def process_analysis(data: dict) -> pd.DataFrame:
    """
    Procesa el objeto de análisis proveniente de yfinance (generalmente un dict)
    y lo convierte en un DataFrame con una sola fila.

    Parámetros:
    - data (dict): Datos de análisis.

    Retorna:
    - pd.DataFrame: DataFrame con la información de análisis.
    """
    if isinstance(data, dict):
        return pd.DataFrame([data])
    else:
        raise ValueError("El parámetro 'data' no es un diccionario.")

def process_calendar(data: dict) -> pd.DataFrame:
    """
    Procesa el calendario (earnings, etc.) proveniente de yfinance (generalmente un dict),
    convirtiendo listas en cadenas separadas por comas.

    Parámetros:
    - data (dict): Datos del calendario.

    Retorna:
    - pd.DataFrame: DataFrame con la información del calendario.
    """
    if isinstance(data, dict):
        processed_data = {}
        for key, value in data.items():
            if isinstance(value, list):
                processed_data[key] = ", ".join(map(str, value))
            else:
                processed_data[key] = value
        return pd.DataFrame([processed_data])
    else:
        raise ValueError("El parámetro 'data' no es un diccionario.")

def process_info(data: dict) -> pd.DataFrame:
    """
    Procesa la información (info) proveniente de yfinance (generalmente un dict).
    Filtra solo las claves con valores planos (no listas ni diccionarios).

    Parámetros:
    - data (dict): Diccionario con la información de la acción.

    Retorna:
    - pd.DataFrame: DataFrame con la información filtrada.
    """
    if isinstance(data, dict):
        processed_data = {k: v for k, v in data.items() if not isinstance(v, (list, dict))}
        return pd.DataFrame([processed_data])
    else:
        raise ValueError("El parámetro 'data' no es un diccionario.")

def process_news(data: list) -> pd.DataFrame:
    """
    Procesa la variable 'news' proveniente de yfinance (generalmente una lista de diccionarios).
    Cada elemento de la lista representa una noticia.

    Parámetros:
    - data (list): Lista de diccionarios con información de noticias.

    Retorna:
    - pd.DataFrame: DataFrame con la información de las noticias.
    """
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return pd.DataFrame(data)
    else:
        raise ValueError("El parámetro 'data' no es una lista de diccionarios.")

def save_data_to_csv(df: pd.DataFrame, ticker: str, filename: str) -> None:
    """
    Guarda un DataFrame en un archivo CSV en la carpeta 'data' (ruta relativa al script).

    Parámetros:
    - df (pd.DataFrame): DataFrame con la información a guardar.
    - ticker (str): Símbolo del ticker (ej. 'AAPL').
    - filename (str): Nombre base del archivo (se le agregará extensión .csv).
    """
    try:
        # Obtiene la ruta base del script actual
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.join(base_path, "..", "data")

        # Crea la carpeta 'data' si no existe
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        # Define el nombre completo del archivo
        file_name = os.path.join(data_folder, f"{filename}.csv")

        # Guarda el DataFrame
        df.to_csv(file_name, index=False)
        print(f"[INFO] Datos guardados en: {file_name}")

    except Exception as e:
        print(f"[ERROR] Al guardar datos para {ticker}: {e}")

def fetch_data(
    ticker: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
    interval: str = '1d'
) -> Optional[pd.DataFrame]:
    """
    Descarga datos históricos y otra información relevante de un ticker usando yfinance,
    los procesa y los guarda en archivos CSV.

    Parámetros:
    - ticker (str): Símbolo del ticker (ej. 'AAPL').
    - start_date (dt.datetime): Fecha de inicio.
    - end_date (dt.datetime): Fecha de fin.
    - interval (str): Intervalo de datos (ej. '1d', '1wk', '1mo').

    Retorna:
    - pd.DataFrame o None: DataFrame con datos históricos OHCL, o None si hay algún error.
    """
    try:
        # Formateamos fechas para usarlas en los nombres de archivo
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        # Instancia del objeto de yfinance
        stock = yf.Ticker(ticker)

        # Descarga de datos históricos
        df_history = stock.history(start=start_date, end=end_date, interval=interval)

        # Si no hay datos, se interrumpe la ejecución
        if df_history.empty:
            print(f"[WARNING] No se encontraron datos para {ticker} en el rango proporcionado.")
            return None

        # Resetea el índice para que la columna 'Date' sea parte del DataFrame
        df_history.reset_index(inplace=True)

        # Descarga y procesa datos adicionales
        # Nota: cada método de yfinance puede retornar distintos tipos de datos
        dividends = stock.get_actions()          # Incluye dividendos y splits
        analysis = stock.get_analyst_price_targets()
        balance = stock.get_balance_sheet()
        calendar = stock.get_calendar()
        cashflow = stock.get_cashflow()
        info = stock.get_info()
        inst_holders = stock.get_institutional_holders()
        news = stock.get_news()
        recommendations = stock.get_recommendations()
        sustainability = stock.get_sustainability()

        # Procesa cada dataset si no está vacío y es el tipo esperado
        try:
            df_analysis = process_analysis(analysis) if analysis else pd.DataFrame()
        except ValueError as ve:
            print(f"[ERROR] Al procesar 'analysis' de {ticker}: {ve}")
            df_analysis = pd.DataFrame()

        try:
            df_calendar = process_calendar(calendar) if calendar else pd.DataFrame()
        except ValueError as ve:
            print(f"[ERROR] Al procesar 'calendar' de {ticker}: {ve}")
            df_calendar = pd.DataFrame()

        try:
            df_info = process_info(info) if info else pd.DataFrame()
        except ValueError as ve:
            print(f"[ERROR] Al procesar 'info' de {ticker}: {ve}")
            df_info = pd.DataFrame()

        try:
            df_news = process_news(news) if news else pd.DataFrame()
        except ValueError as ve:
            print(f"[ERROR] Al procesar 'news' de {ticker}: {ve}")
            df_news = pd.DataFrame()

        # Guarda los DataFrames resultantes en CSV
        # El sufijo indica la "categoría" de datos
        save_data_to_csv(df_history, ticker, f"{ticker}_{start_str}_{end_str}_history")
        save_data_to_csv(dividends, ticker, f"{ticker}_{start_str}_{end_str}_dividends")
        save_data_to_csv(df_analysis, ticker, f"{ticker}_{start_str}_{end_str}_analysis")
        save_data_to_csv(balance, ticker, f"{ticker}_{start_str}_{end_str}_balance")
        save_data_to_csv(df_calendar, ticker, f"{ticker}_{start_str}_{end_str}_calendar")
        save_data_to_csv(cashflow, ticker, f"{ticker}_{start_str}_{end_str}_cashflow")
        save_data_to_csv(df_info, ticker, f"{ticker}_{start_str}_{end_str}_info")
        save_data_to_csv(inst_holders, ticker, f"{ticker}_{start_str}_{end_str}_holders")
        save_data_to_csv(df_news, ticker, f"{ticker}_{start_str}_{end_str}_news")
        save_data_to_csv(recommendations, ticker, f"{ticker}_{start_str}_{end_str}_recs")
        save_data_to_csv(sustainability, ticker, f"{ticker}_{start_str}_{end_str}_sustain")

        return df_history

    except Exception as e:
        print(f"[ERROR] Al obtener datos para {ticker}: {e}")
        return None

def main() -> None:
    """
    Función principal para probar la descarga de datos.
    """
    # Parámetros
    ticker = 'AAPL'
    start_date = dt.datetime.today() - dt.timedelta(days=365)
    end_date = dt.datetime.today()
    interval = '1d'  # Datos diarios

    # Llamada a la función principal de descarga
    df = fetch_data(ticker, start_date, end_date, interval)

    if df is not None:
        print("[INFO] Descarga y guardado de datos completados con éxito.")
    else:
        print("[WARNING] No se pudo completar la descarga o no se encontraron datos.")

if __name__ == "__main__":
    main()