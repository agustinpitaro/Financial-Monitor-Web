import yfinance as yf
import pandas as pd
import os
import datetime as dt

def fetch_data(ticker, start_date, end_date, interval='1d'):
    """
    Fetch historical data for a given ticker from Yahoo Finance.

    Parameters:
    - ticker (str): Ticker symbol (e.g., 'AAPL' for Apple).
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - interval (str): Data interval ('1d', '1wk', '1mo', etc.).

    Returns:
    - df (DataFrame): Historical data for the ticker.
    """
    try:
        filename = f"{ticker} - {start_date.date()} - {end_date.date()}"
        # Create a Ticker object
        stock = yf.Ticker(ticker)

        # Fetch historical data
        df = stock.history(start=start_date, end=end_date, interval=interval)

        # Check if data is available
        if df.empty:
            print(f"No data found for ticker {ticker} within the provided date range.")
            return None

        # Reset index to make 'Date' a column
        df.reset_index(inplace=True)

        # If data is available, save it to a CSV file
        if df is not None:
            save_data_to_csv(df, ticker,filename)

        return df

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def save_data_to_csv(df, ticker, filename):
    """
    Save the DataFrame to a CSV file in the 'data/' folder.

    Parameters:
    - df (DataFrame): Historical data for the ticker.
    - ticker (str): Ticker symbol.
    """
    try:
        # Create the 'data' folder if it doesn't exist
        if not os.path.exists('../data'):
            os.makedirs('../data')

        # Define the filename with the current date
       
        file_name = f"../data/{filename}.csv"

        # Save the DataFrame to a CSV file
        df.to_csv(file_name, index=False)
        print(f"Data saved to {file_name}")

    except Exception as e:
        print(f"Error saving data for {ticker}: {e}")

def main():
    # Define parameters
    ticker = 'AAPL'  # Example: Apple Inc.
    start_date = dt.datetime.today() - dt.timedelta(365)
    end_date = dt.datetime.today()
    interval = '1d'  # Daily data
    # Fetch data
    df = fetch_data(ticker, start_date, end_date, interval)


if __name__ == "__main__":
    main()