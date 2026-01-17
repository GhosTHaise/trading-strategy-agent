import yfinance as yf

def main():
    ticker_symbol = "MSFT"

    # Create a Ticker object
    ticker = yf.Ticker(ticker_symbol)

    # Fetch historical market data for the last 30 days
    historical_data = ticker.history(period="1mo")  # data for the last month

    # Display a summary of the fetched data
    print(f"Summary of Historical Data for {ticker_symbol}:")
    print(historical_data[['Open', 'High', 'Low', 'Close', 'Volume']])


if __name__ == "__main__":
    main()
