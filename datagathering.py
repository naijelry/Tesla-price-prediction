pip install alpaca-py
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from google.colab import files  # Import Colab's file handling module

# Alpaca API credentials
API_KEY = 'PKEGTI5D9EVKGJE5I40E'
SECRET_KEY = 'NrYmNN8sWL9etthfK9F2mLxCD49EyX2aORP7cBSm'  # Replace with your own Secret Key

# Initialize the Alpaca Data API client
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# Define the symbol, timeframe, and date range
symbol = 'TSLA'
timeframe = TimeFrame.Minute  # 1-minute interval for each data point
start_date = '2016-01-22'
end_date = '2016-01-24'  # Adjust this as per your needs

# Request historical data for the specified symbol and timeframe
request_params = StockBarsRequest(
    symbol_or_symbols=symbol,
    timeframe=timeframe,
    start=start_date,
    end=end_date
)
bars = data_client.get_stock_bars(request_params).df

# Ensure the data index (timestamp) is included as a column in the DataFrame
bars.reset_index(inplace=True)

# Remove timezone information from the timestamp column
bars['timestamp'] = bars['timestamp'].dt.tz_localize(None)

# Filter and reorder columns to match your specified structure
bars = bars[['timestamp', 'close', 'high', 'low', 'trade_count', 'open', 'volume', 'vwap']]

# Configure display settings to show all rows in the output
pd.set_option('display.max_rows', None)

# Display the entire DataFrame in Colab
print("Displaying all rows of data:")
print(bars)

# Save the DataFrame to an Excel file
excel_file = 'tesla_stock_data.xlsx'
bars.to_excel(excel_file, sheet_name='TeslaData', index=False)

# Download the file
files.download(excel_file) 