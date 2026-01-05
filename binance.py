# app/data_providers/binance.py
import pandas as pd
import aiohttp
from typing import Optional, Dict, Any
from ..base import DataProvider

class BinanceDataProvider(DataProvider):
    BASE_URL = "https://api.binance.com/api/v3"
    
    async def connect(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.BASE_URL}/ping") as response:
                    if response.status == 200:
                        self.connected = True
                        return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect to Binance: {str(e)}")
            return False

    async def get_historical_data(self, symbol: str, interval: str, 
                                start_time: Optional[int] = None,
                                end_time: Optional[int] = None,
                                limit: int = 1000) -> pd.DataFrame:
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.BASE_URL}/klines", params=params) as response:
                    data = await response.json()
                    
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data from Binance: {str(e)}")
            raise