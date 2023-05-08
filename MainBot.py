import Constants as keys
# from Market import MarketData
from binance.client import Client
import time
from datetime import datetime, timedelta
import requests
import json
import decimal
import pytz
import importlib
from importlib import reload 

import pandas as pd
import numpy as np
import talib
from binance.client import Client
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import csv


api_key = keys.API_KEY
api_secret = keys.API_SECRET
tz = pytz.timezone('Asia/Ho_Chi_Minh')
client = Client(api_key, api_secret)

def execute_trading_strategy():
      # Khởi tạo client của Binance
    api_key = keys.API_KEY
    api_secret = keys.API_SECRET
    client = Client(api_key, api_secret)
    print('Start Bot - 1st Scanning Data')
    while True:
            importlib.reload(keys)
            # print(keys.price_diff_order)
            
            time.sleep(5)
            today = datetime.now(tz).strftime("%d-%m-%y, %H:%M:%S")
            # Quét dữ liệu để đưa ra quyết định mua/bán
            decision,total_scores = evaluate_trading_decision()
            if 'Buy signal detected' in decision:
                alarm = f'BUY detected - {total_scores}' 
            elif 'Sell signal detected' in decision:
                alarm = f'SELL detected - {total_scores}'
            else:
                alarm = f'No signal detected - {total_scores}'

            # Kiểm tra vị thế hiện tại trên tài khoản
            position = check_position(client)
            # print(f"Position: {position['unRealizedProfit'] }")
            if position is None:
                
                print(f'No position found - {today} - {alarm}')
                # Kiểm tra danh sách lệnh đã đặt
                open_orders = check_open_orders(client)
                if open_orders:
                    
                   # Kiểm tra chênh lệch giá hiện tại và lệnh đặt
                    current_price = get_current_price()
                    order_prices = [float(order['price']) for order in open_orders]
                    maximum_order_price = max(order_prices) if decision == 'BUY' else min(order_prices)

                    for order in open_orders:
                        order_price = float(order['price'])
                        price_diff_percent = calculate_price_diff_percent(current_price, order_price)
                        print(f'Open order - Current Price: {current_price} - Order Price: {order_price} - Price Diff Percent: {round(price_diff_percent,2)}%')

                    print('=========================================================')

                    if abs(calculate_price_diff_percent(current_price, maximum_order_price)) >= keys.price_diff_order:
                        print(f'Price diff percent is greater than {keys.price_diff_order}% - Canceling open orders')
                        cancel_orders()
                        place_orders(client, decision)

                else:
                    print('No open orders found - Placing new orders')
                    place_orders(client, decision)
            else:
                print(f'Position found -{today} - {alarm} - Checking position')
                # Kiểm tra lợi nhuận
                open_orders = check_open_orders(client)
                if open_orders:
                     # Kiểm tra chênh lệch giá hiện tại và lệnh đặt
                    current_price = get_current_price()
                    order_prices = [float(order['price']) for order in open_orders]
                    maximum_order_price = max(order_prices) if decision == 'BUY' else min(order_prices)

                    for order in open_orders:
                        order_price = float(order['price'])
                        price_diff_percent = calculate_price_diff_percent(current_price, order_price)
                        print(f'Open order - Current Price: {current_price} - Order Price: {order_price} - Price Diff Percent: {round(price_diff_percent,2)}%')

                    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                
                unrealized_profit = position['unRealizedProfit']
                unrealized_profit = float(unrealized_profit)        
                pnl_percent = calculate_pnl_percent(position)
                print(f"{unrealized_profit} USD -  PnL Percent: {round(pnl_percent,2)} %")
                print('=========================================================')
                if pnl_percent > keys.pnl_percent or pnl_percent < -1*keys.pnl_percent:
                    print(f'PnL percent is greater than {pnl_percent}% or lost {-1*keys.pnl_percent} - Closing position')
                    close_position(client)
                    continue
                if unrealized_profit > keys.amount_takeprofit:
                    print(f'Unrealized profit is greater than {keys.amount_takeprofit} USD - Closing position')
                    close_position(client)
                    continue

                # # Kiểm tra biến động thị trường
                # market_conditions = check_market_conditions()
                # if market_conditions:
                #     print('Market conditions are not favorable - Closing position')
                #     close_position(client)
                #     continue

                # Kiểm tra giá hiện tại và đặt lệnh mua/bán
                #takeprofit
                current_price = get_current_price()
                current_price = float(current_price)
                if position['type'] == 'buy' and current_price >= float(position['entryPrice']) * (1+keys.price_diff_takeprofit/100):
                    print('Current price is greater than 10% of entry price - buying - Closing position')
                    close_position(client)
                    continue
                elif position['type'] == 'sell' and current_price <= float(position['entryPrice']) * (1-keys.price_diff_takeprofit/100):
                    print('Current price is less than 5% of entry price - selling - Closing position')
                    close_position(client)
                    continue

                #stoploss
                if position['type'] == 'buy' and current_price * (1+keys.price_diff_stoploss/100) <= float(position['entryPrice']):
                    print('Current price is greater than 10% of entry price - buying - Closing position')
                    close_position(client)
                    continue
                elif position['type'] == 'sell' and current_price * (1-keys.price_diff_stoploss/100) >= float(position['entryPrice']):
                    print('Current price is less than 5% of entry price - selling - Closing position')
                    close_position(client)
                    continue
def check_position(client):
    symbol = keys.symbol  # Thay thế bằng cặp giao dịch futures bạn muốn kiểm tra vị thế
    try:
        positions = client.futures_position_information(symbol=symbol)
        for position in positions:
            if float(position['positionAmt']) != 0:
                # Kiểm tra xem vị thế là mua hay bán
                if float(position['positionAmt']) > 0:
                    position['type'] = 'buy'
                else:
                    position['type'] = 'sell'
                return position
        return None
    except Exception as e:
        print('Error occurred while checking position:', e)
        return None



def evaluate_trading_decision():
    intervals = ['1m', '15m', '30m', '1h', '4h', '6h', '1d', '1w', '1M']
    avg_high_num, avg_low_num, uptrend_count, downtrend_count, sideway_count, buy_signal_count, sell_signal_count, rsi_values, stochastic_k, stochastic_d, fibonacci_retracement, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span = MarketData()

    # Đánh giá và quyết định mua/bán dựa trên các chỉ báo và dữ liệu đã tính toán
    decision = ""
     # Scoring weights for different indicators and signals
    uptrend_weight = 3
    downtrend_weight = 3
    sideway_weight = 1
    buy_signal_weight = 2
    sell_signal_weight = 2
    fibonacci_weight = 1
    rsi_weight = 1
    stochastic_weight = 1
    ichimoku_weight = 1

    # Calculate scores for trend analysis
    uptrend_score = uptrend_count * uptrend_weight
    downtrend_score = downtrend_count * downtrend_weight
    sideway_score = sideway_count * sideway_weight

    # Calculate scores for signal analysis
    buy_signal_score = buy_signal_count * buy_signal_weight
    sell_signal_score = sell_signal_count * sell_signal_weight

    # Calculate scores for Fibonacci retracement
    fibonacci_score = sum(fibonacci_retracement.values()) * fibonacci_weight

    # Calculate scores for RSI values
    rsi_score = sum(rsi_values.values()) * rsi_weight

    # Calculate scores for Stochastic indicators
    stochastic_scores = [stochastic_k[interval] + stochastic_d[interval] for interval in stochastic_k]
    stochastic_score = sum(stochastic_scores) * stochastic_weight
    # Calculate scores for Ichimoku Cloud
    ichimoku_scores = []
    for interval in tenkan_sen:
        if tenkan_sen[interval] is not None and kijun_sen[interval] is not None and senkou_span_a[interval] is not None and senkou_span_b[interval] is not None and chikou_span[interval] is not None:
            ichimoku_scores.append(tenkan_sen[interval] + kijun_sen[interval] + senkou_span_a[interval] + senkou_span_b[interval] + chikou_span[interval])
    ichimoku_score = sum(ichimoku_scores) * ichimoku_weight

    # Calculate the total score for each trend category
    total_scores = {
        "Uptrend": uptrend_score,
        "Downtrend": downtrend_score,
        "Sideway": sideway_score
    }
    # Find the trend category with the highest score
    max_score_category = max(total_scores, key=total_scores.get)

    # Make the decision based on the highest scoring trend category
    if max_score_category == "Uptrend":
        decision += "Uptrend detected.\n"
        decision += "Recommendation: Buy\n"
    elif max_score_category == "Downtrend":
        decision += "Downtrend detected.\n"
        decision += "Recommendation: Sell\n"
    else:
        decision += "Sideway trend detected.\n"
        decision += "Recommendation: Hold\n"

    # Make the decision based on the highest scoring signal category
    if buy_signal_score > sell_signal_score:
        decision += "Buy signal detected.\n"
    else:
        decision += "Sell signal detected.\n"
    decision += "Score: {}\n".format(total_scores)
    decision += "Average High: {}\n".format(avg_high_num)
    decision += "Average Low: {}\n".format(avg_low_num)

    decision += "\nRSI Values:\n"
    for interval, rsi_value in rsi_values.items():
        decision += "{}: {}\n".format(interval, rsi_value)

    decision += "\nStochastic Oscillator:\n"
    for interval, k_value in stochastic_k.items():
        d_value = stochastic_d[interval]
        decision += "{} - %K: {}, %D: {}\n".format(interval, k_value, d_value)

    decision += "\nFibonacci Retracement:\n"
    for interval, retracement_value in fibonacci_retracement.items():
        decision += "{}: {}\n".format(interval, retracement_value)

    decision += "\nIchimoku Cloud:\n"
    for interval in intervals:
        decision += "{} - Tenkan-sen: {}, Kijun-sen: {}, Senkou Span A: {}, Senkou Span B: {}, Chikou Span: {}\n".format(
            interval, tenkan_sen[interval], kijun_sen[interval], senkou_span_a[interval], senkou_span_b[interval], chikou_span[interval])

    return decision,total_scores

def place_orders(client, decision):
    symbol = keys.symbol  # Replace with the futures trading pair you want to trade
    buy_price_diff = keys.price_diff  # Buy price is 5% lower than the current price
    sell_price_diff = keys.price_diff  # Sell price is 5% higher than the current price
    Orderquantity = keys.Orderquantity  # Number of orders you want to place (e.g., 10 orders)
    multi = keys.multi  # Leverage multiplier

    account_info = client.futures_account()
    usdt_balance = float(account_info['totalWalletBalance'])

    current_price = float(client.futures_mark_price(symbol=symbol)['markPrice'])
    if usdt_balance <= 0:
        print("Insufficient funds in the account.")
        exit()

    # Calculate the order quantity based on 90% of the balance and the number of trades
    
    quantity_usdt = usdt_balance * keys.amountplay / Orderquantity
    

    # Temporary variables
    buy_price_adjustment = 0
    sell_price_adjustment = 0

    # Check the current price
   

    # Calculate the buy/sell price based on the deviation and adjustments
    buy_price = round(current_price * (1 - buy_price_diff - buy_price_adjustment), 2)
    sell_price = round(current_price * (1 + sell_price_diff + sell_price_adjustment), 2)
    quantity_sell = (quantity_usdt * multi/ sell_price) /Orderquantity
    quantity_buy = (quantity_usdt  * multi/ buy_price)/Orderquantity


    # Get the tick size
    tick_size = get_ticksize(symbol)
    step_size = 0.00001

    if tick_size:
        buy_price = get_trimmed_price(float(buy_price), float(tick_size))
        sell_price = get_trimmed_price(float(sell_price), float(tick_size))
        # quantity = get_trimmed_quantity(float(quantity), float(step_size))
       

    # Kiểm tra lệnh mua/bán dựa trên quyết định
    quantity_buy = adjust_precision(quantity_buy, 3)
    quantity_sell = adjust_precision(quantity_sell, 3)
    buy_price = adjust_precision(buy_price, 2)
    sell_price = adjust_precision(sell_price, 2)
    if 'Buy signal detected' in decision:
        print('Placing buy orders...')
        for i in range(Orderquantity):
            try:
                client.futures_create_order(
                    symbol=symbol,
                    side=Client.SIDE_BUY,
                    type=Client.ORDER_TYPE_LIMIT,
                    timeInForce=Client.TIME_IN_FORCE_GTC,
                    quantity=quantity_buy,
                    price=sell_price
                )
                
                buy_price_adjustment += (buy_price_diff*i)
                # buy_price = current_price * (1 - buy_price_diff - buy_price_adjustment)
                buy_price = current_price * (1 - buy_price_diff*(i+2)-buy_price_adjustment)
                if tick_size:
                    buy_price = round_step_size(float(buy_price), float(tick_size))
            except Exception as e:
                print('Error placing buy order:', e)
    elif 'Sell signal detected' in decision:
        print('Placing sell orders...')
        for i in range(Orderquantity):
            try:
                # quantity=0.001
                # sell_price=30346.01
                # print(quantity_sell,sell_price)
                client.futures_create_order(
                    symbol=symbol,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_LIMIT,
                    timeInForce=Client.TIME_IN_FORCE_GTC,
                    quantity=quantity_sell,
                    price=sell_price
                )
                
                sell_price_adjustment += (sell_price_diff*i)
                sell_price = current_price * (1 + sell_price_diff*(i+2)+sell_price_adjustment)
                if tick_size:
                    sell_price = round_step_size(float(sell_price), float(tick_size))
            except Exception as e:
                print('Error placing sell order:', e)
    else:
        print('No trading signal found.')


def calculate_pnl_percent(position):
    entry_price = float(position['entryPrice'])
    mark_price = float(position['markPrice'])
    # print(entry_price,mark_price)
    quantity = float(position['positionAmt'])
    if quantity > 0:
        pnl = (mark_price - entry_price) * quantity
    else:
        pnl = (entry_price - mark_price) * quantity

    pnl_percent = keys.multi*(pnl / (entry_price * quantity)) * 100
    
    
    return pnl_percent


def close_position(client):
     # Lấy thông tin vị thế hiện tại từ tài khoản futures
    positions = client.futures_position_information()

    # Kiểm tra và đóng vị thế hiện tại (nếu có)
    for position in positions:
        # symbol = position['BTCUSDT']
        symbol = keys.symbol
        position_amt = float(position['positionAmt'])

        if position_amt != 0:
            # Đặt lệnh đóng vị thế
            side = 'SELL' if position_amt > 0 else 'BUY'
            try:
                order = client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=abs(position_amt)
                )
                unrealized_profit = float(position['unRealizedProfit'])
                mark_price = float(position['markPrice'])
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                write_to_csv('pnl_data.csv', [timestamp, unrealized_profit, mark_price])
                print("Closed position: Symbol={}, Side={}, Quantity={}".format(symbol, side, abs(position_amt)))
            except Exception as e:
                print("Error closing position: {}".format(str(e)))
def check_market_conditions():
    api_key = keys.API_KEY
    api_secret = keys.API_SECRET
    client = Client(api_key, api_secret)

        # Lấy dữ liệu giá BTC/USDT của 100 cây nến gần nhất
    candles = client.futures_klines(symbol=keys.symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=100)
    # Tạo list chứa giá đóng cửa của các cây nến
    closes = [float(candle[4]) for candle in candles]
    # Tính giá trung bình động 10 nến gần nhất
    sma10 = sum(closes[-10:]) / 10
    # Tính giá trung bình động 30 nến gần nhất
    sma30 = sum(closes[-30:]) / 30
    # Nếu giá trung bình động 10 nến gần nhất vượt qua giá trung bình động 30 nến gần nhất
    if sma10 > sma30:
        return 'buy'
    # Nếu giá trung bình động 10 nến gần nhất thấp hơn giá trung bình động 30 nến gần nhất
    else:
        return 'sell'
def get_current_price():
    api_key = keys.API_KEY
    api_secret = keys.API_SECRET
    client = Client(api_key, api_secret)

    symbol = keys.symbol
    try:
        price_info = client.futures_mark_price(symbol=symbol)
        current_price = float(price_info['markPrice'])
        return current_price
    except Exception as e:
        print("Error getting current price: {}".format(str(e)))
        return None


def MarketData ():

    symbols = [keys.symbol]
    limit = 1000
    intervals = ['1m','15m','30m', '1h', '4h', '6h', '1d', '1w', '1M']
    start_time = '7 days ago'
    avg_high = []
    avg_low = []
    downtrend_count = 0
    uptrend_count = 0
    Sideway_count = 0
    buy_signal_count = 0
    sell_signal_count = 0
    rsi_values = {}
    stochastic_k = {}
    stochastic_d = {}
    fibonacci_retracement = {}
    tenkan_sen = {}
    kijun_sen = {}
    senkou_span_a = {}
    senkou_span_b = {}
    chikou_span = {}

    for symbol in symbols:
        for interval in intervals:
            # Lấy dữ liệu
            # print(f'Getting data of {symbol} at {interval} interval')
            
            url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
            response = requests.get(url)
            data = response.json()
            df = pd.DataFrame(data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
            df['Low_prev'] = 0
            df['High_prev'] = 0
            df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
            df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
            df.set_index('Open time', inplace=True)
            # Áp dụng các chỉ báo kỹ thuật
            df['EMA_12'] = talib.EMA(df['Close'], timeperiod=12)
            df['EMA_26'] = talib.EMA(df['Close'], timeperiod=26)
            df['MACD'], _, _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
            df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
            df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
            df['SMA_100'] = talib.SMA(df['Close'], timeperiod=100)
            df['Upper'], df['Middle'], df['Lower'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
                # Tính giá đỉnh, giá đáy
            df['High_roll'] = df['High'].rolling(window=21).max()
            df['Low_roll'] = df['Low'].rolling(window=21).min()
            # Tính toán tín hiệu giao dịch
            
            df['MACD_Signal'] = talib.EMA(df['MACD'], timeperiod=9)
            df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
            df['RSI_Signal'] = 50  # Tín hiệu RSI đơn giản là 50
            df['MA_Signal'] = (df['Close'].astype(float) > df['SMA_20']) & (df['SMA_20'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_100'])
            df['BB_Signal'] = (df['Close'].astype(float)< df['Lower']) | (df['Close'].astype(float) > df['Upper'])
        
            # Áp dụng chỉ báo Stochastic Oscillator
            df['%K'], df['%D'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

            # Áp dụng chỉ báo Fibonacci Retracement
            # Cần có dữ liệu giá cao nhất và giá thấp nhất trước đó để tính toán
            df['High'] = pd.to_numeric(df['High'], errors='coerce')
            df['Low_prev'] = pd.to_numeric(df['Low_prev'], errors='coerce')
            df['High_prev'] = pd.to_numeric(df['High_prev'], errors='coerce')
            df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
            df['High_prev'] = df['High'].shift(1)
            df['Low_prev'] = df['Low'].shift(1)
            df['Fibonacci_Retracement'] = (df['High'] - df['Low_prev']) / (df['High_prev'] - df['Low_prev'])

            # Áp dụng chỉ báo Ichimoku Cloud
            df['Tenkan_sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
            df['Kijun_sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
            df['Senkou_span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
            df['Senkou_span_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
            df['Chikou_span'] = df['Close'].shift(-26)
            if interval in ['1m', '15m', '30m', '1h', '4h', '6h', '1d', '1w', '1M']:
                rsi_values[interval] = df['RSI'].iloc[-1]  # Lưu giá trị RSI của khung thời gian vào dictionary                   # RSI
                

                # Stochastic Oscillator
                stochastic_k[interval] = df['%K'].iloc[-1]
                stochastic_d[interval] = df['%D'].iloc[-1]

                # Fibonacci Retracement
                fibonacci_retracement[interval] = df['Fibonacci_Retracement'].iloc[-1]

                # Ichimoku Cloud
                tenkan_sen[interval] = df['Tenkan_sen'].iloc[-1]
                kijun_sen[interval] = df['Kijun_sen'].iloc[-1]
                senkou_span_a[interval] = df['Senkou_span_A'].iloc[-1]
                senkou_span_b[interval] = df['Senkou_span_B'].iloc[-1]
                chikou_span[interval] = df['Chikou_span'].iloc[-1]


            # Xóa các cột không cần thiết
            df.drop(['High_prev', 'Low_prev'], axis=1, inplace=True)
                    
            last_row = df.iloc[-1]
            if last_row['MACD_Diff'] > 0:
                # print("Tín hiệu MACD: MUA")
                buy_signal_count += 1
            else:
                # print("Tín hiệu MACD: BÁN")
                sell_signal_count += 1

            if last_row['RSI'] > last_row['RSI_Signal']:
                # print(f"Tín hiệu RSI {interval}: MUA")
                buy_signal_count += 1
            else:
                # print(f"Tín hiệu RSI{interval}: BÁN")
                sell_signal_count += 1

            if last_row['MA_Signal']:
                # print("Tín hiệu MA: MUA")
                buy_signal_count += 1
            else:
                # print("Tín hiệu MA: BÁN")
                sell_signal_count += 1

            if last_row['BB_Signal']:
                # print("Tín hiệu Bollinger Bands: MUA")
                buy_signal_count += 1
            else:
                # print("Tín hiệu Bollinger Bands: BÁN")
                sell_signal_count += 1
            # Lấy dữ liệu 7 ngày gần nhất
            df_recent = df[-7:].copy()  # Make a copy of the DataFrame to avoid chained indexing
            df_recent['Close'] = df_recent['Close'].astype(float)
            df_recent.loc[:, 'SMA_20'] = df_recent['SMA_20'].astype(float)

            if (df_recent['Close'] > df_recent['SMA_20']).all():
                # print(f"{symbol} {interval}: Uptrend")
                uptrend_count += 1
            elif (df_recent['Close'] < df_recent['SMA_20']).all():
                # print(f"{symbol} {interval}: Downtrend")
                downtrend_count += 1
            else:
                # print(f"{symbol} {interval}: Sideway")
                Sideway_count += 1


            
            next_high = df_recent['High_roll'].max()
            avg_high.append(df_recent['High_roll'].max())
            avg_low.append(df_recent['Low_roll'].min())
            next_low = df_recent['Low_roll'].min()
    
    avg_high_num = sum(avg_high) / len(avg_high)
    avg_low_num = sum(avg_low) / len(avg_low)

    return avg_high_num, avg_low_num, uptrend_count, downtrend_count, Sideway_count, buy_signal_count, sell_signal_count,rsi_values, stochastic_k, stochastic_d, fibonacci_retracement, tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
def get_ticksize(symbol):
    info = client.futures_exchange_info()
    symbols = info['symbols']
    for s in symbols:
        if s['symbol'] == symbol:
            filters = s['filters']
            for f in filters:
                if f['filterType'] == 'PRICE_FILTER':
                    return f['tickSize']
    return None

def round_step_size(value, step_size):
    step_decimal = decimal.Decimal(str(step_size))
    return float((decimal.Decimal(round(value)) // step_decimal) * step_decimal)

def get_trimmed_quantity(quantity, step_size):
    trimmed_quantity = round(quantity / step_size) * step_size
    return trimmed_quantity

def get_trimmed_price(price, tick_size):
    trimmed_price = round(price / tick_size) * tick_size
    return trimmed_price
def adjust_precision(value, precision):
    return round(value, precision)
def check_open_orders(client):
    symbol = keys.symbol
    orders = client.futures_get_open_orders(symbol=symbol)
    return orders

def calculate_price_diff_percent(price1, price2):
    price_diff = price1 - price2
    price_diff_percent = (price_diff / price2) * 100
    return price_diff_percent
def cancel_orders():
    symbol = keys.symbol
    client = Client(api_key, api_secret)
    open_orders = client.futures_get_open_orders(symbol=symbol)
    for order in open_orders:
        order_id = order['orderId']
        symbol = order['symbol']
        
        result = client.futures_cancel_order(
            symbol=symbol,
            orderId=order_id
        )
        
        if result['status'] == 'CANCELED':
            print(f"Order {order_id} for symbol {symbol} has been canceled successfully")
        else:
            print(f"Failed to cancel order {order_id} for symbol {symbol}")

def write_to_csv(filename, data):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
execute_trading_strategy()
# print(check_open_orders(client))
# print(evaluate_trading_decision())
