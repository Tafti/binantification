from __future__ import absolute_import
import numpy as np
import pandas as pd
from binance.enums import *
from binance.client import Client
from binance.websockets import BinanceSocketManager
from pyti import catch_errors
from pyti.function_helper import fill_for_noncomputable_vals
from six.moves import range
from six.moves import zip
import matplotlib.pyplot as plt
import telegram_send
from telegram import bot
import time
from binance.client import Client
# requires dateparser package
import dateparser
import pytz
from datetime import datetime

api_key = ""
api_secret = ""
client = Client(api_key, api_secret)

def date_to_milliseconds(date_str):

    # get epoch value in UTC
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = dateparser.parse(date_str)
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)
def interval_to_milliseconds(interval):
    """Convert a Binance interval string to milliseconds
    :param interval: Binance interval string 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w
    :type interval: str
    :return:
         None if unit not one of m, h, d or w
         None if string not in correct format
         int value of interval in milliseconds
    """
    ms = None
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60
    }

    unit = interval[-1]
    if unit in seconds_per_unit:
        try:
            ms = int(interval[:-1]) * seconds_per_unit[unit] * 1000
        except ValueError:
            pass
    return ms
def get_historical_klines(symbol, interval, start_str, end_str=None):

    # create the Binance client, no need for api key
    client = Client("", "")

    # init our list
    output_data = []

    # setup the max limit
    limit = 500

    # convert interval to useful value in seconds
    timeframe = interval_to_milliseconds(interval)

    # convert our date strings to milliseconds
    start_ts = date_to_milliseconds(start_str)

    # if an end time was passed convert it
    end_ts = None
    if end_str:
        end_ts = date_to_milliseconds(end_str)

    idx = 0
    # it can be difficult to know when a symbol was listed on Binance so allow start time to be before list date
    symbol_existed = False
    while True:
        # fetch the klines from start_ts up to max 500 entries or the end_ts if set
        temp_data = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            startTime=start_ts,
            endTime=end_ts
        )

        # handle the case where our start date is before the symbol pair listed on Binance
        if not symbol_existed and len(temp_data):
            symbol_existed = True

        if symbol_existed:
            # append this loops data to our output data
            output_data += temp_data

            # update our start timestamp using the last value in the array and add the interval timeframe
            start_ts = temp_data[len(temp_data) - 1][0] + timeframe
        else:
            # it wasn't listed yet, increment our start date
            start_ts += timeframe

        idx += 1
        # check if we received less than the required limit and exit the loop
        if len(temp_data) < limit:
            # exit the while loop
            break

        # sleep after every 3rd call to be kind to the API
        if idx % 3 == 0:
            time.sleep(1)

    return output_data
##klines = get_historical_klines("ETHBTC", Client.KLINE_INTERVAL_30MINUTE, "4 Nov, 2018", "6 Nov, 2018")
def coin_prices(symbol_info):
    #Will print to screen, prices of coins on 'watch list'
    #returns all prices
    prices = client.get_all_tickers()
    #print("\nSelected (watch list) Ticker Prices: ")
    for price in prices:
        if price['symbol'] in ["ETHBTC"]:
            #print("\n" "" + price['symbol'] + " " "Prices: ")
            #print(price["price"])
            r = price
    return r
coins = client.get_all_tickers()
list_coins = []
for coin in coins:
    suffix = "BTC";
    if coin['symbol'].endswith(suffix):
       list_coins.append(coin['symbol'])
symbol = list_coins
#coin_symbol_df = pd.Series(symbol)
#print(pd.Series(symbol))

#strs = ["Client.KLINE_INTERVAL_2HOUR" for x in range(len(symbol))]

#df_data = pd.DataFrame({'symbol': symbol, 'RSI': 30*np.ones(len(symbol)) ,'interval': strs })

#df_data.to_csv('coin_symbol.csv')

df = pd.read_csv('coin_symbol.csv')
with open('coin_symbol.csv') as f:
    row_count = sum(1 for line in f)
signal_col = np.zeros(row_count-1)
while True:

        k = 0

        for index, row in df.iterrows():



            klines = get_historical_klines(row["symbol"], eval(row["intervals"]) , "30 day ago UTC")

            klines_close_list = [float(i[4]) for i in klines]



            def check_for_period_error(data, period):
                """
                Check for Period Error.
                This method checks if the developer is trying to enter a period that is
                larger than the data set being entered. If that is the case an exception is
                raised with a custom message that informs the developer that their period
                is greater than the data set.
                """
                period = int(period)
                data_len = len(data)
                if data_len < period:
                    raise Exception("Error: data_len < period")

            def check_for_input_len_diff(*args):
                """
                Check for Input Length Difference.
                This method checks if multiple data sets that are inputted are all the same
                size. If they are not the same length an error is raised with a custom
                message that informs the developer that the data set's lengths are not the
                same.
                """
                arrays_len = [len(arr) for arr in args]
                if not all(a == arrays_len[0] for a in arrays_len):
                    err_msg = ("Error: mismatched data lengths, check to ensure that all "
                               "input data is the same length and valid")
                    raise Exception(err_msg)

            def relative_strength_index(data, period):
                """
                Relative Strength Index.
                Formula:
                RSI = 100 - (100 / 1 + (prevGain/prevLoss))
                """
                catch_errors.check_for_period_error(data, period)

                period = int(period)
                changes = [data_tup[1] - data_tup[0] for data_tup in zip(data[::1], data[1::1])]

                filtered_gain = [val < 0 for val in changes]
                gains = [0 if filtered_gain[idx] is True else changes[idx] for idx in range(0, len(filtered_gain))]

                filtered_loss = [val > 0 for val in changes]
                losses = [0 if filtered_loss[idx] is True else abs(changes[idx]) for idx in range(0, len(filtered_loss))]

                avg_gain = np.mean(gains[:period])
                avg_loss = np.mean(losses[:period])

                rsi = []
                if avg_loss == 0:
                    rsi.append(100)
                else:
                    rs = avg_gain / avg_loss
                    rsi.append(100 - (100 / (1 + rs)))

                for idx in range(1, len(data) - period):
                    avg_gain = ((avg_gain * (period - 1) +
                                gains[idx + (period - 1)]) / period)
                    avg_loss = ((avg_loss * (period - 1) +
                                losses[idx + (period - 1)]) / period)

                    if avg_loss == 0:
                        rsi.append(100)
                    else:
                        rs = avg_gain / avg_loss
                        rsi.append(100 - (100 / (1 + rs)))

                rsi = fill_for_noncomputable_vals(data, rsi)

                return rsi

            data = list(klines_close_list)
            period = 14

            y = relative_strength_index(data, period)

            for i in range(0, len(y)):
                if np.isnan(y[i]).any():
                    y[i] = 0
            print(y[-1])
            if y[-1] <= float(row["RSI"]) and signal_col[k] == 0 :
                signal_col[k] = 1
                BOT_TOKEN=''
                CHAT_ID='-213082329'
                tbot = bot.Bot(BOT_TOKEN)
                #signal_col.append(1)
                #tbot.send_message(chat_id=CHAT_ID, text=' '+ row["symbol"] +' '+ row["intervals"].split('_')[-1]+ ' RSI is under ' + str(row["RSI"]) +' ')
                print(' '+ row["symbol"] +' '+ eval(row["intervals"]).split('-')[0]+ ' RSI is under ' + str(row["RSI"]) +' ')
                #print(row["intervals"].split("_")[-1])
            elif y[-1] > float(row["RSI"]):
                signal_col[k] == 0
            k += 1
            print(k)
            #print(signal_col[0:10])
time.sleep(300)
"""plt.plot(relative_strength_index(data, period))
   plt.ylabel('RSI')
   plt.show()"""
