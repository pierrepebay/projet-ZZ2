import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def compute_rsi_list(price_list, period):
    size = len(price_list)
    up_move = [np.nan] * size
    down_move = [np.nan] * size
    avg_up = [np.nan] * size
    avg_down = [np.nan] * size
    rs = [np.nan] * size
    rsi = [np.nan] * size

    # Calculate Up Move & Down Move
    prev_price = price_list[0]
    for i in range(1, size):
        up_move[i]= 0
        down_move[i] = 0
        
        if price_list[i] > prev_price:
            up_move[i] = price_list[i] - prev_price
            
        if price_list[i] < prev_price:
            down_move[i] = abs(price_list[i] - prev_price)

        prev_price = price_list[i]


    # Calculate initial Average Up & Down, RS and RSI
    avg_up[period] = sum(up_move[1:period + 1]) / len(up_move[1:period + 1])
    avg_down[period] = sum(down_move[1:period + 1]) / len(down_move[1:period + 1])
    rs[period] = avg_up[period] / avg_down[period]
    rsi[period] = 100 - (100 / (1 + rs[period]))

    # Calculate rest of Average Up, Average Down, RS, RSI
    prev_avg_up = avg_up[period]
    prev_avg_down = avg_down[period]
    for i in range(period + 1, size):
        avg_up[i] = (prev_avg_up * (period - 1) + up_move[i]) / period
        avg_down[i] = (prev_avg_down * (period - 1) + down_move[i]) / period
        prev_avg_down = avg_down[i]
        prev_avg_up = avg_up[i]

        rs[i] = avg_up[i] / avg_down[i]

        rsi[i] = 100 - (100 / (1 + rs[i]))
    
    return rsi

def compute_value(price_list, rsi):
    size = len(price_list)
    buy_amount = 1
    sell_amount = 1
    money = 10000
    n_shares  = money / price_list[0]
    money_list = []
    prev_buy = False
    prev_sell = False
    buy_x = []
    buy_y = []
    sell_x = []
    sell_y = []

    for i in range(1,size):
        if(rsi[i] < 30):
            prev_buy = True
            prev_sell = False
            money -= price_list[i] * buy_amount
            buy_x.append(price_list[i])
            buy_y.append(i)
        elif(rsi[i] > 70):
            prev_sell = True
            prev_buy = False
            n_shares -= 1
            money += price_list[i] * sell_amount
            sell_x.append(price_list[i])
            sell_y.append(i)
        money_list.append(money)
    return money_list #, buy_x, buy_y, sell_x, sell_y

def main():
    stock_name = "TSLA"    

    stock = yf.Ticker(stock_name)

    data=pd.read_csv("AAPL_stock_sample/AAPL_1hour_sample.txt", sep=",", header=None, names=["DateTime", "Open", "High", "Low", "Close", "Volume"])
    close = data['Close'].to_list()

    close = stock.history(period = '1y')['Close'].to_list()
    
    plt.title("{} stock and RSI usage".format(stock_name))
    for period in [5, 6, 7, 8, 10, 14]:
        rsi = compute_rsi_list(close, period)
        plt.subplot(3, 1, 2)
        plt.xlabel("t")
        plt.ylabel("RSI")
        plt.plot(rsi, label = period)
        money_list = compute_value(close, rsi)
        plt.subplot(3, 1, 3)
        plt.xlabel("t")
        plt.ylabel("cash")
        plt.plot(money_list, label = period)
        plt.legend()
    plt.subplot(3, 1, 2)
    plt.axhline(y=30, color='red', linestyle='--')
    plt.axhline(y=70, color='green', linestyle='--')
    plt.legend()
    plt.subplot(3, 1, 1)
    plt.xlabel("t")
    plt.ylabel("stock price")
    plt.plot(close)
    
    plt.show()

    # plt.title("{} stock and RSI usage".format(stock_name))
    # rsi = compute_rsi_list(close, 7)
    # plt.subplot(3, 1, 2)
    # plt.xlabel("t")
    # plt.ylabel("RSI")
    # plt.plot(rsi, label = 7)
    # money_list, buy_x, buy_y, sell_x, sell_y = compute_value(close, rsi)
    # plt.subplot(3, 1, 3)
    # plt.xlabel("t")
    # plt.ylabel("cash")
    # plt.plot(money_list, label = 7)
    # plt.legend()
    # plt.subplot(3, 1, 2)
    # plt.axhline(y=30, color='green', linestyle='--')
    # plt.axhline(y=70, color='red', linestyle='--')
    # plt.legend()
    # plt.subplot(3, 1, 1)
    # plt.scatter(buy_y, buy_x, color="green")
    # plt.scatter(sell_y, sell_x, color="red")
    # plt.xlabel("t")
    # plt.ylabel("stock price")
    # plt.plot(close)
    
    # plt.show()

if __name__=="__main__":
    main()