"""Run Monte Carlo Simulations on any stock of using choosing, implemented along with multithreading for efficiency, and tested against true prices of the last n days"""
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from joblib import Parallel, delayed

"""Steps for MCS:
1. Get data
2. Identify Input Distribution
3. Random Variable Generation
4. Run Simulations on Random data for a large number of times to have the law of large numbers take effect
5. Make analysis from the simulations ran

We will use the Geometric Brownian Motion model to simulate stock price movements in financial markets. For that, we need the mean, std and a number extra noise, taken from the Normal distribution.
"""

def prepare_data(stock_df):
    data = stock_df["Close"]
    returns = data.pct_change()
    
    last_price = data.iloc[-1]
    mean = returns.mean()
    std = returns.std()
    return (last_price, mean, std)

def _gbm_model(prev_price, mean, std):
    return prev_price * np.exp((mean - 0.5 * np.square(std)) + (std * np.random.normal()))

def single_simulation(last_price, mean, std, days):
    prices = [last_price]
    for day in range(days):
        price = _gbm_model(prices[-1], mean, std)
        prices.append(price)
    
    return pd.Series(prices[1:])


def monte_carlo(stock_data, n_days, n_simulations, axs):
    last_price, mean, std = prepare_data(stock_data)

    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(single_simulation)(last_price, mean, std, n_days)
        for _ in range(n_simulations)
    )
    simulated_prices = np.column_stack(results)

    axs[0].plot(simulated_prices)
    axs[0].set_xlabel("Days")
    axs[0].set_ylabel("Stock Prices")
    
    final_prices = simulated_prices[-1:]
    most_probable_sim_idx = np.argmin(np.abs(final_prices - np.median(final_prices)))
    most_probable_sim = simulated_prices[:, most_probable_sim_idx]
    
    axs[1].plot(most_probable_sim, label="Most probable simulation")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="MCS_Stock_Portfolio",
        description="Run Monte Carlo Simulations on a stock of your choosing"
    )
    parser.add_argument("-t", "--ticker", type=str, default="AAPL", help="The ticker of the stock that you would like to simulate. Default: AAPL (Apple)")
    parser.add_argument("-n", "--n-sims", type=int, default=10000, help="Number of simulations to run. Default: 10000")
    parser.add_argument("-d", "--days", type=int, default=30, help="Number of days for which you want to simulate the stock movement. Default: 30")
    # parser.print_help()
    args = parser.parse_args()
    
    print("Downloading stock data and saving as dataframe")
    stock = yf.download(args.ticker, period="5y", multi_level_index=False)
    
    _, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=False, sharex=False)
    ax = ax.flatten()
    ax[0].set_title("All simulations")
    
    ax[1].set_xlabel("Days")
    ax[1].set_ylabel("Stock Prices")
    ax[1].set_title("Most probable simulation vs True prices")   
    
    last_30_days = stock.iloc[-args.days:, 0].reset_index(drop=True)
    ax[1].plot(last_30_days, label="True prices")
    
    monte_carlo(stock.iloc[:-args.days], n_days=args.days, n_simulations=args.n_sims, axs=ax)

    plt.legend()    
    plt.show()