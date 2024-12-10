import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize


scraperURL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
scrapedData = pd.read_html(scraperURL)

startDate = '2023-09-01'
endDate = '2023-12-01'

sp500 = scrapedData[0]
sp500['Date added'] = pd.to_datetime(sp500['Date added'], errors='coerce')
filtered_sp500 = sp500[sp500['Date added'] < startDate].iloc[:, [0,5]]

tickers = filtered_sp500['Symbol'].tolist()

tickers = [ticker.replace('.','-') for ticker in tickers] # Yahoo Finance uses dashes instead of dots

# We get the data from Yahoo Finance for all of the tickers about the adjusted close price

stockData = yf.download(tickers=tickers, start=startDate, end=endDate).stack(future_stack=True)
stockData.index.names = ['date', 'ticker']
stockData.columns = stockData.columns.str.lower()

data = stockData['adj close'].unstack(level='ticker')


# Calculate the logarithmic returns of the adjusted close price
logReturns = np.log(data/data.shift(1)).dropna()
correlationMatrix = logReturns.corr()

def selectLeastGroupCorrelatedTickers(correlationMatrix, top_n):
    # Create a copy of the correlation matrix and replace the diagonal with NaN to ignore self-correlation
    correlationMatrix = correlationMatrix.copy()
    np.fill_diagonal(correlationMatrix.values, np.nan)
    
    # Compute average correlation for each ticker
    avg_correlation = correlationMatrix.mean(axis=0)
    
    # Start with the ticker having the lowest average correlation
    selected_tickers = [avg_correlation.idxmin()]
    remaining_tickers = list(set(correlationMatrix.columns) - set(selected_tickers))
    
    while len(selected_tickers) < top_n and remaining_tickers:
        # Calculate average correlation of each remaining ticker with the selected tickers
        group_correlation = correlationMatrix.loc[selected_tickers, remaining_tickers].mean(axis=0)
        
        # Find the ticker with the lowest average group correlation
        next_ticker = group_correlation.idxmin()
        selected_tickers.append(next_ticker)
        remaining_tickers = list(set(remaining_tickers) - {next_ticker})
    
    return selected_tickers

# Get the top "tickerAmount" least group-correlated tickers
tickerAmount = 6
top_least_group_correlated_tickers = selectLeastGroupCorrelatedTickers(correlationMatrix, tickerAmount)

# Filter the data to include only the selected tickers

filtered_data = data[top_least_group_correlated_tickers]
tickers = filtered_data.columns
logReturns = np.log(filtered_data/filtered_data.shift(1)).dropna()
expectedReturns = logReturns.mean() * 252
covarianceMatrix = logReturns.cov() * 252

# where E[Ai] is the expected return of the asset i found in the expectedReturns array, and E[Aj] is the expected return of the asset j found in the expectedReturns array.

# We define a function to generate a 1xn tuple of random weights that sum to 1

def generateRandomWeights(n):
    weights = np.random.rand(n)
    weights /= np.sum(weights)
    return weights

# Our start date was Dec 1 2019. At this date, the risk free return rate
# was around 1.5%, but has increased to 4% at the end of the 5 year period.
# To avoid obvious bias from too high or too low of a rate, we averaged it 
# to 2.75% for the 5 year period. This is not the most scientific way, but 
# it is a good approximation.

riskFreeRate = 0.0275

# Create a function to return a dictionary of metrics for a portfolio (return, risk, and sharpe ratio).

def getPortfolioStatistics(randomAssetWeights):
    randomAssetWeights = np.array(randomAssetWeights) # convert the tuple to an array

    # this is basically the dot product of the expected returns and the random weights, annualized. the summation of the expected returns of the assets weighted by the random weights gives the expected return of the portfolio.
    portfolioReturn = np.sum(expectedReturns * randomAssetWeights)
    # 5 years for 252 trading days in a year, since we have the daily expected returns.
    # This is a dot product between the vector of weights, and expected returns. This gets expected portfolio return
    # With these given weights.
    portfolioRisk = np.sqrt(np.dot(randomAssetWeights.T, np.dot(covarianceMatrix, randomAssetWeights)))
    sharpeRatio = (portfolioReturn - riskFreeRate) / portfolioRisk # We calculate the Sharpe ratio
    
    # Return a dictionary of the portfolio metrics
    return {
        'return': portfolioReturn,
        'risk': portfolioRisk,
        'sharpe': sharpeRatio
    }
    
# Run a Monte Carlo simulation to generate n random portfolios and store the risk and return values in a list.

def simulatePortfolios(numberOfRandomPortfolios):
    portfolioRisks = []
    portfolioReturns = []
    portfolioSharpes = []
    weights = []
    for _ in range(numberOfRandomPortfolios):
        randomAssetWeights = generateRandomWeights(len(tickers))
        portfolioStatistics = getPortfolioStatistics(randomAssetWeights)
        portfolioRisks.append(portfolioStatistics['risk'])
        portfolioReturns.append(portfolioStatistics['return'])
        portfolioSharpes.append(portfolioStatistics['sharpe'])
        weights.append(randomAssetWeights)
    return {
        'risks': np.array(portfolioRisks),
        'returns': np.array(portfolioReturns),
        'sharpes': np.array(portfolioSharpes),
        'weights': np.array(weights)
    }

simulatedPortfolios = simulatePortfolios(100000)
sharpeRatios = simulatedPortfolios['sharpes']
simulatedReturns = simulatedPortfolios['returns']
simulatedRisks = simulatedPortfolios['risks']

# Get the sharpe ratio for each portfolio.

portfolioIndex = sharpeRatios.argmax()
max_sr_ret = simulatedReturns[portfolioIndex] # return corresponding to maximum sharpe ratio
max_sr_vol = simulatedRisks[portfolioIndex] # risk corresponding to maximum sharpe ratio
max_sr_weights = simulatedPortfolios['weights'][portfolioIndex]

# map the array to a dictionary with the tickers as keys and the weights as values but convert from np.float to regular float

max_sr_weights_dict = {ticker: float(weight) for ticker, weight in zip(tickers, max_sr_weights)}

# We need to get new adjusted close prices to test our weights on. We will use from january 4th 2021 to january 2nd 2024.

# We need to get new adjusted close prices to test our weights on. We will use from january 4th 2021 to january 2nd 2024.
tickersNew = list(max_sr_weights_dict.keys())
weights = list(max_sr_weights_dict.values())

newStartDate = '2023-09-01'
newEndDate = '2024-01-02'

# Download historical data for the tickers
data = yf.download(tickers=tickersNew, start=newStartDate, end=newEndDate)['Adj Close']

# Calculate daily returns for each stock
daily_returns = data.pct_change().dropna()

# Calculate portfolio daily returns
portfolio_returns = (daily_returns * weights).sum(axis=1)

# Calculate cumulative returns
cumulative_returns = (1 + portfolio_returns).cumprod()
equal_weights = [1 / len(tickersNew)] * len(tickersNew)
equal_portfolio_returns = (daily_returns * equal_weights).sum(axis=1)
equal_cumulative_returns = (1 + equal_portfolio_returns).cumprod()

# Plot both cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns, label="Proportionally Weighted Portfolio")
plt.plot(equal_cumulative_returns, label="Equally Weighted Portfolio", linestyle='--')
plt.title("Cumulative Returns: Proportional vs. Equal Weights (2021-2024)")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(18,10))
plt.scatter(simulatedRisks, simulatedReturns, c=sharpeRatios, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.scatter(max_sr_vol, max_sr_ret,c='red', s=50) # red dot
plt.show()