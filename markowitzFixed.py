import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

scraperURL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
scrapedData = pd.read_html(scraperURL)

startDate = '2019-12-01'
endDate = pd.to_datetime(startDate) + pd.DateOffset(365*5)

sp500 = scrapedData[0]
sp500['Date added'] = pd.to_datetime(sp500['Date added'], errors='coerce')
filtered_sp500 = sp500[sp500['Date added'] < startDate].iloc[:, [0,5]]

tickers = filtered_sp500['Symbol'].tolist()

tickers = [ticker.replace('.','-') for ticker in tickers] # Yahoo Finance uses dashes instead of dots

tickers = tickers[:300]


# We get the data from Yahoo Finance for all of the tickers about the adjusted close price

stockData = yf.download(tickers=tickers, start=startDate, end=endDate).stack()
stockData.index.names = ['date', 'ticker']
stockData.columns = stockData.columns.str.lower()

data = stockData['adj close'].unstack(level='ticker')

# Calculate the logarithmic returns of the adjusted close price
logReturns = np.log(data/data.shift(1))
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

# Get the top 100 least group-correlated tickers
top_least_group_correlated_tickers = selectLeastGroupCorrelatedTickers(correlationMatrix, 5)

# Filter the data to include only the selected tickers

filtered_data = data[top_least_group_correlated_tickers]
tickers = filtered_data.columns
data = filtered_data
logReturns = np.log(data/data.shift(1))
expectedReturns = logReturns.mean()
covarianceMatrix = logReturns.cov()

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
    portfolioReturn = np.sum(expectedReturns * randomAssetWeights) * 252 * 5 
    # 5 years for 252 trading days in a year, since we have the daily expected returns.
    # This is a dot product between the vector of weights, and expected returns. This gets expected portfolio return
    # With these given weights.
    portfolioRisk = np.sqrt(np.dot(randomAssetWeights.T, np.dot(covarianceMatrix * 252 * 5, randomAssetWeights)))
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
    for _ in range(numberOfRandomPortfolios):
        randomAssetWeights = generateRandomWeights(len(tickers))
        portfolioStatistics = getPortfolioStatistics(randomAssetWeights)
        portfolioRisks.append(portfolioStatistics['risk'])
        portfolioReturns.append(portfolioStatistics['return'])
    return np.array(portfolioRisks), np.array(portfolioReturns)

# We run the simulation with 1000 random portfolios.

simulatedReturns, simulatedRisks = simulatePortfolios(1000000)

# We plot the simulated portfolios in a scatter plot.

# Get the sharpe ratio for each portfolio.
sharpeRatio = simulatedReturns / simulatedRisks
max_sr_ret = simulatedReturns[sharpeRatio.argmax()] # return corresponding to maximum sharpe ratio
max_sr_vol = simulatedRisks[sharpeRatio.argmax()] # risk corresponding to maximum sharpe ratio

plt.figure(figsize=(18,10))
plt.scatter(simulatedRisks, simulatedReturns, c=sharpeRatio, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.scatter(max_sr_vol, max_sr_ret,c='red', s=50) # red dot
plt.show()