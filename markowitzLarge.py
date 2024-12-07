import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Scrape S&P 500 tickers from Wikipedia
scraperURL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
scrapedData = pd.read_html(scraperURL)

# Set the start and end date for the 5-year analysis
startDate = '2019-12-01'
endDate = pd.to_datetime(startDate) + pd.DateOffset(365*5)

# Filter tickers based on "Date added" prior to startDate
sp500 = scrapedData[0]
sp500['Date added'] = pd.to_datetime(sp500['Date added'], errors='coerce')
filtered_sp500 = sp500[sp500['Date added'] < startDate].iloc[:, [0,5]]

tickers = filtered_sp500['Symbol'].tolist()
tickers = [ticker.replace('.', '-') for ticker in tickers]  # Adjust for Yahoo Finance format

# Download adjusted close price data from Yahoo Finance
stockData = yf.download(tickers=tickers, start=startDate, end=endDate, group_by='ticker')

# Check if 'Adj Close' column exists and extract it
try:
    data = stockData['Adj Close']
except KeyError:
    raise KeyError("'Adj Close' column not found. Verify data structure or check for API changes.")

# Calculate log returns
logReturns = np.log(data / data.shift(1))
correlationMatrix = logReturns.corr()

# Function to select least group-correlated tickers
def selectLeastGroupCorrelatedTickers(correlationMatrix, top_n):
    correlationMatrix = correlationMatrix.copy()
    np.fill_diagonal(correlationMatrix.values, np.nan)  # Ignore self-correlation

    avg_correlation = correlationMatrix.mean(axis=0)
    selected_tickers = [avg_correlation.idxmin()]
    remaining_tickers = list(set(correlationMatrix.columns) - set(selected_tickers))

    while len(selected_tickers) < top_n and remaining_tickers:
        group_correlation = correlationMatrix.loc[selected_tickers, remaining_tickers].mean(axis=0)
        next_ticker = group_correlation.idxmin()
        selected_tickers.append(next_ticker)
        remaining_tickers = list(set(remaining_tickers) - {next_ticker})

    return selected_tickers

# Select top least-correlated tickers
tickerAmount = 3
top_least_group_correlated_tickers = selectLeastGroupCorrelatedTickers(correlationMatrix, tickerAmount)

# Filter the data for selected tickers
filtered_data = data[top_least_group_correlated_tickers]
tickers = filtered_data.columns
logReturns = np.log(filtered_data / filtered_data.shift(1))
expectedReturns = logReturns.mean() * 252 * 5
covarianceMatrix = logReturns.cov() * 252 * 5

# Define function to generate random weights
def generateRandomWeights(n):
    weights = np.random.rand(n)
    weights /= np.sum(weights)
    return weights

# Set the risk-free rate for the 5-year horizon
riskFreeRate = 0.0619

# Function to calculate portfolio statistics
def getPortfolioStatistics(randomAssetWeights):
    randomAssetWeights = np.array(randomAssetWeights)
    portfolioReturn = np.sum(expectedReturns * randomAssetWeights)
    portfolioRisk = np.sqrt(np.dot(randomAssetWeights.T, np.dot(covarianceMatrix, randomAssetWeights)))
    sharpeRatio = (portfolioReturn - riskFreeRate) / portfolioRisk

    return {
        'return': portfolioReturn,
        'risk': portfolioRisk,
        'sharpe': sharpeRatio
    }

# Monte Carlo simulation to generate portfolios
def simulatePortfolios(numberOfRandomPortfolios):
    portfolioRisks = []
    portfolioReturns = []
    for _ in range(numberOfRandomPortfolios):
        randomAssetWeights = generateRandomWeights(len(tickers))
        portfolioStatistics = getPortfolioStatistics(randomAssetWeights)
        portfolioRisks.append(portfolioStatistics['risk'])
        portfolioReturns.append(portfolioStatistics['return'])
    return np.array(portfolioRisks), np.array(portfolioReturns)

# Run simulation
simulatedReturns, simulatedRisks = simulatePortfolios(10000)

# Calculate Sharpe ratios
sharpeRatio = simulatedReturns / simulatedRisks
max_sr_ret = simulatedReturns[sharpeRatio.argmax()]
max_sr_vol = simulatedRisks[sharpeRatio.argmax()]

# Plot the efficient frontier
plt.figure(figsize=(18, 10))
plt.scatter(simulatedRisks, simulatedReturns, c=sharpeRatio, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50, label='Max Sharpe Ratio')
plt.legend()
plt.title('Efficient Frontier')
plt.show()
