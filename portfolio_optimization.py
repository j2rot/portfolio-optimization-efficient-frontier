# Portfolio Optimization and Efficient Frontier for a 5-ETF Portfolio
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. Portfolio Setup
tickers = ['SPY', 'QQQ', 'IWM', 'DIA', 'EFA']
data = yf.download(tickers, start='2024-01-01', end='2025-01-01', auto_adjust=True)['Close']
returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

# 2. Define portfolio performance functions
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    port_return = np.sum(weights * mean_returns) * 252
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (port_return - risk_free_rate) / port_std
    return port_return, port_std, sharpe_ratio

# 3. Objective function (negative Sharpe ratio for minimization)
def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

# 4. Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(tickers)))
initial_guess = [1/len(tickers)]*len(tickers)

# 5. Optimize
opt_result = minimize(neg_sharpe, initial_guess, args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
opt_weights = opt_result.x

# 6. Display results
print("Optimal Portfolio Weights:")
for t, w in zip(tickers, opt_weights):
    print(f"{t}: {w:.2%}")
ret, vol, sharpe = portfolio_performance(opt_weights, mean_returns, cov_matrix)
print(f"\nExpected Annual Return: {ret:.2%}")
print(f"Annual Volatility: {vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")

# 7. Efficient Frontier
num_portfolios = 5000
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    r, s, sh = portfolio_performance(weights, mean_returns, cov_matrix)
    results[0,i] = r
    results[1,i] = s
    results[2,i] = sh

plt.figure(figsize=(10,6))
plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(vol, ret, marker='*', color='r', s=300, label='Optimal Portfolio')
plt.title('Efficient Frontier')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.legend()
plt.show()