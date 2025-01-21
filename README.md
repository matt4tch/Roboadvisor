# RoboAdvisor: High-Risk Portfolio Optimizer

This RoboAdvisor application constructs a portfolio of 12 high-risk stocks by maximizing portfolio risk (standard deviation) and correlation. It emphasizes concentrated, undiversified investments by heavily weighting the riskiest stocks.

---

## 📊 Key Features
- **Stock Filtering**: Filters stocks by:
    - Trading volume (≥ 200,000)
    - Currency (USD)
    - Minimum trading days (20 days/month)
  
- **Portfolio Construction**:
    - Selects 12 stocks with the highest correlation.
    - Prioritizes riskier stocks by assigning higher weights.
  
- **Custom Correlation**: Implements a tailored Pearson’s correlation formula for multivariable evaluation.

- **Risk Optimization**:
    - Maximizes portfolio standard deviation for higher risk exposure.
    - Assigns weights between 4.17% and 25% for each stock.
  
- **Visualization**: Graphs portfolio performance and compares equal-weighted vs optimized portfolios.

---

## 🛠 Methodology

### 1. Stock Filtering
- Input a CSV file of tickers (`Tickers-Copy1.csv`).
- Ensure valid stocks meet these criteria:
    - Traded in USD.
    - Monthly average volume ≥ 200,000.
    - Minimum 20 trading days per month.

### 2. Portfolio Construction
- Start with the two most correlated stocks.
- Iteratively add the next most correlated stock until the portfolio contains 12 stocks.

### 3. Weight Optimization
- Heavily weight stocks with higher standard deviations.
- Maintain weights between 4.17% and 25%.

### 4. Risk Analysis
- Calculate portfolio standard deviation and correlation.
- Display portfolio performance over time.

---

## 📈 Usage
1. **Prepare Your Input**:
    - Place the CSV file of tickers in the project directory (`Tickers-Copy1.csv`).
2. **Run the Script**:
    - Execute the Python script to filter stocks, build the portfolio, and calculate optimal weights.
3. **Outputs**:
    - Final portfolio of 12 stocks with weights and expected values.
    - Graph of portfolio performance over time.

---

## ⚡ Visualization Example

The script generates a graph comparing equal-weighted and optimized portfolios, emphasizing the differences in performance and risk.

![Portfolio Graph Example](example-portfolio-graph.png)

---

## 🎯 Motivation
Standard deviation is a key measure of risk, reflecting the volatility of stock prices. This RoboAdvisor focuses on maximizing standard deviation and correlation, creating a high-risk, high-reward portfolio that benefits from repeated price patterns.

---