# Task 1: Machine Data Analysis

This folder contains the analysis of 3 years of cyclone machine sensor data to detect shutdowns, operational states, anomalies, and perform short-term forecasting.

## Overview

The analysis explores approximately 370,000 records of cyclone sensor data at 5-minute intervals, focusing on:

1. Data Preparation & Exploratory Analysis
2. Shutdown / Idle Period Detection
3. Machine State Segmentation (Clustering)
4. Contextual Anomaly Detection + Root Cause Analysis
5. Short-Horizon Forecasting
6. Insights & Storytelling

## Running the Code

### Prerequisites

The following Python packages are required:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- statsmodels
- prophet
- tensorflow (optional, for advanced models)

You can install these packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels prophet
```

### Execution

To run the full analysis:

```bash
cd /workspaces/exactspace
python Task1/task1_analysis.py
```

## Outputs

The script generates the following outputs:

### CSV Files

- `summary_statistics.csv` - Basic statistics for each sensor variable
- `shutdown_periods.csv` - Detected shutdowns with start, end, and duration
- `clusters_summary.csv` - Summary statistics for each operational state
- `forecasts.csv` - True vs predicted values for forecasting tasks
- `anomalous_periods.csv` - Detected anomalies with metadata

### Plots

All visualizations are saved in the `plots/` directory:

- `correlation_matrix.png` - Correlation matrix of sensor variables
- `one_week_*.png` - One week sample of each variable
- `one_year_*.png` - One year sample of each variable (daily average)
- `one_year_shutdowns.png` - One year of data with shutdowns highlighted
- `cluster_pca.png` - PCA visualization of operational states
- `cluster_timeseries.png` - Time series with cluster assignments
- `anomaly_*.png` - Visualizations of selected anomalies
- `forecasting_results.png` - Comparison of forecasting models

### Text Files

- `insights.txt` - Key insights and recommendations based on the analysis
- `anomaly_*_analysis.txt` - Root cause analysis for selected anomalies

## Methodology

### Data Preparation

- Conversion to proper data types
- Handling missing values using forward/backward fill
- Ensuring strict 5-minute intervals
- Outlier treatment using IQR method

### Shutdown Detection

- Used threshold-based approach on Cyclone_Inlet_Draft
- Identified contiguous periods of shutdown
- Calculated total downtime and frequency statistics

### Machine State Segmentation

- Applied KMeans clustering to identify operational states
- Also tried DBSCAN for comparison
- Used PCA for visualization of clusters
- Generated statistics for each cluster

### Anomaly Detection

- Used context-aware Isolation Forest for each operational state
- Identified variables most implicated in each anomaly
- Performed root cause analysis on selected anomalies

### Forecasting

- Predicted Cyclone_Inlet_Gas_Temp for 1 hour ahead (12 steps)
- Compared persistence model, ARIMA, and Prophet
- Evaluated using RMSE and MAE metrics

## Results Summary

The analysis revealed distinct operational states of the cyclone machine, patterns in shutdown events, and anomalies that may serve as early warning indicators. The forecasting models showed variable performance across different operational states, suggesting potential for state-aware predictive models.

Key findings and recommendations are detailed in the `insights.txt` file.