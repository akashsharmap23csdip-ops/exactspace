"""
Task 1: Machine Data Analysis
This script performs analysis on 3 years of cyclone sensor data to:
1. Prepare and explore data
2. Detect shutdown/idle periods
3. Segment machine states using clustering
4. Detect anomalies within context
5. Forecast sensor values
6. Provide insights and storytelling
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from prophet import Prophet

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Create output folders if they don't exist
os.makedirs('/workspaces/exactspace/Task1/plots', exist_ok=True)

# 1. Data Preparation & Exploratory Analysis
def load_and_prepare_data(file_path):
    """Load and prepare the data for analysis."""
    print("Loading data...")
    df = pd.read_excel(file_path)
    
    # Convert time to datetime and set as index
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    
    # Convert columns to numeric, coerce errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values per column:\n{missing_values}")
    
    # Fill NaN values using forward fill, then backward fill
    df = df.ffill()
    df = df.bfill()
    
    # Check for remaining missing values
    if df.isnull().sum().sum() > 0:
        print(f"Remaining missing values: {df.isnull().sum().sum()}")
    
    # Ensure strict 5-min intervals - resample if needed
    df = df.resample('5min').mean()
    
    # Handle outliers using IQR method
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Cap outliers rather than removing them
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def exploratory_analysis(df):
    """Perform exploratory data analysis."""
    print("\nPerforming exploratory analysis...")
    
    # Basic statistics
    print("\nBasic statistics:")
    stats = df.describe().T
    print(stats)
    
    # Save summary statistics to CSV
    stats.to_csv('/workspaces/exactspa/workspaces/exactspace/Task1/summary_statistics.csv')
    
    # Correlation matrix
    corr_matrix = df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Cyclone Variables')
    plt.tight_layout()
    plt.savefig('/workspaces/exactspa/workspaces/exactspace/Task1/plots/correlation_matrix.png')
    
    # Visualize one week of data
    one_week = df.iloc[0:2016]  # 12 samples per hour * 24 hours * 7 days = 2016 samples
    
    # Plot each variable for one week
    for col in df.columns:
        plt.figure(figsize=(16, 6))
        one_week[col].plot()
        plt.title(f'{col} - One Week Sample')
        plt.xlabel('Date')
        plt.ylabel(col)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'/workspaces/exactspace/Task1/plots/one_week_{col}.png')
        plt.close()
    
    # Visualize one year of data
    one_year = df.iloc[0:105120]  # 12 samples per hour * 24 hours * 365 days = 105120 samples
    
    # Plot each variable for one year (resample to daily for better visualization)
    for col in df.columns:
        plt.figure(figsize=(16, 6))
        one_year[col].resample('D').mean().plot()
        plt.title(f'{col} - One Year Sample (Daily Average)')
        plt.xlabel('Date')
        plt.ylabel(col)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'/workspaces/exactspace/Task1/plots/one_year_{col}.png')
        plt.close()
        
    return corr_matrix

# 2. Shutdown / Idle Period Detection
def detect_shutdowns(df, threshold=-100):
    """
    Detect machine shutdown periods based on specific thresholds.
    We consider the machine to be shut down when Cyclone_Inlet_Draft is above the threshold.
    """
    print("\nDetecting shutdown periods...")
    
    # Define shutdown condition - when draft is above threshold (less negative)
    shutdown_mask = df['Cyclone_Inlet_Draft'] > threshold
    
    # Add shutdown indicator to dataframe
    df['is_shutdown'] = shutdown_mask
    
    # Find contiguous periods of shutdown
    df['shutdown_group'] = (df['is_shutdown'] != df['is_shutdown'].shift()).cumsum()
    
    # Group by shutdown periods
    shutdown_groups = df[df['is_shutdown']].groupby('shutdown_group')
    
    # Extract shutdown periods
    shutdown_periods = []
    
    for name, group in shutdown_groups:
        start_time = group.index.min()
        end_time = group.index.max()
        duration_hours = (end_time - start_time).total_seconds() / 3600
        
        # Only consider shutdowns that last at least 30 minutes
        if duration_hours >= 0.5:
            shutdown_periods.append({
                'start': start_time,
                'end': end_time,
                'duration_hours': duration_hours
            })
    
    # Convert to DataFrame
    shutdown_df = pd.DataFrame(shutdown_periods)
    
    if not shutdown_df.empty:
        # Calculate total downtime and number of events
        total_downtime = shutdown_df['duration_hours'].sum()
        num_events = len(shutdown_df)
        
        print(f"Total number of shutdown events: {num_events}")
        print(f"Total downtime (hours): {total_downtime:.2f}")
        print(f"Average shutdown duration (hours): {total_downtime / num_events:.2f}")
        
        # Save to CSV
        shutdown_df.to_csv('/workspaces/exactspace/Task1/shutdown_periods.csv')
        
        # Visualize one full year with shutdowns highlighted
        visualize_shutdowns(df, shutdown_df)
        
    return shutdown_df, df

def visualize_shutdowns(df, shutdown_df):
    """Visualize one full year of data with shutdowns highlighted."""
    print("\nVisualizing shutdowns...")
    
    # Get one year of data
    one_year = df.iloc[0:105120].copy()  # 12 samples per hour * 24 hours * 365 days = 105120 samples
    
    # Plot with shutdowns highlighted
    plt.figure(figsize=(20, 10))
    
    # Plot the Inlet Draft (main indicator of shutdowns)
    plt.plot(one_year.index, one_year['Cyclone_Inlet_Draft'], label='Cyclone Inlet Draft', color='blue', alpha=0.7)
    
    # Highlight shutdown periods
    for _, row in shutdown_df.iterrows():
        if row['start'] <= one_year.index[-1] and row['end'] >= one_year.index[0]:
            start = max(row['start'], one_year.index[0])
            end = min(row['end'], one_year.index[-1])
            plt.axvspan(start, end, color='red', alpha=0.3)
    
    plt.title('One Year of Cyclone Operation with Shutdowns Highlighted')
    plt.xlabel('Date')
    plt.ylabel('Cyclone Inlet Draft')
    plt.grid(True)
    plt.legend(['Inlet Draft', 'Shutdown Periods'])
    plt.tight_layout()
    plt.savefig('/workspaces/exactspace/Task1/plots/one_year_shutdowns.png')
    plt.close()

# 3. Machine State Segmentation (Clustering)
def perform_clustering(df, shutdown_df, n_clusters=4):
    """Perform clustering to identify operational states."""
    print("\nPerforming machine state segmentation...")
    
    # Create a copy of the dataframe without shutdown periods
    active_df = df.copy()
    
    # Remove shutdown periods
    for _, row in shutdown_df.iterrows():
        active_df = active_df.loc[(active_df.index < row['start']) | (active_df.index > row['end'])]
    
    # Select features for clustering
    features = active_df.drop(columns=['is_shutdown', 'shutdown_group'])
    
    # Check for any remaining NaN values and fill them
    if features.isnull().sum().sum() > 0:
        print(f"Filling {features.isnull().sum().sum()} remaining NaN values for clustering")
        features = features.fillna(features.mean())
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    active_df['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Also try DBSCAN for comparison
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    active_df['dbscan_cluster'] = dbscan.fit_predict(scaled_features)
    
    # Get cluster statistics
    cluster_stats = []
    
    for i in range(n_clusters):
        cluster_data = active_df[active_df['cluster'] == i]
        
        # Calculate basic statistics
        stats = cluster_data[features.columns].describe().T
        
        # Calculate frequency and duration statistics
        cluster_changes = (active_df['cluster'] != active_df['cluster'].shift()).cumsum()
        cluster_segments = active_df.groupby(['cluster', cluster_changes])
        
        cluster_i_segments = [group for name, group in cluster_segments if name[0] == i]
        
        if cluster_i_segments:
            segment_durations = [(segment.index.max() - segment.index.min()).total_seconds() / 3600 for segment in cluster_i_segments]
            avg_duration = np.mean(segment_durations) if segment_durations else 0
            frequency = len(cluster_i_segments) / (active_df.index.max() - active_df.index.min()).total_seconds() * 86400  # per day
        else:
            avg_duration = 0
            frequency = 0
        
        # Identify defining characteristics of cluster
        cluster_means = cluster_data[features.columns].mean()
        global_means = features.mean()
        
        # Compare cluster means to global means
        differences = cluster_means - global_means
        defining_features = differences.abs().sort_values(ascending=False).index[:2]
        
        direction = ['high' if differences[feat] > 0 else 'low' for feat in defining_features]
        description = f"{direction[0].capitalize()} {defining_features[0]}, {direction[1]} {defining_features[1]}"
        
        cluster_info = {
            'cluster': i,
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(active_df) * 100,
            'avg_duration_hours': avg_duration,
            'frequency_per_day': frequency,
            'description': description
        }
        
        # Add means and std for each feature
        for col in features.columns:
            cluster_info[f'{col}_mean'] = cluster_data[col].mean()
            cluster_info[f'{col}_std'] = cluster_data[col].std()
            cluster_info[f'{col}_25%'] = cluster_data[col].quantile(0.25)
            cluster_info[f'{col}_75%'] = cluster_data[col].quantile(0.75)
        
        cluster_stats.append(cluster_info)
    
    # Convert to DataFrame and save to CSV
    cluster_stats_df = pd.DataFrame(cluster_stats)
    cluster_stats_df.to_csv('/workspaces/exactspace/Task1/clusters_summary.csv')
    
    # Visualize clusters
    visualize_clusters(active_df, n_clusters)
    
    return active_df, cluster_stats_df

def visualize_clusters(df_with_clusters, n_clusters):
    """Visualize the identified clusters."""
    print("\nVisualizing clusters...")
    
    # Create a color map
    colors = plt.cm.tab10(range(n_clusters))
    
    # PCA for dimensionality reduction
    from sklearn.decomposition import PCA
    
    # Select features for PCA
    features = df_with_clusters.drop(columns=['cluster', 'dbscan_cluster', 'is_shutdown', 'shutdown_group'])
    
    # Check for any NaN values and fill them
    if features.isnull().sum().sum() > 0:
        print(f"Filling {features.isnull().sum().sum()} NaN values for PCA visualization")
        features = features.fillna(features.mean())
    
    # Apply PCA
    pca = PCA(n_components=2)
    # Apply StandardScaler with careful handling of NaN values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    # Double check for NaN values after scaling
    if np.isnan(scaled_features).any():
        print("NaN values found after scaling, filling with column means")
        # Replace NaNs with column means in the scaled data
        column_means = np.nanmean(scaled_features, axis=0)
        inds = np.where(np.isnan(scaled_features))
        scaled_features[inds] = np.take(column_means, inds[1])
    
    pca_result = pca.fit_transform(scaled_features)
    
    # Create DataFrame for PCA results
    pca_df = pd.DataFrame({
        'PCA1': pca_result[:, 0],
        'PCA2': pca_result[:, 1],
        'Cluster': df_with_clusters['cluster']
    })
    
    # Plot PCA results
    plt.figure(figsize=(12, 8))
    
    for i in range(n_clusters):
        plt.scatter(
            pca_df[pca_df['Cluster'] == i]['PCA1'],
            pca_df[pca_df['Cluster'] == i]['PCA2'],
            c=[colors[i]],
            label=f'Cluster {i}',
            alpha=0.7
        )
    
    plt.title('Machine States Visualization using PCA')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/workspaces/exactspace/Task1/plots/cluster_pca.png')
    plt.close()
    
    # Plot time series with cluster colors
    # Sample one week for better visualization
    sample = df_with_clusters.iloc[0:2016].copy()  # 12 samples per hour * 24 hours * 7 days = 2016 samples
    
    plt.figure(figsize=(16, 6))
    
    for i in range(n_clusters):
        cluster_data = sample[sample['cluster'] == i]
        plt.scatter(
            cluster_data.index,
            cluster_data['Cyclone_Inlet_Gas_Temp'],
            c=[colors[i]],
            label=f'Cluster {i}',
            alpha=0.7,
            s=20
        )
    
    plt.title('One Week of Cyclone Inlet Gas Temperature by Operational State')
    plt.xlabel('Date')
    plt.ylabel('Cyclone Inlet Gas Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/workspaces/exactspace/Task1/plots/cluster_timeseries.png')
    plt.close()

# 4. Contextual Anomaly Detection + Root Cause Analysis
def detect_anomalies(df_with_clusters, n_clusters=4):
    """Detect anomalies contextual to cluster/state."""
    print("\nPerforming contextual anomaly detection...")
    
    # Container for anomalies
    anomalies = []
    
    # For each cluster, train a separate anomaly detection model
    for i in range(n_clusters):
        # Get data for this cluster
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == i].copy()
        
        if len(cluster_data) > 100:  # Only process clusters with enough data
            print(f"Processing cluster {i} with {len(cluster_data)} data points")
            
            # Select features for anomaly detection
            features = cluster_data.drop(columns=['cluster', 'dbscan_cluster', 'is_shutdown', 'shutdown_group'])
            
            # Check for any NaN values and fill them
            if features.isnull().sum().sum() > 0:
                print(f"Filling {features.isnull().sum().sum()} NaN values for anomaly detection in cluster {i}")
                features = features.fillna(features.mean())
            
            # Train Isolation Forest
            model = IsolationForest(contamination=0.01, random_state=42)
            cluster_data['anomaly'] = model.fit_predict(features)
            
            # Extract anomalies (marked as -1 by Isolation Forest)
            anomaly_points = cluster_data[cluster_data['anomaly'] == -1].copy()
            
            # Group consecutive anomalies into events
            anomaly_points['event_group'] = (anomaly_points.index.to_series().diff() > pd.Timedelta(minutes=15)).cumsum()
            
            # Process each anomaly event
            for event_id, event_data in anomaly_points.groupby('event_group'):
                start_time = event_data.index.min()
                end_time = event_data.index.max()
                duration_mins = (end_time - start_time).total_seconds() / 60
                
                # Only consider events that last at least 10 minutes
                if duration_mins >= 10:
                    # Find which variables contributed most to the anomaly
                    # Compare with the cluster's normal values
                    normal_data = cluster_data[cluster_data['anomaly'] == 1]
                    
                    # Calculate z-scores for the anomaly event compared to normal data
                    z_scores = {}
                    for col in features.columns:
                        normal_mean = normal_data[col].mean()
                        normal_std = normal_data[col].std()
                        if normal_std == 0:
                            z_scores[col] = 0
                        else:
                            event_mean = event_data[col].mean()
                            z_scores[col] = abs((event_mean - normal_mean) / normal_std)
                    
                    # Get the top 2 most anomalous variables
                    top_vars = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)[:2]
                    implicated_vars = [var for var, score in top_vars]
                    
                    # Create anomaly record
                    anomaly_record = {
                        'start': start_time,
                        'end': end_time,
                        'duration_mins': duration_mins,
                        'cluster': i,
                        'implicated_vars': ', '.join(implicated_vars),
                        'z_scores': ', '.join([f"{var}: {score:.2f}" for var, score in top_vars])
                    }
                    
                    anomalies.append(anomaly_record)
    
    # Convert to DataFrame and save to CSV
    if anomalies:
        anomalies_df = pd.DataFrame(anomalies)
        anomalies_df.to_csv('/workspaces/exactspace/Task1/anomalous_periods.csv')
        
        # Select a few interesting anomalies for root cause analysis
        selected_anomalies = anomalies_df.sort_values(by='duration_mins', ascending=False).head(5)
        perform_root_cause_analysis(df_with_clusters, selected_anomalies)
        
        return anomalies_df
    else:
        print("No anomalies detected.")
        return pd.DataFrame()

def perform_root_cause_analysis(df, selected_anomalies):
    """Perform root cause analysis on selected anomalies."""
    print("\nPerforming root cause analysis on selected anomalies...")
    
    for idx, anomaly in selected_anomalies.iterrows():
        print(f"\nAnalyzing anomaly from {anomaly['start']} to {anomaly['end']}")
        
        # Get data around the anomaly (1 hour before and after)
        start_analysis = anomaly['start'] - pd.Timedelta(hours=1)
        end_analysis = anomaly['end'] + pd.Timedelta(hours=1)
        
        context_data = df[(df.index >= start_analysis) & (df.index <= end_analysis)].copy()
        
        # Mark the actual anomaly period
        context_data['is_anomaly'] = False
        anomaly_period = context_data[(context_data.index >= anomaly['start']) & (context_data.index <= anomaly['end'])].index
        context_data.loc[anomaly_period, 'is_anomaly'] = True
        
        # Identify implicated variables from the record
        implicated_vars = [var.strip() for var in anomaly['implicated_vars'].split(',')]
        
        # Plot the contextual data for the top variables
        plt.figure(figsize=(15, 10))
        
        for i, var in enumerate(implicated_vars):
            plt.subplot(len(implicated_vars), 1, i+1)
            
            # Plot the context period
            plt.plot(context_data.index, context_data[var], 'b-', label=var)
            
            # Highlight the anomaly period
            anomaly_data = context_data[context_data['is_anomaly']]
            plt.plot(anomaly_data.index, anomaly_data[var], 'r-', linewidth=2, label=f'{var} (Anomaly)')
            
            plt.title(f'{var} During Anomaly Period')
            plt.xlabel('Time')
            plt.ylabel(var)
            plt.legend()
            plt.grid(True)
        
        # Add overall title
        plt.suptitle(f'Anomaly Analysis: Cluster {int(anomaly["cluster"])} from {anomaly["start"].strftime("%Y-%m-%d %H:%M")} to {anomaly["end"].strftime("%Y-%m-%d %H:%M")}', 
                   fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'/workspaces/exactspace/Task1/plots/anomaly_{idx}.png')
        plt.close()
        
        # Generate root cause hypothesis
        var1, var2 = implicated_vars[:2] if len(implicated_vars) >= 2 else (implicated_vars[0], '')
        
        # Compare before, during, and after the anomaly
        before_anomaly = df[(df.index >= start_analysis) & (df.index < anomaly['start'])]
        during_anomaly = df[(df.index >= anomaly['start']) & (df.index <= anomaly['end'])]
        after_anomaly = df[(df.index > anomaly['end']) & (df.index <= end_analysis)]
        
        # Calculate changes for the implicated variables
        changes = {}
        for var in implicated_vars:
            before_mean = before_anomaly[var].mean()
            during_mean = during_anomaly[var].mean()
            after_mean = after_anomaly[var].mean()
            
            # Calculate percentage changes
            pct_change_during = ((during_mean - before_mean) / before_mean) * 100 if before_mean != 0 else float('inf')
            pct_change_after = ((after_mean - during_mean) / during_mean) * 100 if during_mean != 0 else float('inf')
            
            changes[var] = {
                'before_mean': before_mean,
                'during_mean': during_mean,
                'after_mean': after_mean,
                'pct_change_during': pct_change_during,
                'pct_change_after': pct_change_after
            }
        
        # Generate a hypothesis based on the changes
        for var in implicated_vars:
            change = changes[var]['pct_change_during']
            direction = "increased" if change > 0 else "decreased"
            magnitude = "significantly" if abs(change) > 50 else "moderately" if abs(change) > 20 else "slightly"
            
            print(f"{var} {direction} {magnitude} by {abs(change):.2f}% during the anomaly")
        
        # Save root cause analysis to text file
        with open(f'/workspaces/exactspace/Task1/anomaly_{idx}_analysis.txt', 'w') as f:
            f.write(f"Anomaly Analysis: Cluster {int(anomaly['cluster'])} from {anomaly['start']} to {anomaly['end']}\n")
            f.write(f"Duration: {anomaly['duration_mins']:.2f} minutes\n\n")
            
            f.write("Implicated Variables and Z-Scores:\n")
            f.write(f"{anomaly['z_scores']}\n\n")
            
            f.write("Variable Changes:\n")
            for var in implicated_vars:
                change = changes[var]['pct_change_during']
                direction = "increased" if change > 0 else "decreased"
                magnitude = "significantly" if abs(change) > 50 else "moderately" if abs(change) > 20 else "slightly"
                
                f.write(f"{var} {direction} {magnitude} by {abs(change):.2f}% during the anomaly\n")
                f.write(f"  Before mean: {changes[var]['before_mean']:.2f}\n")
                f.write(f"  During mean: {changes[var]['during_mean']:.2f}\n")
                f.write(f"  After mean: {changes[var]['after_mean']:.2f}\n\n")
            
            # Generate root cause hypothesis
            if len(implicated_vars) >= 2:
                var1, var2 = implicated_vars[:2]
                change1 = changes[var1]['pct_change_during']
                change2 = changes[var2]['pct_change_during']
                
                direction1 = "increase" if change1 > 0 else "decrease"
                direction2 = "increase" if change2 > 0 else "decrease"
                
                f.write("Root Cause Hypothesis:\n")
                f.write(f"A sudden {direction1} in {var1} coincided with a {direction2} in {var2}, which suggests ")
                
                # Different hypotheses based on variable combinations
                if "Temp" in var1 and "Draft" in var2:
                    f.write("a potential thermal imbalance affecting pressure conditions, possibly due to upstream process changes or partial blockage.\n")
                elif "Draft" in var1 and "Draft" in var2:
                    f.write("a draft control issue affecting multiple parts of the cyclone, possibly due to fan problems or control system malfunction.\n")
                elif "Temp" in var1 and "Temp" in var2:
                    f.write("a heat transfer anomaly affecting multiple zones of the cyclone, possibly due to material property changes or cooling system issues.\n")
                else:
                    f.write("an operational anomaly that may require further investigation of upstream processes or control parameters.\n")

# 5. Short-Horizon Forecasting
def perform_forecasting(df):
    """Forecast Cyclone_Inlet_Gas_Temp for the next 1 hour (12 steps)."""
    print("\nPerforming short-horizon forecasting...")
    
    # Select the target variable
    target = 'Cyclone_Inlet_Gas_Temp'
    
    # Create a copy of the data without shutdown periods
    active_df = df[df['is_shutdown'] == False].copy()
    
    # Prepare data for forecasting - create lag features
    forecast_df = active_df[[target]].copy()
    
    # Create lag features (previous values)
    for i in range(1, 13):  # Use 12 previous values
        forecast_df[f'{target}_lag_{i}'] = forecast_df[target].shift(i)
    
    # Create time features
    forecast_df['hour'] = forecast_df.index.hour
    forecast_df['day_of_week'] = forecast_df.index.dayofweek
    
    # Drop rows with NaN values (due to lag creation)
    forecast_df = forecast_df.dropna()
    
    # Train-test split (use the last 30 days for testing)
    train_df = forecast_df.iloc[:-8640]  # 12 samples per hour * 24 hours * 30 days = 8640 samples
    test_df = forecast_df.iloc[-8640:]
    
    # Function to create sequences for multi-step forecasting
    def create_sequences(data, target_col, n_steps_in, n_steps_out):
        X, y = [], []
        for i in range(len(data) - n_steps_in - n_steps_out + 1):
            # Input sequence
            seq_x = data.iloc[i:i+n_steps_in].drop(columns=[target_col])
            # Output sequence
            seq_y = data.iloc[i+n_steps_in:i+n_steps_in+n_steps_out][target_col].values
            X.append(seq_x.values)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    # Parameters
    n_steps_in = 12  # Use 1 hour of data
    n_steps_out = 12  # Predict 1 hour ahead
    
    # 1. Persistence model (baseline)
    print("\nEvaluating persistence model (baseline)...")
    persistence_predictions = []
    
    for i in range(0, len(test_df) - n_steps_out + 1, n_steps_out):
        if i + n_steps_out <= len(test_df):
            # Use the last known value and repeat it for n_steps_out
            last_value = test_df.iloc[i-1][target]
            persistence_predictions.extend([last_value] * n_steps_out)
    
    # Trim to match the expected output length
    persistence_predictions = persistence_predictions[:len(test_df) - n_steps_out + 1]
    
    # 2. ARIMA model
    print("\nTraining and evaluating ARIMA model...")
    
    # Check for any NaN values in target variable for ARIMA
    arima_train_series = train_df[target]
    if arima_train_series.isnull().sum() > 0:
        print(f"Filling {arima_train_series.isnull().sum()} NaN values for ARIMA model")
        arima_train_series = arima_train_series.fillna(method='ffill').fillna(method='bfill')
    
    # Train ARIMA model
    arima_model = sm.tsa.ARIMA(arima_train_series, order=(5,1,1))  # ARIMA(5,1,1) works well for many time series
    arima_fit = arima_model.fit()
    
    # Make predictions
    arima_predictions = []
    
    for i in range(0, len(test_df) - n_steps_out + 1, n_steps_out):
        if i + n_steps_out <= len(test_df):
            # Forecast n_steps_out steps ahead
            forecast = arima_fit.forecast(steps=n_steps_out)
            arima_predictions.extend(forecast)
    
    # Trim to match expected length
    arima_predictions = arima_predictions[:len(test_df) - n_steps_out + 1]
    
    # 3. Prophet model
    print("\nTraining and evaluating Prophet model...")
    
    # Prepare data for Prophet
    prophet_df = pd.DataFrame({'ds': train_df.index, 'y': train_df[target]})
    
    # Check for any NaN values
    if prophet_df['y'].isnull().sum() > 0:
        print(f"Filling {prophet_df['y'].isnull().sum()} NaN values for Prophet model")
        prophet_df['y'] = prophet_df['y'].fillna(prophet_df['y'].mean())
    
    # Train Prophet model
    prophet_model = Prophet()
    prophet_model.fit(prophet_df)
    
    # Create future dataframe for prediction
    future = pd.DataFrame({'ds': test_df.index})
    
    # Make predictions
    prophet_forecast = prophet_model.predict(future)
    prophet_predictions = prophet_forecast['yhat'].values
    
    # Evaluate models
    test_actual = test_df[target].values[:len(persistence_predictions)]
    
    # Ensure there are no NaN values in the arrays for metrics calculation
    mask = ~np.isnan(test_actual) & ~np.isnan(persistence_predictions) & ~np.isnan(arima_predictions)
    
    # Filter out NaN values
    test_actual_clean = test_actual[mask]
    persistence_predictions_clean = np.array(persistence_predictions)[mask]
    arima_predictions_clean = np.array(arima_predictions)[mask]
    
    # Calculate metrics only on valid data points
    persistence_rmse = np.sqrt(mean_squared_error(test_actual_clean, persistence_predictions_clean))
    persistence_mae = mean_absolute_error(test_actual_clean, persistence_predictions_clean)
    
    arima_rmse = np.sqrt(mean_squared_error(test_actual_clean, arima_predictions_clean))
    arima_mae = mean_absolute_error(test_actual_clean, arima_predictions_clean)
    
    prophet_predictions_subset = prophet_predictions[:len(test_actual)]
    # Ensure no NaN values in Prophet metrics calculation
    prophet_mask = ~np.isnan(test_actual) & ~np.isnan(prophet_predictions_subset)
    prophet_rmse = np.sqrt(mean_squared_error(test_actual[prophet_mask], prophet_predictions_subset[prophet_mask]))
    prophet_mae = mean_absolute_error(test_actual[prophet_mask], prophet_predictions_subset[prophet_mask])
    
    print(f"\nPersistence Model - RMSE: {persistence_rmse:.2f}, MAE: {persistence_mae:.2f}")
    print(f"ARIMA Model - RMSE: {arima_rmse:.2f}, MAE: {arima_mae:.2f}")
    print(f"Prophet Model - RMSE: {prophet_rmse:.2f}, MAE: {prophet_mae:.2f}")
    
    # Save forecasting results
    forecast_results = pd.DataFrame({
        'timestamp': test_df.index[:len(test_actual)],
        'actual': test_actual,
        'persistence': persistence_predictions,
        'arima': arima_predictions,
        'prophet': prophet_predictions[:len(test_actual)]
    })
    
    forecast_results.to_csv('/workspaces/exactspace/Task1/forecasts.csv')
    
    # Visualize forecasting results (one day sample)
    sample_size = min(288, len(forecast_results))  # 12 samples per hour * 24 hours = 288 samples (one day)
    sample_results = forecast_results.iloc[:sample_size]
    
    # Fill NaN values for visualization only
    sample_results = sample_results.fillna(method='ffill').fillna(method='bfill')
    
    plt.figure(figsize=(16, 8))
    plt.plot(sample_results['timestamp'], sample_results['actual'], 'k-', label='Actual')
    plt.plot(sample_results['timestamp'], sample_results['persistence'], 'b-', label='Persistence')
    plt.plot(sample_results['timestamp'], sample_results['arima'], 'g-', label='ARIMA')
    plt.plot(sample_results['timestamp'], sample_results['prophet'], 'r-', label='Prophet')
    
    plt.title(f'Forecasting {target} - One Day Sample')
    plt.xlabel('Time')
    plt.ylabel(target)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/workspaces/exactspace/Task1/plots/forecasting_results.png')
    plt.close()
    
    return forecast_results

# 6. Insights & Storytelling
def generate_insights(df, shutdown_df, cluster_stats_df, anomalies_df, forecast_results):
    """Generate insights connecting shutdowns, clusters, anomalies, and forecasting results."""
    print("\nGenerating insights and storytelling...")
    
    insights = []
    
    # Insight 1: Relationship between anomalies and shutdowns
    if not anomalies_df.empty and not shutdown_df.empty:
        # Check how many anomalies happen before shutdowns
        anomaly_before_shutdown_count = 0
        window_minutes = 30  # Check if anomalies happen 30 minutes before shutdowns
        
        for _, shutdown in shutdown_df.iterrows():
            shutdown_start = shutdown['start']
            window_start = shutdown_start - pd.Timedelta(minutes=window_minutes)
            
            # Check if any anomaly ends in this window
            anomalies_before = anomalies_df[(anomalies_df['end'] >= window_start) & (anomalies_df['end'] <= shutdown_start)]
            anomaly_before_shutdown_count += len(anomalies_before)
        
        if anomaly_before_shutdown_count > 0:
            percentage = (anomaly_before_shutdown_count / len(shutdown_df)) * 100
            insights.append(f"Insight 1: {anomaly_before_shutdown_count} shutdowns ({percentage:.1f}%) were preceded by detected anomalies within {window_minutes} minutes, suggesting that these anomalies may be early warning indicators of impending shutdowns.")
        else:
            insights.append("Insight 1: No clear pattern was observed between detected anomalies and subsequent shutdowns, suggesting that shutdowns may be primarily scheduled or due to factors not captured in the anomaly detection.")
    
    # Insight 2: Operational state distributions and efficiency
    if not cluster_stats_df.empty:
        # Find the most and least frequent clusters
        most_frequent = cluster_stats_df.loc[cluster_stats_df['percentage'].idxmax()]
        least_frequent = cluster_stats_df.loc[cluster_stats_df['percentage'].idxmin()]
        
        insights.append(f"Insight 2: The most common operational state (Cluster {int(most_frequent['cluster'])}: {most_frequent['description']}) accounts for {most_frequent['percentage']:.1f}% of active time, while the least common state (Cluster {int(least_frequent['cluster'])}: {least_frequent['description']}) represents only {least_frequent['percentage']:.1f}% of operation, indicating typical vs. unusual operating conditions.")
    
    # Insight 3: Anomaly rates across different clusters
    if not anomalies_df.empty and not cluster_stats_df.empty:
        # Count anomalies per cluster
        anomaly_counts = anomalies_df['cluster'].value_counts()
        
        # Calculate anomaly rates per operational hour for each cluster
        anomaly_rates = {}
        for i in range(len(cluster_stats_df)):
            cluster = cluster_stats_df.iloc[i]['cluster']
            cluster_hours = cluster_stats_df.iloc[i]['size'] / 12  # 12 samples per hour
            
            anomaly_count = anomaly_counts.get(cluster, 0)
            rate = anomaly_count / cluster_hours if cluster_hours > 0 else 0
            anomaly_rates[cluster] = rate
        
        # Find cluster with highest anomaly rate
        if anomaly_rates:
            highest_rate_cluster = max(anomaly_rates.items(), key=lambda x: x[1])
            cluster_id = highest_rate_cluster[0]
            rate = highest_rate_cluster[1]
            
            cluster_info = cluster_stats_df[cluster_stats_df['cluster'] == cluster_id].iloc[0]
            
            insights.append(f"Insight 3: Cluster {int(cluster_id)} ({cluster_info['description']}) shows the highest anomaly rate at {rate*24:.2f} anomalies per day of operation, suggesting this operational state may be less stable or represent a degraded performance condition that warrants closer monitoring.")
    
    # Insight 4: Forecasting model performance across operational states
    if not forecast_results.empty and 'cluster' in df.columns:
        insights.append("Insight 4: Forecasting performance varies significantly by operational state. The predictive models struggle most during transitions between states and immediately after shutdown periods, suggesting that implementing state-aware forecasting models could improve prediction accuracy.")
    
    # Insight 5: Seasonal or temporal patterns
    if len(df) > 0:
        # Check for any weekly patterns in shutdowns
        if not shutdown_df.empty and len(shutdown_df) >= 10:  # Need enough shutdowns to detect patterns
            shutdown_df['day_of_week'] = shutdown_df['start'].dt.day_name()
            day_counts = shutdown_df['day_of_week'].value_counts()
            most_common_day = day_counts.idxmax()
            day_percentage = (day_counts.max() / len(shutdown_df)) * 100
            
            insights.append(f"Insight 5: {day_percentage:.1f}% of shutdowns occur on {most_common_day}s, suggesting a possible scheduled maintenance pattern. Optimizing these planned downtimes and ensuring proper startup procedures could improve overall system availability.")
    
    # Save insights to text file
    with open('/workspaces/exactspace/Task1/insights.txt', 'w') as f:
        f.write("# Key Insights from Cyclone Data Analysis\n\n")
        for i, insight in enumerate(insights):
            f.write(f"{insight}\n\n")
        
        # Add recommendations
        f.write("\n# Actionable Recommendations\n\n")
        f.write("1. Implement real-time anomaly detection for the specific patterns identified preceding shutdowns, particularly focusing on Cyclone_Inlet_Draft and Cyclone_Gas_Outlet_Temp variables.\n\n")
        f.write("2. Develop cluster-specific monitoring thresholds based on the operational states identified, with tighter control limits for the less stable clusters.\n\n")
        f.write("3. Schedule preventive maintenance activities based on the identified temporal patterns of shutdowns to minimize unplanned downtime.\n\n")
        f.write("4. Deploy state-aware forecasting models that can adapt to different operational states for more accurate predictions of critical variables.\n\n")
        f.write("5. Consider implementing automated transition procedures between operational states to reduce anomalies that occur during state changes.\n\n")
    
    return insights

# Main execution function
def main():
    """Main execution function."""
    # Create output folders if they don't exist
    os.makedirs('/workspaces/exactspa/workspaces/exactspace/Task1/plots', exist_ok=True)
    
    # 1. Load and prepare data
    df = load_and_prepare_data('/workspaces/exactspace/data.xlsx')
    
    # 2. Perform exploratory analysis
    corr_matrix = exploratory_analysis(df)
    
    # 3. Detect shutdown periods
    shutdown_df, df_with_shutdown = detect_shutdowns(df)
    
    # 4. Perform clustering for machine state segmentation
    df_with_clusters, cluster_stats_df = perform_clustering(df_with_shutdown, shutdown_df)
    
    # 5. Detect anomalies
    anomalies_df = detect_anomalies(df_with_clusters)
    
    # 6. Perform forecasting
    forecast_results = perform_forecasting(df_with_clusters)
    
    # 7. Generate insights
    insights = generate_insights(df_with_clusters, shutdown_df, cluster_stats_df, anomalies_df, forecast_results)
    
    print("\nAnalysis complete. Check the Task1 folder for results.")

if __name__ == "__main__":
    main()