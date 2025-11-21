import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from datetime import datetime, timedelta
import itertools
from matplotlib import pyplot as plt
import os
import re


def fill_missing_time_rows(
    df: pd.DataFrame,
    combination_cols: list,
    date_col: str,
    freq: str,
    fill_zero_cols: list
) -> pd.DataFrame:
    """
    Fill missing time rows in a dataframe for each group defined by combination_cols.
    
    Parameters:
    - df: Input DataFrame
    - combination_cols: Columns to group by (e.g., ['store_cd', 'product_cd'])
    - date_col: The name of the datetime column (e.g., 'weekkey')
    - freq: Frequency string for date range (e.g., 'W-SUN' for weekly on Sundays)
    - fill_zero_cols: List of columns to be filled with 0 for inserted rows
    
    Returns:
    - DataFrame with missing time rows filled
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    global_max_date = df[date_col].max()

    # List to hold completed group data
    result_frames = []

    # Group by the combinations
    grouped = df.groupby(combination_cols, sort=False)

    for group_keys, group_df in grouped:
        group_df = group_df.copy()
        group_min_date = group_df[date_col].min()
        full_date_range = pd.date_range(start=group_min_date, end=global_max_date, freq=freq)

        # Create a DataFrame for the full date range
        full_index = pd.DataFrame({date_col: full_date_range})

        # Add back combination columns to align
        if len(combination_cols) == 1:
            for col in combination_cols:
                full_index[col] = group_keys
        else:
            for col, val in zip(combination_cols, group_keys):
                full_index[col] = val

        # Merge with existing group data
        merged = full_index.merge(group_df, on=combination_cols + [date_col], how='left')

        # Fill specific columns with 0
        for col in fill_zero_cols:
            merged[col] = merged[col].fillna(0)

        # Forward fill remaining columns assuming they are constant within group
        for col in df.columns:
            if col not in combination_cols + [date_col] + fill_zero_cols:
                # merged[col] = merged[col].fillna(method='ffill').fillna(method='bfill')
                merged[col] = merged[col].ffill().bfill()
        result_frames.append(merged)

    final_df = pd.concat(result_frames, ignore_index=True)
    return final_df

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX 
tbats_available = False
try:
    from tbats import TBATS
    tbats_available = True
except ImportError:
    tbats_available = False
    warnings.warn("TBATS package not installed. TBATS model will be skipped. Install with: pip install tbats")

def forecast_time_series(data, forecast_date, date_column="Date", target_column="Weekly_Sales", 
                         forecast_periods=21, plot_results=True,results_save=False, max_plot_date = None,
                         models=None, seasonality_mode='additive',
                         es_seasonal_periods=52, prophet_yearly_seasonality=True,
                         prophet_weekly_seasonality=True, prophet_daily_seasonality=False,
                         sarima_seasonal_periods=[52], sarima_orders=None,
                         tbats_seasonal_periods=[52, 4], fit_tbats=True):
    """
    Forecast time series data using multiple forecasting methods: Exponential Smoothing,
    Prophet, SARIMA, and TBATS. Data before the forecast_date is used for training,
    and forecasts are generated from that date forward.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Time series data with date and target columns
    forecast_date : str or datetime
        Date from which to start forecasting. Data before this date will be used for training.
    date_column : str, default='ds'
        Name of the date column
    target_column : str, default='y'
        Name of the target column
    forecast_periods : int, default=30
        Number of periods to forecast after the forecast_date
    plot_results : bool, default=True
        Whether to plot the forecasting results
    models : list, default=None 
        List of models to fit. Options: ['es', 'prophet', 'sarima', 'tbats']. 
        If None, all available models will be used.
    seasonality_mode : str, default='additive'
        Seasonality mode for methods that support it ('additive' or 'multiplicative')
    es_seasonal_periods : int, default=7
        Seasonal periods for Exponential Smoothing
    prophet_yearly_seasonality : bool, default=True
        Whether to include yearly seasonality in Prophet
    prophet_weekly_seasonality : bool, default=True
        Whether to include weekly seasonality in Prophet
    prophet_daily_seasonality : bool, default=False
        Whether to include daily seasonality in Prophet
    sarima_seasonal_periods : list, default=[52, 4]
        List of seasonal periods for SARIMA models
    sarima_orders : list of tuples, default=None
        List of (p,d,q) orders for SARIMA. If None, a grid search will be performed.
    tbats_seasonal_periods : list, default=[52, 4]
        List of seasonal periods for TBATS model
    fit_tbats : bool, default=True
        Whether to fit TBATS model (can be slow for large datasets)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'original_data': original input data
        - 'training_data': data used for training (before forecast_date)
        - 'test_data': data after forecast_date (if available)
        - 'forecast_df': DataFrame containing forecasts from all models
        - 'fitted_models': dictionary of fitted model objects
        - 'evaluation_metrics': dict of evaluation metrics if test data is available
    """
    # Check which models to fit
    available_models = [
                        'es', 
                        # 'prophet'
                        'sarima'
                        ]
    if sarima_orders is not None or sarima_seasonal_periods is not None:
        available_models.append('sarima')
    if tbats_available and fit_tbats:
        available_models.append('tbats')
    
    if models is None:
        models = available_models
    else:
        # Validate selected models
        models = [m.lower() for m in models if m.lower() in available_models]
        if not models:
            raise ValueError(f"No valid models selected. Available models: {available_models}")
    
    # Make a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Ensure the DataFrame has the expected column names for Prophet
    if date_column != 'ds':
        df.rename(columns={date_column: 'ds'}, inplace=True)
    if target_column != 'y':
        df.rename(columns={target_column: 'y'}, inplace=True)
    
    # Ensure the date column is in datetime format
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Convert forecast_date to datetime if it's not already
    if not isinstance(forecast_date, datetime):
        forecast_date = pd.to_datetime(forecast_date)
    
    # Sort the data by date
    df = df.sort_values('ds').reset_index(drop=True)
    
    # Split the data into training and test sets based on forecast_date
    train_df = df[df['ds'] < forecast_date].copy()
    test_df = df[df['ds'] >= forecast_date].copy()
    
    
    # Check if we have enough training data
    if len(train_df) < max([es_seasonal_periods] + sarima_seasonal_periods + tbats_seasonal_periods):
        print(f"Not enough training data. Need more historical points given the seasonal periods.")
        return {}
    
    # Determine the frequency of the data for creating future dates
    if len(df) >= 2:
        # Calculate the most common time delta between consecutive points
        date_diffs = df['ds'].diff().dropna()
        if not date_diffs.empty:
            # Find the most common time difference
            most_common_diff = date_diffs.mode()[0]
            if most_common_diff <= timedelta(hours=1):
                freq = 'H'  # Hourly
            elif most_common_diff <= timedelta(days=1):
                freq = 'D'  # Daily
            elif most_common_diff <= timedelta(days=7):
                freq = 'W'  # Weekly
            elif most_common_diff <= timedelta(days=31):
                freq = 'M'  # Monthly
            else:
                freq = 'D'  # Default to daily
        else:
            freq = 'D'  # Default to daily
    else:
        freq = 'D'  # Default to daily
    
    # Create future dates for forecasting
    # If test data exists, use those dates first, then extend if necessary
    if len(test_df) > 0:
        future_dates = test_df['ds'].tolist()
        if len(future_dates) < forecast_periods:
            # Need to extend beyond available test data
            last_test_date = future_dates[-1]
            additional_dates = pd.date_range(
                start=last_test_date + pd.Timedelta(days=1), 
                periods=forecast_periods - len(future_dates),
                freq=freq
            )
            future_dates.extend(additional_dates)
    else:
        # Create all future dates since we have no test data
        future_dates = pd.date_range(
            start=forecast_date,
            periods=forecast_periods,
            freq=freq
        )
    
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Initialize results dictionary
    results = {
        'original_data': df,
        'training_data': train_df,
        'test_data': test_df if not test_df.empty else None,
        'forecast_date': forecast_date
    }
    
    # Dictionary to store fitted models
    fitted_models = {}
    
    # Initialize forecast DataFrame
    forecast_df = pd.DataFrame({'ds': pd.concat([train_df['ds'], future_df['ds']]).unique()})
    forecast_df = forecast_df.sort_values('ds').reset_index(drop=True)
    
    # Add actual values where available
    forecast_df = pd.merge(forecast_df, df[['ds', 'y']], on='ds', how='left')
    
    # ---------------
    # Exponential Smoothing
    # ---------------
    if 'es' in models:
        # Prepare data for Exponential Smoothing
        es_data = train_df.set_index('ds')['y']
        
        # Initialize and fit Exponential Smoothing model
        es_model = ExponentialSmoothing(
            es_data, 
            seasonal=seasonality_mode,
            seasonal_periods=es_seasonal_periods,
            trend='add'
        )
        es_fit = es_model.fit(optimized=True)
        fitted_models['es'] = es_fit
        
        # Forecast future values
        es_forecast = es_fit.forecast(len(future_dates))
        
        # Create a DataFrame with forecasted values
        es_forecast_df = pd.DataFrame({
            'ds': future_dates[:len(es_forecast)],
            'es_forecast': es_forecast.values
        })
        
        # Add fitted values (in-sample predictions)
        es_fitted = es_fit.fittedvalues
        es_fitted_df = pd.DataFrame({
            'ds': es_fitted.index,
            'es_fitted': es_fitted.values
        })
        
        # Add ES fitted values to forecast_df
        forecast_df = pd.merge(forecast_df, es_fitted_df, on='ds', how='left')
        
        # Add ES forecast values to forecast_df
        forecast_df = pd.merge(forecast_df, es_forecast_df, on='ds', how='left')
        
        # Create a combined ES column (fitted + forecast)
        forecast_df['es_prediction'] = forecast_df['es_fitted'].combine_first(forecast_df['es_forecast'])
    
    # ---------------
    # Prophet
    # ---------------
    if 'prophet' in models:
        # Initialize and fit Prophet model
        prophet_model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=prophet_yearly_seasonality,
            weekly_seasonality=prophet_weekly_seasonality,
            daily_seasonality=prophet_daily_seasonality
        )
        prophet_model.fit(train_df)
        fitted_models['prophet'] = prophet_model
        
        # Create a DataFrame for future predictions
        prophet_future = future_df.copy()
        
        # Generate forecast
        prophet_forecast = prophet_model.predict(prophet_future)
        prophet_forecast_df = prophet_forecast[['ds', 'yhat']].rename(columns={'yhat': 'prophet_forecast'})
        
        # Get in-sample predictions for training data
        prophet_fitted = prophet_model.predict(pd.DataFrame({'ds': train_df['ds']}))
        prophet_fitted_df = prophet_fitted[['ds', 'yhat']].rename(columns={'yhat': 'prophet_fitted'})
        
        # Add Prophet fitted values to forecast_df
        forecast_df = pd.merge(forecast_df, prophet_fitted_df, on='ds', how='left')
        
        # Add Prophet forecast values to forecast_df
        forecast_df = pd.merge(forecast_df, prophet_forecast_df, on='ds', how='left')
        
        # Create a combined Prophet column (fitted + forecast)
        forecast_df['prophet_prediction'] = forecast_df['prophet_fitted'].combine_first(forecast_df['prophet_forecast'])
    
    # ---------------
    # SARIMA
    # ---------------
    if 'sarima' in models:
        # If no orders are specified, perform a simple grid search for the best SARIMA model
        if sarima_orders is None:
            # Define a simple grid of SARIMA parameters - this can be expanded for more comprehensive search
            p = d = q = range(0, 2)
            pdq = list(itertools.product(p, d, q))
            
            best_aic = float('inf')
            best_order = None
            best_seasonal_order = None
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                for order in pdq:
                    for seasonal_period in sarima_seasonal_periods:
                        # Check if we have enough data for this seasonal period
                        if len(train_df) < 2 * seasonal_period:
                            continue
                        
                        # Try a simple seasonal order
                        seasonal_order = (1, 1, 1, seasonal_period)
                        try:
                            model = SARIMAX(train_df['y'], order=order, seasonal_order=seasonal_order)
                            results = model.fit(disp=False)
                            if results.aic < best_aic:
                                best_aic = results.aic
                                best_order = order
                                best_seasonal_order = seasonal_order
                        except:
                            continue
            
            if best_order is None:
                warnings.warn("Could not find a suitable SARIMA model. Skipping SARIMA.")
            else:
                # print(f"Best SARIMA order: {best_order}, seasonal order: {best_seasonal_order}")
                
                # Fit the best model
                sarima_model = SARIMAX(train_df['y'], order=best_order, seasonal_order=best_seasonal_order)
                sarima_fit = sarima_model.fit(disp=False)
                fitted_models['sarima'] = sarima_fit
                
                # In-sample predictions
                sarima_fitted = sarima_fit.get_prediction(start=0, end=len(train_df)-1).predicted_mean
                sarima_fitted_df = pd.DataFrame({
                    'ds': train_df['ds'],
                    'sarima_fitted': sarima_fitted
                })
                
                # Out-of-sample forecast
                sarima_forecast = sarima_fit.get_forecast(steps=len(future_dates))
                sarima_forecast_df = pd.DataFrame({
                    'ds': future_dates[:len(sarima_forecast.predicted_mean)],
                    'sarima_forecast': sarima_forecast.predicted_mean
                })
                
                # Add to forecast_df
                forecast_df = pd.merge(forecast_df, sarima_fitted_df, on='ds', how='left')
                forecast_df = pd.merge(forecast_df, sarima_forecast_df, on='ds', how='left')
                forecast_df['sarima_prediction'] = forecast_df['sarima_fitted'].combine_first(forecast_df['sarima_forecast'])
        else:
            # Use specified orders
            best_aic = float('inf')
            best_model = None
            best_order = None
            best_seasonal_order = None
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                for order in sarima_orders:
                    for seasonal_period in sarima_seasonal_periods:
                        # Basic seasonal order - adjust as needed
                        seasonal_order = (1, 1, 1, seasonal_period)
                        try:
                            model = SARIMAX(train_df['y'], order=order, seasonal_order=seasonal_order)
                            fit = model.fit(disp=False)
                            if fit.aic < best_aic:
                                best_aic = fit.aic
                                best_model = fit
                                best_order = order
                                best_seasonal_order = seasonal_order
                        except:
                            continue
            
            if best_model is not None:
                # print(f"Selected SARIMA order: {best_order}, seasonal order: {best_seasonal_order}")
                sarima_fit = best_model
                fitted_models['sarima'] = sarima_fit
                
                # In-sample predictions
                sarima_fitted = sarima_fit.get_prediction(start=0, end=len(train_df)-1).predicted_mean
                sarima_fitted_df = pd.DataFrame({
                    'ds': train_df['ds'],
                    'sarima_fitted': sarima_fitted
                })
                
                # Out-of-sample forecast
                sarima_forecast = sarima_fit.get_forecast(steps=len(future_dates))
                sarima_forecast_df = pd.DataFrame({
                    'ds': future_dates[:len(sarima_forecast.predicted_mean)],
                    'sarima_forecast': sarima_forecast.predicted_mean
                })
                
                # Add to forecast_df
                forecast_df = pd.merge(forecast_df, sarima_fitted_df, on='ds', how='left')
                forecast_df = pd.merge(forecast_df, sarima_forecast_df, on='ds', how='left')
                forecast_df['sarima_prediction'] = forecast_df['sarima_fitted'].combine_first(forecast_df['sarima_forecast'])
            else:
                warnings.warn("Could not fit any of the specified SARIMA models. Skipping SARIMA.")
    
    # ---------------
    # TBATS
    # ---------------
    if 'tbats' in models and tbats_available and fit_tbats:
        try:
            # Initialize and fit TBATS model
            tbats_model = TBATS(seasonal_periods=tbats_seasonal_periods, use_arma_errors=True)
            tbats_fit = tbats_model.fit(train_df['y'])
            fitted_models['tbats'] = tbats_fit
            
            # In-sample predictions
            tbats_fitted = tbats_fit.y_hat
            tbats_fitted_df = pd.DataFrame({
                'ds': train_df['ds'],
                'tbats_fitted': tbats_fitted
            })
            
            # Out-of-sample forecast
            tbats_forecast = tbats_fit.forecast(steps=len(future_dates))
            tbats_forecast_df = pd.DataFrame({
                'ds': future_dates[:len(tbats_forecast)],
                'tbats_forecast': tbats_forecast
            })
            
            # Add to forecast_df
            forecast_df = pd.merge(forecast_df, tbats_fitted_df, on='ds', how='left')
            forecast_df = pd.merge(forecast_df, tbats_forecast_df, on='ds', how='left')
            forecast_df['tbats_prediction'] = forecast_df['tbats_fitted'].combine_first(forecast_df['tbats_forecast'])
        except Exception as e:
            warnings.warn(f"Error fitting TBATS model: {str(e)}. Skipping TBATS.")
    
    # Add to results
    results['forecast_df'] = forecast_df
    results['fitted_models'] = fitted_models
    # print(results.keys)
    # print(data)
    for col in [c for c in data.columns if c not in [date_column, target_column, 'intersection','IsHoliday']]:
        results[col] = data[col].unique()[0]
    # ---------------
    # Evaluation metrics for test set (if available)
    # ---------------
    evaluation_metrics = {}
    
    if not test_df.empty:
        test_dates = test_df['ds'].tolist()
        test_df_subset = forecast_df[forecast_df['ds'].isin(test_dates)]
        
        # Calculate metrics for each model
        for model_name in models:
            col_name = f"{model_name}_prediction"
            if col_name in test_df_subset.columns and test_df_subset[col_name].notna().any():
                rmse = np.sqrt(np.mean((test_df_subset['y'] - test_df_subset[col_name])**2))
                wmape = np.sum(np.abs((test_df_subset['y'] - test_df_subset[col_name])) / np.sum(test_df_subset['y'])) * 100
                mape = np.mean(np.abs((test_df_subset['y'] - test_df_subset[col_name]) / test_df_subset['y'])) * 100
                
                evaluation_metrics[model_name] = {
                    'rmse': rmse,
                    'wmape': wmape,
                    'mape': mape
                }
    
    results['evaluation_metrics'] = evaluation_metrics
    
    # Find the model with the lowest WMAPE
    
    # ---------------
    # Plotting - Simplified line plots
    # ---------------
    results_dir = "Results"
    data_dir = os.path.join(results_dir, "predictions")
    if plot_results:
        # Create directories for saving results and plots
        
        plots_dir = os.path.join(results_dir, "Plots")
        
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # Plotting
        fig, ax = plt.subplots(figsize=(27, 10))
        
        # Plot historical data
        ax.plot(df['ds'], df['y'], 'k-', linewidth=1.5, label='Historical Data')
        months = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='MS')
        # Plot the forecasts for each model
        colors = ['r', 'b', 'g', 'c', 'm']
        model_colors = dict(zip(models, colors[:len(models)]))
        if max_plot_date:
            plot_df = forecast_df.loc[(forecast_df['ds'] >= forecast_date)&
                                    (forecast_df['ds'] <= max_plot_date)]
        else:
            plot_df = forecast_df.loc[(forecast_df['ds'] >= forecast_date)]
        best_model = None
        best_wmape = float('inf')
        for i, model_name in enumerate(models):
            if model_name in results['evaluation_metrics'] and results['evaluation_metrics'][model_name]['wmape'] < best_wmape:
                best_model = model_name
                best_wmape = results['evaluation_metrics'][model_name]['wmape']
            col_name  = f"{model_name}_prediction"
            if col_name in forecast_df.columns:
            
                ax.plot(
                    plot_df['ds'], 
                    plot_df[col_name], 
                    f"{model_colors[model_name]}-", 
                    linewidth=2, 
                    label=f"{model_name.upper()}"
                )
        
        # Mark the forecast date
        # ax.axvline(x=forecast_date, color='gray', linestyle='--', label='Forecast Date')
        ax.axvspan(forecast_date, 
                   plot_df.loc[plot_df['y'].notna()]['ds'].max(), 
                   color='lightblue', alpha=0.3, label = 'Forecast Period')

        # Add labels and legend
        # ax.set_title(f'SKU: {int(float(data['product_id'].unique()[0].split('_')[-2]))} - Subclass: {data['product_id'].unique()[0].split('_')[-1]} Forecast', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(months)
        ax.set_xticklabels([d.strftime('%b %Y') for d in months], rotation=45)

        # Annotate the best model and its WMAPE
        if best_model:
            annotation_text = f"Best Model: {best_model.upper()}\nWMAPE: {best_wmape:.2f}%"
            ax.annotate(
                annotation_text,
                xy=(forecast_date, df['y'].min()),  # Position near the forecast date and max value
                xytext=(forecast_date + timedelta(days=10), df['y'].min() * 0.9),  # Offset for better visibility
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8)
            )
        plt.show()
        
        replacement_string = '_'
        
        if results_save:    
            # Save the plot
            plot_file = os.path.join(plots_dir, re.sub(r"[|/]", replacement_string,f"forecast_plot_{forecast_date.strftime('%Y%m%d')}_{data['product_id'].unique()[0]}.png"))
            plt.tight_layout()
            plt.savefig(plot_file)
            plt.show()
            
            results_file = os.path.join(data_dir, re.sub(r"[|/]", replacement_string, f"forecast_results_{forecast_date.strftime('%Y%m%d')}_{data['product_id'].unique()[0]}.pkl"))
            pd.to_pickle(results, results_file)
            print(f"Plot saved to: {plot_file}")
            print(f"Results saved to: {results_file}")


    return results

# Example usage
if __name__ == "__main__":
    # Create a synthetic example dataset
    
    df = pd.read_csv("Data/train.csv",parse_dates=["Date"])
    np.random.seed(42)  # deterministic assignments

    # Dept -> (Division, Category)
    unique_depts = list(df['Dept'].unique())
    divisions = list("ABCD")
    dept_divisions = dict(zip(unique_depts, np.random.choice(divisions, size=len(unique_depts), replace=True)))
    dept_categories = dict(zip(unique_depts, np.random.choice(list(range(7)), size=len(unique_depts), replace=True)))

    # Store -> Country
    countries = ["USA", "Canada", "UK", "Germany", "France"]
    unique_stores = list(df['Store'].unique())
    store_countries = dict(zip(unique_stores, np.random.choice(countries, size=len(unique_stores), replace=True)))

    # Apply mappings to dataframe
    df['Division'] = df['Dept'].map(dept_divisions)
    df['Category'] = df['Dept'].map(lambda d: f"{dept_divisions[d]}_{dept_categories[d]}")
    df['Country'] = df['Store'].map(store_countries)
    df_filled = fill_missing_time_rows(df,
                                    combination_cols = ['Country','Store','Division','Category','Dept'],
                                    date_col="Date",
                                    freq="W-FRI",
                                    fill_zero_cols=['Weekly_Sales'])
        
    df_filled['intersection'] = df_filled['Store'].astype(str) + "_" + df_filled['Dept'].astype(str)

        
    vol = df_filled.groupby('intersection')['Weekly_Sales'].sum().sort_values(ascending=False)
    top_20_percent = int(len(vol) * 0.2)
    top_intersections = vol.iloc[:top_20_percent].index
    np.random.seed(42)
    selected_intersections = np.random.choice(top_intersections, size=150, replace=False)
    
    df_filled.sort_values('Date', inplace=True)
    df_filled.reset_index(drop=True, inplace=True)

    # Set forecast date to use 80% of data for training
    forecast_date = pd.to_datetime('2012-04-26')
    final_results = {}
    for i in tqdm(selected_intersections):
        # Run the forecasting function with all available models
        forecast_results = forecast_time_series(
            data=df_filled.loc[df_filled['intersection'] == i],
            forecast_date=forecast_date,
            forecast_periods=52,
            es_seasonal_periods=52,
            sarima_seasonal_periods=[52],
            tbats_seasonal_periods=[52, 4],
            sarima_orders = [(0, 1, 1), (1, 1, 0), (1, 1, 1)],
            max_plot_date=pd.to_datetime('2013-05-01'),
            plot_results=True
        )
        if forecast_results:    
            # Access the combined forecast DataFrame
            combined_forecasts = forecast_results['forecast_df']
            # print(combined_forecasts.head())
            
            # Print evaluation metrics if available
            if forecast_results['evaluation_metrics']:
                # print("\nEvaluation Metrics:")
                for model, metrics in forecast_results['evaluation_metrics'].items():
                    print(f"{model.upper()}: RMSE = {metrics['rmse']:.4f}, WMAPE = {metrics['wmape']:.4f}, MAPE = {metrics['mape']:.2f}%")
        else:
            print(f"Error in forecasting for product ID: {i}")
            
        final_results[i] = forecast_results
    
    replacement_string = '_'
    results_dir = "Results"
    # Save the results dictionary
    results_file = os.path.join(results_dir, re.sub(r"[|/]", replacement_string, f"Stat_forecast_results_{forecast_date.strftime('%Y%m%d')}_select_articles.pkl"))
    pd.to_pickle(final_results, results_file)
    print(f"Results saved to: {results_file}")

# Convert final_results into a DataFrame and save in Results folder
summary_rows = []
for intersection, result in final_results.items():
    if result and 'forecast_df' in result and 'evaluation_metrics' in result:
        forecast_df = result['forecast_df']
        metrics = result['evaluation_metrics']
        for model, vals in metrics.items():
            # Only keep rows in forecast_df where actual sales are available (y notna)
            for _, row in forecast_df.iterrows():
                summary_rows.append({
                    'intersection': intersection,
                    'model': model,
                    'date': row['ds'],
                    'forecast': row.get(f"{model}_prediction"),
                    'actual_sales': row.get('y'),
                    'Country':result['original_data']['Country'].unique()[0],
                    'Store':result['original_data']['Store'].unique()[0],
                    'Division':result['original_data']['Division'].unique()[0],
                    'Category':result['original_data']['Category'].unique()[0],
                    'Product':result['original_data']['Dept'].unique()[0],
                    # 'rmse': vals.get('rmse'),
                    # 'wmape': vals.get('wmape'),
                    # 'mape': vals.get('mape')
                })
            
summary_df = pd.DataFrame(summary_rows)
summary_file = os.path.join(results_dir, re.sub(r"[|/]", replacement_string, f"Stat_forecast_{forecast_date.strftime('%Y%m%d')}_select_articles.csv"))
summary_df.to_csv(summary_file, index=False)
print(f"Summary DataFrame saved to: {summary_file}")
