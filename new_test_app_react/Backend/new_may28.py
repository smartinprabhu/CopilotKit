import traceback
import pandas as pd
from datetime import datetime, timedelta
import calendar
import hashlib
import json
import logging
import ssl
import os
import numpy as np
import boto3
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from flask import Flask, jsonify, request
from flask_cors import CORS
from io import BytesIO
from pandas.tseries.holiday import USFederalHolidayCalendar
from autogen import AssistantAgent
from statistics import mean

# Import Prophet
from prophet import Prophet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Bypass SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)

# LLM Configuration for Autogen agents
llm_config = {
    "config_list": [
        {
            "model": "llama3-70b-8192",
            "api_key": os.environ.get("GROQ_API_KEY", ""),
            "base_url": "https://api.groq.com/openai/v1",
            "price": [0.0, 0.0],
        }
    ],
    "cache_seed": None,
}

# S3 Configuration
ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "")
SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
BUCKET_NAME = "user1-bucket-test"
FILE_KEY = "models/Book3.xlsx"

s3 = boto3.client(
    "s3",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_ACCESS_KEY
)

# Business Unit Configurations
def get_business_unit_config():
    return {
        'wfs': {
            'timestamp_column_keywords': ['week', 'week (sat)', 'week_sat', 'week (saturday)'],
            'timestamp_column_standard_name': 'week',
            'base_measure_keywords': ['total ib units', 'total (ib units)', 'Total IB Units', 'total_ib_units'],
            'base_measure_standard_name': 'total_ib_units',
            'lobs_keywords': {
                'Inventory Management': 'inventory_management',
                'Core Support': 'core_support',
                'Customer Returns': 'customer_returns',
                'inbound exceptions new': 'inbound_exceptions_new',
                'WFS China': 'wfs_china',
                'US Phone': 'us_phone',
                'US Chat': 'us_chat'
            }
        },
        'sff': {
            'timestamp_column_keywords': ['week', 'week (sat)', 'week_sat', 'week (saturday)'],
            'timestamp_column_standard_name': 'week',
            'base_measure_keywords': ['total sellers', 'total_sellers', 'Total Sellers'],
            'base_measure_standard_name': 'total_sellers',
            'lobs_keywords': {
                'SFF Case': 'sff_case',
                'SFF Chat': 'sff_chat',
                'SFF Phone': 'sff_phone',
                'Manual Dispute': 'manual_dispute',
                'Automated Dispute': 'automated_dispute'
            }
        }
    }

# Model Paths Configuration
model_types_config = {
    'wfs': {
        'Prophet': {
            'total_ib_units': '',
            'inventory_management': '',
            'customer_returns': '',
            'inbound_exceptions_new': '',
            'us_phone': '',
            'core_support': '',
            'us_chat': '',
            'wfs_china': ''
        }
    },
    'sff': {
        'Prophet': {
            'total_sellers': '',
            'sff_case': '',
            'sff_chat': '',
            'sff_phone': '',
            'manual_dispute': '',
            'automated_dispute': ''
        }
    }
}

# Adjustments Configuration
adjustments_config = {
    'wfs': {
        'Prophet': 0
    },
    'sff': {
        'Prophet': 0
    }
}

# Utility Functions
def standardize_column_name(col_name):
    return str(col_name).strip().lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')

def get_adjustment_sequence(model_type, weeks):
    return np.ones(weeks)

def fetch_file_from_s3(bucket_name, file_key):
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read()
        return BytesIO(file_content)
    except Exception as e:
        logger.error(f"Error fetching file from S3: {e}")
        return None

def create_holiday_regressor(start_date, end_date):
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    holiday_df = pd.DataFrame({'ds': date_range})
    holiday_df['holiday'] = holiday_df['ds'].isin(holidays).astype(int)
    holiday_df['week'] = holiday_df['ds'].dt.to_period('W-SAT').dt.start_time
    weekly_holidays = holiday_df.groupby('week')['holiday'].max().reset_index()
    weekly_holidays.rename(columns={'week': 'ds'}, inplace=True)
    return weekly_holidays

# Flask Routes
@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        all_metrics = {}
        data_frames = {}
        historical_metrics = {}
        historical_data_frames = {}

        weeks = request.form.get('weeks', default=12, type=int)
        requested_model_type = request.form.get('model_type', default='Prophet')
        model_type = 'Prophet'
        use_regressors = request.form.get('use_regressors', default='false').lower() == 'true'
        logger.info(f"Forecasting for {weeks} weeks using FORCED Prophet model. Regressors: {use_regressors}")

        file = None
        if 'file' in request.files and request.files['file'].filename != '':
            uploaded_file = request.files['file']
            if not uploaded_file.filename.endswith(('.xlsx', '.xls')):
                return jsonify({"error": "Invalid file type"}), 400
            max_size = 10 * 1024 * 1024
            if uploaded_file.content_length and uploaded_file.content_length > max_size:
                return jsonify({"error": "File size exceeds limit"}), 400
            file = BytesIO(uploaded_file.read())
        else:
            file = fetch_file_from_s3(BUCKET_NAME, FILE_KEY)
            if file is None:
                return jsonify({"error": "Failed to fetch default file from S3"}), 500

        try:
            dz_df = pd.read_excel(file, header=0)
            logger.info(f"Original columns of the uploaded Excel file: {dz_df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            return jsonify({"error": "Failed to read Excel file. Please ensure it's a valid Excel format and headers are in the first row."}), 400

        original_excel_cols = dz_df.columns.tolist()
        standardized_cols_map = {col: standardize_column_name(str(col)) for col in original_excel_cols}
        dz_df.rename(columns=standardized_cols_map, inplace=True)
        logger.info(f"Standardized columns in DataFrame: {dz_df.columns.tolist()}")

        identified_business_unit = None
        variables_to_process = {}
        display_names_for_agents = {}
        timestamp_col_standard_name = None

        bu_configs = get_business_unit_config()
        for bu_name, bu_config in bu_configs.items():
            found_timestamp_col_in_df = None
            for keyword in bu_config['timestamp_column_keywords']:
                standardized_keyword_from_config = standardize_column_name(keyword)
                if standardized_keyword_from_config in dz_df.columns:
                    found_timestamp_col_in_df = standardized_keyword_from_config
                    timestamp_col_standard_name = bu_config['timestamp_column_standard_name']
                    variables_to_process[timestamp_col_standard_name] = found_timestamp_col_in_df
                    break

            if found_timestamp_col_in_df:
                dz_df[timestamp_col_standard_name] = pd.to_datetime(dz_df[timestamp_col_standard_name], errors='coerce')
                dz_df.dropna(subset=[timestamp_col_standard_name], inplace=True)
                logger.info(f"Data type of timestamp column '{timestamp_col_standard_name}': {dz_df[timestamp_col_standard_name].dtype}")

                found_base_measure_col_in_df = None
                for keyword in bu_config['base_measure_keywords']:
                    standardized_keyword_from_config = standardize_column_name(keyword)
                    if standardized_keyword_from_config in dz_df.columns:
                        found_base_measure_col_in_df = standardized_keyword_from_config
                        break

                if found_base_measure_col_in_df:
                    identified_business_unit = bu_name
                    variables_to_process[bu_config['base_measure_standard_name']] = found_base_measure_col_in_df
                    display_names_for_agents[bu_config['base_measure_standard_name']] = bu_config['base_measure_standard_name'].replace('_', ' ').title()

                    for lob_keyword, lob_standard_name in bu_config['lobs_keywords'].items():
                        standardized_lob_keyword_from_config = standardize_column_name(lob_keyword)
                        if standardized_lob_keyword_from_config in dz_df.columns:
                            variables_to_process[lob_standard_name] = standardized_lob_keyword_from_config
                            display_names_for_agents[lob_standard_name] = lob_standard_name.replace('_', ' ').title()
                    break

        if not identified_business_unit:
            return jsonify({"error": "Could not identify business unit (WFS or SFF) from the provided Excel file. Please ensure it contains a 'Week' column and relevant base measure columns like 'Total IB Units' or 'Total Sellers'."}), 400

        rename_map_for_df = {v: k for k, v in variables_to_process.items()}
        dz_df.rename(columns=rename_map_for_df, inplace=True)

        base_measure_standard_name = bu_configs[identified_business_unit]['base_measure_standard_name']
        dz_df[base_measure_standard_name] = pd.to_numeric(dz_df[base_measure_standard_name], errors='coerce')
        dz_df = dz_df[(dz_df[base_measure_standard_name].notna()) & (dz_df[base_measure_standard_name] != 0)]

        logger.info(f"Data type of base measure column '{base_measure_standard_name}': {dz_df[base_measure_standard_name].dtype}")
        logger.info(f"First few rows of the DataFrame:\n{dz_df.head()}")

        last_date = dz_df[timestamp_col_standard_name].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=weeks, freq='W-SAT')
        future_df = pd.DataFrame({timestamp_col_standard_name: future_dates})

        variables_for_forecast_keys = list(model_types_config[identified_business_unit]['Prophet'].keys())
        current_adjustment_factor = adjustments_config[identified_business_unit]['Prophet']

        holiday_df = None
        if use_regressors:
            start_date_regressor = dz_df[timestamp_col_standard_name].min()
            end_date_regressor = future_dates[-1]
            holiday_df = create_holiday_regressor(start_date_regressor, end_date_regressor)

            dz_df = pd.merge(dz_df, holiday_df, left_on=timestamp_col_standard_name, right_on='ds', how='left')
            dz_df['holiday'] = dz_df['holiday'].fillna(0)

            future_holidays = holiday_df[holiday_df['ds'].isin(future_dates)]
            future_df = pd.merge(future_df, future_holidays, left_on=timestamp_col_standard_name, right_on='ds', how='left')
            future_df['holiday'] = future_df['holiday'].fillna(0)

        for var in variables_for_forecast_keys:
            if var not in dz_df.columns:
                logger.warning(f"Variable '{var}' not found in DataFrame columns. Skipping forecast for this variable.")
                future_df[var] = np.nan
                future_df[f'{var}_lower'] = np.nan
                future_df[f'{var}_upper'] = np.nan
                all_metrics[var] = {k: float('nan') for k in ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE']}
                data_frames[var] = future_df[[timestamp_col_standard_name, var, f'{var}_lower', f'{var}_upper']]
                historical_metrics[var] = {k: float('nan') for k in ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE']}
                historical_data_frames[var] = pd.DataFrame(columns=[timestamp_col_standard_name, var, 'yhat', 'yhat_lower', 'yhat_upper'])
                continue

            dz_df[var] = pd.to_numeric(dz_df[var], errors='coerce')
            prophet_df = dz_df[[timestamp_col_standard_name, var]].rename(columns={timestamp_col_standard_name: 'ds', var: 'y'}).dropna(subset=['y'])

            logger.info(f"First few rows of prophet_df for variable '{var}':\n{prophet_df.head()}")

            model = Prophet()

            if use_regressors and 'holiday' in prophet_df.columns:
                model.add_regressor('holiday')
            elif use_regressors:
                logger.warning(f"Holiday regressor requested but 'holiday' column not found in data for {var}. Skipping regressor for this variable.")

            if not prophet_df.empty:
                model.fit(prophet_df)
            else:
                logger.warning(f"No valid data to fit Prophet model for {var}. Skipping forecast for this variable.")
                future_df[var] = np.nan
                future_df[f'{var}_lower'] = np.nan
                future_df[f'{var}_upper'] = np.nan
                all_metrics[var] = {k: float('nan') for k in ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE']}
                data_frames[var] = future_df[[timestamp_col_standard_name, var, f'{var}_lower', f'{var}_upper']]
                historical_metrics[var] = {k: float('nan') for k in ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE']}
                historical_data_frames[var] = pd.DataFrame(columns=[timestamp_col_standard_name, var, 'yhat', 'yhat_lower', 'yhat_upper'])
                continue

            model_future = model.make_future_dataframe(periods=weeks, freq='W-SAT')
            if use_regressors and holiday_df is not None:
                model_future = pd.merge(model_future, holiday_df, on='ds', how='left')
                model_future['holiday'] = model_future['holiday'].fillna(0)

            forecast_result = model.predict(model_future)
            forecast_values = forecast_result[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(weeks)

            forecast_values['yhat'] *= (1 + current_adjustment_factor)
            forecast_values['yhat_lower'] *= (1 + current_adjustment_factor)
            forecast_values['yhat_upper'] *= (1 + current_adjustment_factor)

            adjustment_sequence = get_adjustment_sequence(model_type, weeks)
            forecast_values['yhat'] *= adjustment_sequence
            forecast_values['yhat_lower'] *= adjustment_sequence
            forecast_values['yhat_upper'] *= adjustment_sequence

            future_df[var] = forecast_values['yhat'].values
            future_df[f'{var}_lower'] = forecast_values['yhat_lower'].values
            future_df[f'{var}_upper'] = forecast_values['yhat_upper'].values

            true_values = dz_df[var].tail(weeks).values
            yhat = forecast_values['yhat'].values

            true_values = np.asarray(true_values, dtype=float)
            yhat = np.asarray(yhat, dtype=float)

            min_len = min(len(true_values), len(yhat))
            true_values_aligned = true_values[:min_len]
            yhat_aligned = yhat[:min_len]

            mask = (~np.isnan(true_values_aligned)) & (~np.isnan(yhat_aligned))
            true_values_filtered = true_values_aligned[mask]
            yhat_filtered = yhat_aligned[mask]

            if len(true_values_filtered) > 0 and len(yhat_filtered) > 0:
                mae = mean_absolute_error(true_values_filtered, yhat_filtered)
                mse = mean_squared_error(true_values_filtered, yhat_filtered)
                rmse = mse ** 0.5
                r2 = r2_score(true_values_filtered, yhat_filtered)
                mape = np.mean(np.abs((true_values_filtered - yhat_filtered) / true_values_filtered[true_values_filtered != 0])) * 100 if np.any(true_values_filtered != 0) else float('inf')
            else:
                mae = mse = rmse = r2 = mape = float('nan')

            all_metrics[var] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape,
            }

            data_frames[var] = future_df[[timestamp_col_standard_name, var, f'{var}_lower', f'{var}_upper']]

        for var in variables_for_forecast_keys:
            if var not in dz_df.columns:
                logger.warning(f"Variable '{var}' not found in DataFrame columns for historical validation. Skipping historical validation for this variable.")
                historical_metrics[var] = {k: float('nan') for k in ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE']}
                historical_data_frames[var] = pd.DataFrame(columns=[timestamp_col_standard_name, var, 'yhat', 'yhat_lower', 'yhat_upper'])
                continue

            prophet_df_hist = dz_df[[timestamp_col_standard_name, var]].rename(columns={timestamp_col_standard_name: 'ds', var: 'y'}).dropna(subset=['y'])

            model_hist = Prophet()

            if use_regressors and 'holiday' in prophet_df_hist.columns:
                model_hist.add_regressor('holiday')
            elif use_regressors:
                logger.warning(f"Holiday regressor requested but 'holiday' column not found in historical data for {var}. Skipping regressor for historical validation.")

            if not prophet_df_hist.empty:
                model_hist.fit(prophet_df_hist)
            else:
                logger.warning(f"No valid historical data to fit Prophet model for {var}. Skipping historical validation for this variable.")
                historical_metrics[var] = {k: float('nan') for k in ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE']}
                historical_data_frames[var] = pd.DataFrame(columns=[timestamp_col_standard_name, var, 'yhat', 'yhat_lower', 'yhat_upper'])
                continue

            model_future_hist = model_hist.make_future_dataframe(periods=0)
            if use_regressors and holiday_df is not None:
                model_future_hist = pd.merge(model_future_hist, holiday_df, on='ds', how='left')
                model_future_hist['holiday'] = model_future_hist['holiday'].fillna(0)

            forecast_hist = model_hist.predict(model_future_hist)
            forecast_values_hist = forecast_hist[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

            merged_df = pd.merge(dz_df, forecast_values_hist, left_on=timestamp_col_standard_name, right_on='ds', how='inner')

            merged_df[[var, 'yhat', 'yhat_lower', 'yhat_upper']] = merged_df[[var, 'yhat', 'yhat_lower', 'yhat_upper']].fillna(0)

            if np.any(merged_df['yhat'].values != 0):
                scaling_factor = np.mean(merged_df[var].values / merged_df['yhat'].values)
            else:
                scaling_factor = 1.0

            merged_df['yhat'] *= scaling_factor
            merged_df['yhat_lower'] *= scaling_factor
            merged_df['yhat_upper'] *= scaling_factor

            true_values = merged_df[var].values
            yhat = merged_df['yhat'].values

            true_values = np.asarray(true_values, dtype=float)
            yhat = np.asarray(yhat, dtype=float)

            mask = (~np.isnan(true_values)) & (~np.isnan(yhat))
            true_values_filtered = true_values[mask]
            yhat_filtered = yhat[mask]

            if len(true_values_filtered) > 0 and len(yhat_filtered) > 0:
                mae = mean_absolute_error(true_values_filtered, yhat_filtered)
                mse = mean_squared_error(true_values_filtered, yhat_filtered)
                rmse = mse ** 0.5
                r2 = r2_score(true_values_filtered, yhat_filtered)
                mape = np.mean(np.abs((true_values_filtered - yhat_filtered) / true_values_filtered[true_values_filtered != 0])) * 100 if np.any(true_values_filtered != 0) else float('inf')
            else:
                mae = mse = rmse = r2 = mape = float('nan')

            historical_metrics[var] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape,
            }

            historical_data_frames[var] = merged_df[[timestamp_col_standard_name, var, 'yhat', 'yhat_lower', 'yhat_upper']]

        combined_json_new = {}
        for var, df in historical_data_frames.items():
            # Ensure the timestamp column is properly converted to datetime
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col_standard_name]):
                df[timestamp_col_standard_name] = pd.to_datetime(df[timestamp_col_standard_name], errors='coerce')

            if pd.api.types.is_datetime64_any_dtype(df[timestamp_col_standard_name]):
                df[timestamp_col_standard_name] = df[timestamp_col_standard_name].dt.strftime('%d-%b-%Y')
            else:
                logger.error(f"Column '{timestamp_col_standard_name}' is not of datetime type. Skipping formatting.")

            combined_json_new[var] = json.loads(df.to_json(orient='records'))

        # Ensure the timestamp column is properly converted to datetime
        if not pd.api.types.is_datetime64_any_dtype(dz_df[timestamp_col_standard_name]):
            dz_df[timestamp_col_standard_name] = pd.to_datetime(dz_df[timestamp_col_standard_name], errors='coerce')

        if pd.api.types.is_datetime64_any_dtype(dz_df[timestamp_col_standard_name]):
            dz_df[timestamp_col_standard_name] = dz_df[timestamp_col_standard_name].dt.strftime('%d-%b-%Y')
        else:
            logger.error(f"Column '{timestamp_col_standard_name}' is not of datetime type. Skipping formatting.")

        # Ensure the timestamp column is properly converted to datetime
        if not pd.api.types.is_datetime64_any_dtype(future_df[timestamp_col_standard_name]):
            future_df[timestamp_col_standard_name] = pd.to_datetime(future_df[timestamp_col_standard_name], errors='coerce')

        if pd.api.types.is_datetime64_any_dtype(future_df[timestamp_col_standard_name]):
            future_df[timestamp_col_standard_name] = future_df[timestamp_col_standard_name].dt.strftime('%d-%b-%Y')
        else:
            logger.error(f"Column '{timestamp_col_standard_name}' is not of datetime type. Skipping formatting.")

        final_df = pd.concat([dz_df, future_df], ignore_index=True)
        # Ensure the timestamp column is properly converted to datetime
        if not pd.api.types.is_datetime64_any_dtype(final_df[timestamp_col_standard_name]):
            final_df[timestamp_col_standard_name] = pd.to_datetime(final_df[timestamp_col_standard_name], errors='coerce')

        if pd.api.types.is_datetime64_any_dtype(final_df[timestamp_col_standard_name]):
            final_df[timestamp_col_standard_name] = final_df[timestamp_col_standard_name].dt.strftime('%d-%b-%Y')
        else:
            logger.error(f"Column '{timestamp_col_standard_name}' is not of datetime type. Skipping formatting.")

        # Map the identified business unit to its full name
        business_unit_full_name = {
            'wfs': 'Walmart Fulfillment Services',
            'sff': 'Seller Fulfilled Prime'
        }.get(identified_business_unit, identified_business_unit)

        response = {
            'future_df': json.loads(future_df.to_json(orient='records')),
            'dz_df': json.loads(dz_df.to_json(orient='records')),
            'final_df': json.loads(final_df.to_json(orient='records')),
            'metrics': all_metrics,
            'historical_metrics': historical_metrics,
            'combined': combined_json_new,
            'identified_business_unit': business_unit_full_name,
            'display_names_for_agents': display_names_for_agents
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in forecast: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

# Autogen Agent Setup
class InsightAgent:
    def __init__(self, name, actual_col, forecast_col, category):
        self.actual_col = actual_col
        self.forecast_col = forecast_col
        self.category = category
        self.agent = AssistantAgent(
            name=name,
            system_message=(
                f"You are a {category} analyst. Analyze trends in actual "
                "data (past) vs forecast data (future) using 'Week' time "
                "series. Identify significant changes (>20%). Highlight "
                "risks/opportunities. Provide operational recommendations."
            ),
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

    def analyze(self, actual_data, forecast_data):
        try:
            if not actual_data or not forecast_data:
                return "No valid data for analysis."

            actual_values = [d.get(self.actual_col, 0) for d in actual_data if d.get(self.actual_col) is not None]
            forecast_values = [d.get(self.forecast_col, 0) for d in forecast_data if d.get(self.forecast_col) is not None]

            if not actual_values or not forecast_values:
                return "No valid data points for analysis."

            actual_avg = mean(actual_values) if actual_values else 0
            forecast_avg = mean(forecast_values) if forecast_values else 0

            change_percent = 0
            if actual_avg != 0:
                change_percent = ((forecast_avg - actual_avg) / actual_avg) * 100
            elif forecast_avg != 0:
                change_percent = float('inf')

            message = (
                f"Analyzing {self.category} for {self.actual_col.replace('_', ' ').title()}:\n"
                f"- Past average: {actual_avg:.2f}\n"
                f"- Future average: {forecast_avg:.2f}\n"
                f"- Percentage change: {change_percent:.2f}%\n"
                f"Provide insights on trends, risks, and recommendations in 3-5 lines."
            )

            response = self.agent.generate_reply(messages=[{"content": message, "role": "user"}])
            return response.content if hasattr(response, 'content') else response

        except Exception as e:
            logging.error(f"Error in {self.category} analysis: {str(e)}")
            logging.error(f"Traceback:\n{traceback.format_exc()}")
            return f"Analysis failed for {self.category}: {str(e)}"

def create_agents(business_unit_name, lobs_for_analysis):
    agents = {}
    for standardized_name, display_name in lobs_for_analysis.items():
        agent_name = f"{business_unit_name}_{standardized_name.replace('_', '')}_agent"
        agents[display_name] = InsightAgent(agent_name, standardized_name, standardized_name, display_name)
    return agents

@app.route("/analyze_forecasts", methods=["POST"])
def analyze_forecasts():
    try:
        with app.test_client() as c:
            form_data = dict(request.form)
            for key in request.files:
                file = request.files[key]
                file.stream.seek(0)
                form_data[key] = (file.stream, file.filename)

            response = c.post('/forecast', data=form_data)
            api_data = response.get_json()

        if isinstance(api_data, dict) and "error" in api_data:
            return jsonify(api_data), 500

        dz_df_raw = api_data.get("dz_df", [])
        future_df_raw = api_data.get("future_df", [])
        identified_business_unit = api_data.get("identified_business_unit")
        display_names_for_agents = api_data.get("display_names_for_agents")

        if not dz_df_raw or not future_df_raw:
            return jsonify({"error": "Missing historical or forecasted data for analysis."}), 400

        if not identified_business_unit or not display_names_for_agents:
             return jsonify({"error": "Business unit identification or LOB display names missing from forecast response. Cannot perform analysis."}), 500

        agents = create_agents(identified_business_unit, display_names_for_agents)
        insights = {}

        for standardized_name, display_name in display_names_for_agents.items():
            agent = agents[display_name]

            actual_data = [d for d in dz_df_raw if standardized_name in d]
            forecast_data = [d for d in future_df_raw if standardized_name in d]

            insights[display_name] = agent.analyze(actual_data, forecast_data)

        return jsonify({
            "insights": insights,
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as e:
        logging.error(f"Error in analyze_forecasts: {str(e)}")
        logging.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5011, debug=True)
