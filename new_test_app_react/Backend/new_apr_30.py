import hashlib
import json
import logging
import ssl
import os
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
import boto3
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from flask import Flask, jsonify, request
from flask_cors import CORS
from io import BytesIO
from pandas.tseries.holiday import USFederalHolidayCalendar
from autogen import AssistantAgent
from statistics import mean
import json
# Load environment variables
# load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Bypass SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)

# LLM Configuration
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


# Define variables and corresponding model filenames for each model type
model_types = {
    'Prophet': {
        'Total IB Units': './models/Prophet/Prophet_Total_IB_Units.pkl',
        'inventory': './models/Prophet/Prophet_Inventory_Management.pkl',
        'returns': './models/Prophet/Prophet_Customer_Returns.pkl',
        'exceptions': './models/Prophet/Prophet_Inbound_Exceptions_New.pkl',
        'wfs_china': './models/Prophet/Prophet_WFS_China.pkl'
    },
    'ARIMA': {
        'Total IB Units': './models/ARIMA/ARIMA_Total_IB_Units.pkl',
        'inventory': './models/ARIMA/ARIMA_Inventory_Management.pkl',
        'returns': './models/ARIMA/ARIMA_Customer_Returns.pkl',
        'exceptions': './models/ARIMA/ARIMA_Inbound_Exceptions_New.pkl',
        'wfs_china': './models/ARIMA/ARIMA_WFS_China.pkl'
    },
    'LightGBM': {
        'Total IB Units': './models/LightGBM/LightGBM_Total_IB_Units.pkl',
        'inventory': './models/LightGBM/LightGBM_Inventory_Management.pkl',
        'returns': './models/LightGBM/LightGBM_Customer_Returns.pkl',
        'exceptions': './models/LightGBM/LightGBM_Inbound_Exceptions_New.pkl',
        'wfs_china': './models/LightGBM/LightGBM_WFS_China.pkl'
    },
    'XGBoost': {
        'Total IB Units': './models/XGBoost/XGBoost_Total_IB_Units.pkl',
        'inventory': './models/XGBoost/XGBoost_Inventory_Management.pkl',
        'returns': './models/XGBoost/XGBoost_Customer_Returns.pkl',
        'exceptions': './models/XGBoost/XGBoost_Inbound_Exceptions_New.pkl',
        'wfs_china': './models/XGBoost/XGBoost_WFS_China.pkl'
    },
    'SARIMAX': {
        'Total IB Units': './models/SARIMAX/SARIMAX_Total_IB_Units.pkl',
        'inventory': './models/SARIMAX/SARIMAX_Inventory_Management.pkl',
        'returns': './models/SARIMAX/SARIMAX_Customer_Returns.pkl',
        'exceptions': './models/SARIMAX/SARIMAX_Inbound_Exceptions_New.pkl',
        'wfs_china': './models/SARIMAX/SARIMAX_WFS_China.pkl'
    },
    'Ensemble': {
        'Total IB Units': './models/Ensemble/Ensemble_Total_IB_Units.pkl',
        'inventory': './models/Ensemble/Ensemble_Inventory_Management.pkl',
        'returns': './models/Ensemble/Ensemble_Customer_Returns.pkl',
        'exceptions': './models/Ensemble/Ensemble_Inbound_Exceptions_New.pkl',
        'wfs_china': './models/Ensemble/Ensemble_WFS_China.pkl'
    },
    'CatBoost': {
        'Total IB Units': './models/CatBoost/CatBoost_Total_IB_Units.pkl',
        'inventory': './models/CatBoost/CatBoost_Inventory_Management.pkl',
        'returns': './models/CatBoost/CatBoost_Customer_Returns.pkl',
        'exceptions': './models/CatBoost/CatBoost_Inbound_Exceptions_New.pkl',
        'wfs_china': './models/CatBoost/CatBoost_WFS_China.pkl'
    }
}

adjustments = {
    'Prophet': 0,
    'ARIMA': 0.10,
    'LightGBM': -0.50,
    'XGBoost': 0.3,
    'SARIMAX': 0.40,
    'Ensemble': -0.2,
    'CatBoost': -0.3
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

def get_adjustment_sequence(model_type, weeks):
    if model_type == 'Prophet':
        return np.ones(weeks)
    elif model_type == 'ARIMA':
        return np.linspace(1, 1.1, weeks)
    elif model_type == 'LightGBM':
        return np.linspace(1, 0.9, weeks)
    elif model_type == 'XGBoost':
        return 1 + 0.05 * np.sin(np.linspace(0, 2 * np.pi, weeks))
    elif model_type == 'SARIMAX':
        return np.linspace(1, 1.05, weeks)
    elif model_type == 'Ensemble':
        return np.linspace(1, 0.95, weeks)
    elif model_type == 'CatBoost':
        return 1 + 0.03 * np.cos(np.linspace(0, 2 * np.pi, weeks))
    else:
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

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Reset metrics/dataframes for each request
        all_metrics = {}
        data_frames = {}

        # Extract form data
        weeks = request.form.get('weeks', default=12, type=int)
        model_type = request.form.get('model_type', default='Prophet')
        use_regressors = request.form.get('use_regressors', default='false').lower() == 'true'
        logger.info(f"Forecasting for {weeks} weeks using {model_type} models. Regressors: {use_regressors}")

        if model_type not in model_types:
            return jsonify({"error": "Invalid model type specified"}), 400

        variables = model_types[model_type]

        # Handle file upload
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
            dz = pd.read_excel(file)
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            return jsonify({"error": "Failed to read Excel file"}), 400

        # Process DataFrame
        dz.columns = dz.iloc[1]
        dz = dz[3:].reset_index(drop=True)
        dz = dz.dropna(how='all').reset_index(drop=True)
        dz_df = dz.loc[dz.first_valid_index():dz.last_valid_index()].reset_index(drop=True)
        cutoff_date = pd.to_datetime('2025-03-22')
        dz_df = dz_df[dz_df['Total IB Units'].notna().cummax()]
        dz_df = dz_df[~((dz_df['Total IB Units'].isna()) & (pd.to_datetime(dz_df['Week']) > cutoff_date))]
        dz_df = dz_df.rename(columns={
            'Inventory Management': 'inventory',
            'Customer Returns': 'returns',
            'Inbound Exceptions New': 'exceptions',
            'WFS China': 'wfs_china'
        })
        dz_df = dz_df[dz_df['Week'] <= cutoff_date]
        dz_df['Week'] = pd.to_datetime(dz_df['Week'])

        last_date = dz_df['Week'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=weeks, freq='W-SAT')
        future_df = pd.DataFrame({'Week': future_dates})

        # Holiday regressor
        if use_regressors and model_type == 'Prophet':
            start_date = dz_df['Week'].min()
            end_date = future_dates[-1]
            holiday_df = create_holiday_regressor(start_date, end_date)
            dz_df = pd.merge(dz_df, holiday_df, left_on='Week', right_on='ds', how='left')
            dz_df['holiday'] = dz_df['holiday'].fillna(0)
            future_holidays = holiday_df[holiday_df['ds'].isin(future_dates)]
            future_df = pd.merge(future_df, future_holidays, left_on='Week', right_on='ds', how='left')
            future_df['holiday'] = future_df['holiday'].fillna(0)

        # Forecasting loop
        for var, model_file in variables.items():
            model = joblib.load(model_file)
            if use_regressors and model_type == 'Prophet':
                if 'holiday' not in model.extra_regressors:
                    model.add_regressor('holiday')

            model_future = model.make_future_dataframe(periods=weeks, freq='W-SAT')
            if use_regressors and model_type == 'Prophet':
                model_future = pd.merge(model_future, holiday_df, on='ds', how='left')
                model_future['holiday'] = model_future['holiday'].fillna(0)

            forecast = model.predict(model_future)
            forecast_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(weeks)
            adjustment = adjustments[model_type]
            forecast_values['yhat'] *= (1 + adjustment)
            forecast_values['yhat_lower'] *= (1 + adjustment)
            forecast_values['yhat_upper'] *= (1 + adjustment)

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
            mask = (~np.isnan(true_values)) & (~np.isnan(yhat))
            true_values_filtered = true_values[mask]
            yhat_filtered = yhat[mask]

            if len(true_values_filtered) > 0 and len(yhat_filtered) > 0:
                mae = mean_absolute_error(true_values_filtered, yhat_filtered)
                mse = mean_squared_error(true_values_filtered, yhat_filtered)
                rmse = mse ** 0.5
                r2 = r2_score(true_values_filtered, yhat_filtered)
                mape = np.mean(np.abs((true_values_filtered - yhat_filtered) / true_values_filtered)) * 100
            else:
                mae = mse = rmse = r2 = mape = float('nan')

            all_metrics[var] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape,
            }

            data_frames[var] = future_df[['Week', var, f'{var}_lower', f'{var}_upper']]

        # Historical validation
        historical_data_frames = {}
        historical_metrics = {}
        for var, model_file in variables.items():
            model = joblib.load(model_file)
            if use_regressors and model_type == 'Prophet':
                if 'holiday' not in model.extra_regressors:
                    model.add_regressor('holiday')

            model_future = model.make_future_dataframe(periods=0)
            if use_regressors and model_type == 'Prophet':
                model_future = pd.merge(model_future, holiday_df, on='ds', how='left')
                model_future['holiday'] = model_future['holiday'].fillna(0)

            forecast = model.predict(model_future)
            forecast_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            merged_df = pd.merge(dz_df, forecast_values, left_on='Week', right_on='ds', how='inner')
            merged_df[[var, 'yhat', 'yhat_lower', 'yhat_upper']] = merged_df[[var, 'yhat', 'yhat_lower', 'yhat_upper']].fillna(0)

            scaling_factor = np.mean(merged_df[var].values / merged_df['yhat'].values)
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
                mape = np.mean(np.abs((true_values_filtered - yhat_filtered) / true_values_filtered)) * 100
            else:
                mae = mse = rmse = r2 = mape = float('nan')

            historical_metrics[var] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R²': r2,
                'MAPE': mape,
            }

            historical_data_frames[var] = merged_df[['Week', var, 'yhat', 'yhat_lower', 'yhat_upper']]
        combined_json_new = {}
        for var, df in historical_data_frames.items():
            df['Week'] = df['Week'].dt.strftime('%d-%b-%Y')
            combined_json_new[var] = json.loads(df.to_json(orient='records'))
        # Format output
        dz_df['Week'] = dz_df['Week'].dt.strftime('%d-%b-%Y')
        future_df['Week'] = future_df['Week'].dt.strftime('%d-%b-%Y')
        dz_df = dz_df[(dz_df['Total IB Units'].notna()) & (dz_df['Total IB Units'] != 0)]
        final_df = pd.concat([dz_df, future_df], ignore_index=True)
        final_df['Week'] = pd.to_datetime(final_df['Week']).dt.strftime('%d-%b-%Y')

        response = {
            'future_df': json.loads(future_df.to_json(orient='records')),
            'dz_df': json.loads(dz_df.to_json(orient='records')),
            'final_df': json.loads(final_df.to_json(orient='records')),
            'metrics': all_metrics,
            'combined': combined_json_new
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in forecast: {str(e)}")
        return jsonify({"error": str(e)}), 500

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
                return "No valid data for analysis"

            actual_weeks = [(d.get("Week", i), d.get(self.actual_col, 0)) for i, d in enumerate(actual_data) if d.get(self.actual_col) is not None]
            forecast_weeks = [(d.get("Week", i), d.get(self.forecast_col, 0)) for i, d in enumerate(forecast_data) if d.get(self.forecast_col) is not None]

            if not actual_weeks or not forecast_weeks:
                return "No valid data points"

            actual_values = [v for _, v in actual_weeks]
            forecast_values = [v for _, v in forecast_weeks]

            actual_avg = mean(actual_values) if actual_values else 0
            forecast_avg = mean(forecast_values) if forecast_values else 0
            change_percent = ((forecast_avg - actual_avg) / actual_avg * 100) if actual_avg else 0

            message = (
                f"Analyzing {self.category}:\n"
                f"- Past average: {actual_avg:.2f}\n"
                f"- Future average: {forecast_avg:.2f}\n"
                f"- Percentage change: {change_percent:.2f}%\n"
                f"Provide insights on trends, risks, and recommendations in 3-5 lines."
            )

            response = self.agent.generate_reply(messages=[{"content": message, "role": "user"}])
            return response.content if hasattr(response, 'content') else response

        except Exception as e:
            logging.error(f"Error in {self.category} analysis: {str(e)}")
            return f"Analysis failed for {self.category}: {str(e)}"

def create_agents():
    return {
        "Total IB Units": InsightAgent("total_ib_agent", "Total IB Units", "Total IB Units", "Inventory Buffer"),
        "Inventory Management": InsightAgent("inventory_agent", "inventory", "inventory", "Inventory Management"),
        "Customer Returns": InsightAgent("returns_agent", "returns", "returns", "Customer Returns"),
        "Inbound Exceptions": InsightAgent("exceptions_agent", "exceptions", "exceptions", "Inbound Exceptions"),
        "WFS China": InsightAgent("wfs_china_agent", "wfs_china", "wfs_china", "China Operations"),
    }

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

        dz_df = api_data.get("dz_df", [])
        future_df = api_data.get("future_df", [])

        if not dz_df or not future_df:
            return jsonify({"error": "Missing data for analysis"}), 400

        agents = create_agents()
        insights = {}

        for category, agent in agents.items():
            actual_data = [d for d in dz_df if agent.actual_col in d]
            forecast_data = [d for d in future_df if agent.forecast_col in d]
            insights[category] = agent.analyze(actual_data, forecast_data)

        return jsonify({
            "insights": insights,
            "timestamp": datetime.now().isoformat(),
            # "raw_forecast_response": api_data
        })

    except Exception as e:
        logging.error(f"Error in analyze_forecasts: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5011, debug=True)