import React, { useState, useEffect, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  AreaChart,
  Area,
  ComposedChart,
  Bar,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { useToast } from "@/hooks/use-toast";
import { Drawer, Box, Fab } from "@mui/material";
import ExcelJS from 'exceljs';
import { toZonedTime, format } from 'date-fns-tz';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Download, ChevronDown, Info, Settings, Bot, X } from "lucide-react";
import AreaChartOutlinedIcon from '@mui/icons-material/AreaChartOutlined';
import { Separator } from "@/components/ui/separator";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import axios from 'axios';
import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import NoDataFullPage from "@/components/noDataPage";
import AuthService from "@/auth/utils/authService";
import AppConfig from '../auth/config.js';
import { formatWithPrecision } from "@/lib/utils";
import AskAIButton from './AskAIButton';
import AiChat from "./AIChat";


const timeZone = 'Asia/Kolkata';

function getLocalDateString(utcDateStr: string, formatStr = 'yyyy-MM-dd', tz = timeZone): string {
  const fixedUtcString = utcDateStr.replace(' ', 'T') + 'Z';
  const zonedDate = toZonedTime(new Date(fixedUtcString), tz);
  return format(zonedDate, formatStr, { timeZone: tz });
}

function extractModelOnly(fullName: string): string {
  const parts = fullName.split("_");
  return parts.length > 1 ? parts[parts.length - 1] : fullName;
}

function extractModelNameFromItemName(name: string): string | undefined {
  // First: Try to extract something like "Prophet+XGBoost" from "..._<model>-<version>"
  const fullMatch = name.match(/_(.+?)-\d+$/);
  if (fullMatch?.[1]) return extractModelOnly(fullMatch[1]);

  // Fallback: Try to extract model like "_ARIMA-1", "_LSTM-2"
  const capitalWordMatch = name.match(/_([A-Z]+)-\d+$/);
  return capitalWordMatch?.[1] || '';
}

const modelMetricsStatic = {
  'Prophet': { MAPE: 3.8, RMSE: 900, MAE: 650, 'R²': 0.92 },
  'ARIMA': { MAPE: 4.8, RMSE: 1100, MAE: 750, 'R²': 0.88 },
  'SARIMAX': { MAPE: 5.5, RMSE: 1300, MAE: 850, 'R²': 0.82 },
  'Weighted Moving Average': { MAPE: 5.5, RMSE: 1300, MAE: 850, 'R²': 0.82 },
  'ETS': { MAPE: 5.5, RMSE: 1300, MAE: 850, 'R²': 0.82 },
  'CatBoost': { MAPE: 4.2, RMSE: 1000, MAE: 700, 'R²': 0.90 },
  'XGBoost': { MAPE: 4.5, RMSE: 1050, MAE: 720, 'R²': 0.89 },
  'LightGBM': { MAPE: 4.0, RMSE: 950, MAE: 680, 'R²': 0.91 },
  'ARIMA+LightGBM': { MAPE: 5.2, RMSE: 1200, MAE: 800, 'R²': 0.85 },
  'Prophet+XGBoost': { MAPE: 4.1, RMSE: 980, MAE: 690, 'R²': 0.91 },
  'SARIMAX+CatBoost': { MAPE: 4.3, RMSE: 1020, MAE: 710, 'R²': 0.90 },
};


const WEBAPPAPIURL = `${AppConfig.API_URL}/api/v2/`;

export function getModelMAPE(data: any[], modelName: string): number | null {
  if (!data?.length || !modelName) return null;

  const mapeValues: number[] = [];

  data.forEach(row => {
    const actualKey = Object.keys(row).find(key => key.endsWith('_actual'));
    if (!actualKey) return;

    const forecastKey = Object.keys(row).find(
      key => key.endsWith('_yhat') && key.includes(modelName)
    );
    if (!forecastKey) return;

    const actual = Number(row[actualKey]);
    const forecast = Number(row[forecastKey]);

    if (
      typeof actual === 'number' &&
      typeof forecast === 'number' &&
      !isNaN(actual) &&
      !isNaN(forecast) &&
      actual !== 0
    ) {
      const mape = (Math.abs(actual - forecast) / actual) * 100;
      mapeValues.push(mape);
    }
  });

  if (mapeValues.length === 0) return null;

  const averageMAPE = mapeValues.reduce((sum, val) => sum + val, 0) / mapeValues.length;
  return Number(averageMAPE.toFixed(0));
}

export function getModelRMSE(data: any[], modelName: string): number | null {
  if (!data?.length || !modelName) return null;

  const squaredErrors: number[] = [];

  data.forEach(row => {
    const actualKey = Object.keys(row).find(key => key.endsWith('_actual'));
    if (!actualKey) return;

    const forecastKey = Object.keys(row).find(
      key => key.endsWith('_yhat') && key.includes(modelName)
    );
    if (!forecastKey) return;

    const actual = Number(row[actualKey]);
    const forecast = Number(row[forecastKey]);

    if (
      typeof actual === 'number' &&
      typeof forecast === 'number' &&
      !isNaN(actual) &&
      !isNaN(forecast)
    ) {
      const squaredError = Math.pow(forecast - actual, 2);
      squaredErrors.push(squaredError);
    }
  });

  if (squaredErrors.length === 0) return null;

  const mse = squaredErrors.reduce((sum, val) => sum + val, 0) / squaredErrors.length;
  const rmse = Math.sqrt(mse);

  return Number(rmse.toFixed(0));
}

export function getModelRSquared(data: any[], modelName: string): number | null {
  if (!data?.length || !modelName) return null;

  const actualValues: number[] = [];
  const predictedValues: number[] = [];

  data.forEach(row => {
    const actualKey = Object.keys(row).find(key => key.endsWith('_actual'));
    if (!actualKey) return;

    const forecastKey = Object.keys(row).find(
      key => key.endsWith('_yhat') && key.includes(modelName)
    );
    if (!forecastKey) return;

    const actual = Number(row[actualKey]);
    const predicted = Number(row[forecastKey]);

    if (
      typeof actual === 'number' &&
      typeof predicted === 'number' &&
      !isNaN(actual) &&
      !isNaN(predicted)
    ) {
      actualValues.push(actual);
      predictedValues.push(predicted);
    }
  });

  if (actualValues.length === 0) return null;

  const meanActual = actualValues.reduce((sum, val) => sum + val, 0) / actualValues.length;

  const ssr = actualValues.reduce((sum, actual, i) => {
    const pred = predictedValues[i];
    return sum + Math.pow(actual - pred, 2);
  }, 0);

  const sst = actualValues.reduce((sum, actual) => {
    return sum + Math.pow(actual - meanActual, 2);
  }, 0);

  // Handle edge cases
  if (sst === 0) {
    return ssr === 0 ? 1 : 0;
  }

  const r2 = 1 - ssr / sst;
  return Number(r2.toFixed(3));
}



const calculateModelScore = (metrics: any, model: number) => {
  if (!metrics) return 0;

  const mape = !model ? (typeof metrics.MAPE === 'number' ? metrics.MAPE : Number(metrics.MAPE) || 0) : model || 0;
  const rmse = typeof metrics.RMSE === 'number' ? metrics.RMSE : Number(metrics.RMSE) || 0;
  const mae = typeof metrics.MAE === 'number' ? metrics.MAE : Number(metrics.MAE) || 0;
  const r2 = typeof metrics['R²'] === 'number' ? metrics['R²'] : Number(metrics['R²']) || 0;

  const mapeScore = (20 - mape) / 20;
  const rmseScore = (1500000 - rmse) / 1500000;
  const maeScore = (1000000 - mae) / 1000000;
  const r2Score = r2;

  const validScores = [mapeScore, rmseScore, maeScore, r2Score].filter(score => !isNaN(score));
  const score = validScores.reduce((sum, val) => sum + val, 0) / validScores.length;

  return score; // clamp between 0 and 1
};

const ModelRating = ({ score }) => {
  let stars = 1;

  if (score < 5) {
    stars = 5;
  } else if (score < 10) {
    stars = 4;
  } else if (score < 20) {
    stars = 3;
  } else if (score < 50) {
    stars = 2;
  }

  return (
    <div className="flex items-center">
      {Array(5).fill(null).map((_, i) => (
        <svg
          key={i}
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill={i < stars ? 'currentColor' : 'none'}
          stroke="currentColor"
          className="w-4 h-4 text-yellow-400"
        >
          <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
        </svg>
      ))}
    </div>
  );
};

function getRatingStars(mape: number | null) {
  if (mape == null) return "N/A";
  if (mape <= 5) return "⭐⭐⭐⭐⭐";
  if (mape <= 10) return "⭐⭐⭐⭐";
  if (mape <= 15) return "⭐⭐⭐";
  if (mape <= 20) return "⭐⭐";
  return "⭐";
}



const modelAdjustments = {
  'Prophet': 0,
  'ARIMA': -0.1,
  'ETS': -0.125,

  'Weighted Moving Average': -0.2,
  'LightGBM': -0.70,
  'XGBoost': 0.3,
  'SARIMAX': 0.30,
  'CatBoost': -0.3
};

function pivotActualData(raw: any[]): any[] {
  const grouped: Record<string, any> = {};

  raw.forEach((item) => {
    const date = item.date;
    const value = item.value;

    // Use lob_id if present, else fallback to parameter_id
    const metric = item.lob_id?.[1]?.trim() || item.parameter_id?.[1]?.trim();
    if (!metric) return;

    if (!grouped[date]) {
      grouped[date] = { Week: date };
    }

    grouped[date][metric] = (grouped[date][metric] || 0) + value;
  });

  return Object.values(grouped);
}

function pivotForecastData(raw: any[], aggregationType: string = 'Weekly'): any[] {
  const grouped: Record<string, any> = {};

  raw.forEach((item) => {
    const rawDate = item.date;
    const value = item.value;
    const lowerBound = item.lower_bound;
    const upperBound = item.upper_bound;
    const modelId = item.forecast_model_version?.[0];
    const metric = item.lob_id?.[1]?.trim() || item.parameter_id?.[1]?.trim();

    if (!rawDate || !modelId || !metric) return;

    const dateObj = new Date(rawDate);
    if (isNaN(dateObj.getTime())) return;

    let formattedDate = rawDate;
    if (aggregationType === 'Monthly') {
      const month = dateObj.toLocaleString('default', { month: 'long' });
      const year = dateObj.getFullYear();
      formattedDate = `${month} ${year}`;
    }

    // 🔐 Group by both model and time period (monthly or weekly)
    const key = `${formattedDate}-${modelId}`;

    if (!grouped[key]) {
      grouped[key] = {
        Week: formattedDate,
        modelId: modelId,
      };
    }

    // ➕ Aggregate values
    grouped[key][metric] = (grouped[key][metric] || 0) + value;

    if (typeof lowerBound === "number") {
      grouped[key][`${metric}_lower`] = (grouped[key][`${metric}_lower`] || 0) + lowerBound;
    }

    if (typeof upperBound === "number") {
      grouped[key][`${metric}_upper`] = (grouped[key][`${metric}_upper`] || 0) + upperBound;
    }
  });

  return Object.values(grouped);
}




function transformData(data: any[], aggregationType: string): any[] {
  if (aggregationType === 'Monthly') {
    const monthlyData: Record<string, any> = {};

    data.forEach(item => {
      const date = new Date(item.Week);
      if (isNaN(date.getTime())) return;

      const month = date.toLocaleString('default', { month: 'long' });
      const year = date.getFullYear();
      const keyDate = `${month} ${year}`;
      const modelId = item.modelId || ''; // fallback to empty if not present

      const key = `${keyDate}-${modelId}`;

      if (!monthlyData[key]) {
        monthlyData[key] = {
          Week: keyDate,
          modelId: modelId,
        };
      }

      Object.entries(item).forEach(([k, v]) => {
        if (k === 'Week' || k === 'modelId') return;
        monthlyData[key][k] = (monthlyData[key][k] || 0) + (typeof v === 'number' ? v : 0);
      });
    });

    return Object.values(monthlyData);
  }

  // Default Weekly mode — no transformation
  return data;
}




const flattenCombinedData = (
  data,
  forecastData,
  selectedModels,
  selectedMetric,
  availableAlgorithms
) => {
  if (!Array.isArray(data)) {
    console.warn("Invalid data format: expected an array.");
    return [];
  }
  if (!Array.isArray(forecastData)) {
    console.warn("Invalid data format: expected an array.");
    return [];
  }
  if (!Array.isArray(availableAlgorithms)) {
    console.warn("Invalid availableAlgorithms format: expected an array.");
    return [];
  }

  // Step 1: Map selected model names → model version IDs
  const selectedModelVersionMap = new Map(); // "Prophet" → 2

  selectedModels.forEach((modelName) => {
    const match = availableAlgorithms.find(
      (algo) => algo.current_forecast_version && algo.algorithms_id?.[1] === modelName
    );
    if (match?.current_forecast_version?.[0]) {
      selectedModelVersionMap.set(modelName, match.current_forecast_version[0]);
    }
  });


  // Step 2: Build forecast lookup: `${date}-${modelVersionId}`
  const forecastMap = Array.isArray(forecastData)
    ? forecastData.reduce((acc, curr) => {
      const dateKey = getLocalDateString(curr.Week, 'yyyy-MM-dd');
      const modelVersionId = curr.modelId;
      if (dateKey && modelVersionId) {
        acc[`${dateKey}-${modelVersionId}`] = curr;
      }
      return acc;
    }, {})
    : {};


  // Step 3: Combine actuals with forecasts
  return data.map((entry) => {
    const week = getLocalDateString(entry.Week || entry.date, 'yyyy-MM-dd');
    const baseValue = entry[selectedMetric] ?? null;

    const result = {
      Week: week,
      [`${selectedMetric}_actual`]: baseValue,
    };

    selectedModelVersionMap.forEach((modelVersionId, modelName) => {
      const forecast = forecastMap[`${week}-${modelVersionId}`];
      const modelKey = `${selectedMetric}_${modelName}`;

      if (forecast && forecast[selectedMetric] !== undefined) {
        result[`${modelKey}_yhat`] = forecast[selectedMetric];
        result[`${modelKey}_yhat_lower`] = forecast[`${selectedMetric}_lower`] ?? 0;
        result[`${modelKey}_yhat_upper`] = forecast[`${selectedMetric}_upper`] ?? 0;
      }
    });

    return result;
  });
};


const exportToCSV = (data, filename) => {
  if (!data || !data.length) return;

  const headers = Object.keys(data[0]);

  // Step 1: Identify all _actual keys
  const actualKeys = headers.filter(key => key.endsWith('_actual'));

  // Step 2: Build MAPE pairs: match each actual to all *_yhat forecast keys with matching base
  const mapePairs = [];

  actualKeys.forEach(actualKey => {
    const base = actualKey.replace(/_actual$/, '');

    headers.forEach(header => {
      if (header.endsWith('_yhat') && header.includes(base)) {
        mapePairs.push({ base: header.replace(/_yhat$/, ''), actualKey, forecastKey: header });
      }
    });
  });

  // Step 3: Build allHeaders in correct order (interleave _MAPE after matching _yhat)
  let allHeaders = [];

  headers.forEach((key) => {
    allHeaders.push(key);

    // Check if this is a `_yhat_upper` key
    if (key.endsWith('_yhat_upper')) {
      const base = key.replace(/_yhat_upper$/, '');

      // Find the corresponding MAPE pair based on forecastKey
      const matchedMAPE = mapePairs.find(pair => pair.base === base);
      if (matchedMAPE) {
        allHeaders.push(`${base}_MAPE`);
      }
    }
  });


  const csvRows = [];

  // Step 4: Header row formatting
  csvRows.push(
    allHeaders.map((key) => {
      if (key === 'Week') return 'Date';
      if (key.endsWith('_actual')) return key.replace('_actual', ' Actual');
      if (key.endsWith('_yhat')) return key.replace('_yhat', ' Forecast');
      if (key.endsWith('_yhat_lower')) return key.replace('_yhat_lower', ' Forecast Lower Bound');
      if (key.endsWith('_yhat_upper')) return key.replace('_yhat_upper', ' Forecast Upper Bound');
      if (key.endsWith('_MAPE')) return key.replace('_MAPE', ' MAPE %');
      return key;
    }).join(',')
  );

  // Step 5: Data rows
  data.forEach((row) => {
    const rowValues = [];

    allHeaders.forEach((key) => {
      if (key.endsWith('_MAPE')) {
        // Find corresponding MAPE pair
        const pair = mapePairs.find(p => `${p.forecastKey.replace(/_yhat$/, '')}_MAPE` === key);
        if (pair) {
          const actual = row[pair.actualKey];
          const forecast = row[pair.forecastKey];
          if (
            typeof actual === 'number' &&
            typeof forecast === 'number' &&
            actual !== 0
          ) {
            rowValues.push(`${((Math.abs(actual - forecast) / actual) * 100).toFixed(0)} %`);
            return;
          }
        }
        rowValues.push('');
      } else if (key === 'Week') {
        const d = new Date(getLocalDateString(row[key]));
        rowValues.push(isNaN(d.getTime()) ? '' : d.toLocaleDateString());
      } else {
        const val = row[key];
        if (typeof val === 'number') {
          rowValues.push(val.toFixed(0));
        } else {
          rowValues.push(val ? `"${String(val).replace(/"/g, '""')}"` : '');
        }
      }
    });

    csvRows.push(rowValues.join(','));
  });

  // Step 6: Export CSV
  const csvContent = csvRows.join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};


const exportToXLSX = async (data, filename = 'Forecast.xlsx') => {
  if (!data || !data.length) return;

  const headers = Object.keys(data[0]);

  // Step 1: Identify all _actual keys
  const actualKeys = headers.filter(key => key.endsWith('_actual'));

  // Step 2: Build MAPE pairs
  const mapePairs = [];

  actualKeys.forEach(actualKey => {
    const base = actualKey.replace(/_actual$/, '');
    headers.forEach(header => {
      if (header.endsWith('_yhat') && header.includes(base)) {
        mapePairs.push({ base: header.replace(/_yhat$/, ''), actualKey, forecastKey: header });
      }
    });
  });

  // Step 3: Create ordered allHeaders with MAPE
  let allHeaders = [];

  headers.forEach((key) => {
    allHeaders.push(key);
    if (key.endsWith('_yhat_upper')) {
      const base = key.replace(/_yhat_upper$/, '');
      const matchedMAPE = mapePairs.find(pair => pair.base === base);
      if (matchedMAPE) {
        allHeaders.push(`${base}_MAPE`);
      }
    }
  });

  // Step 4: Create workbook and worksheet
  const workbook = new ExcelJS.Workbook();
  const worksheet = workbook.addWorksheet('Forecast');

  // Step 5: Create header row
  const displayHeader = allHeaders.map((key) => {
    if (key === 'Week') return 'Date';
    if (key.endsWith('_actual')) return key.replace('_actual', ' Actual');
    if (key.endsWith('_yhat')) return key.replace('_yhat', ' Forecast');
    if (key.endsWith('_yhat_lower')) return key.replace('_yhat_lower', ' Forecast Lower Bound');
    if (key.endsWith('_yhat_upper')) return key.replace('_yhat_upper', ' Forecast Upper Bound');
    if (key.endsWith('_MAPE')) return key.replace('_MAPE', ' MAPE %');
    return key;
  });

  worksheet.addRow(displayHeader);
  worksheet.getRow(1).font = { bold: true };

  // Step 6: Add data rows
  data.forEach((row) => {
    const rowValues = [];

    allHeaders.forEach((key) => {
      if (key.endsWith('_MAPE')) {
        const pair = mapePairs.find(p => `${p.forecastKey.replace(/_yhat$/, '')}_MAPE` === key);
        if (pair) {
          const actual = row[pair.actualKey];
          const forecast = row[pair.forecastKey];
          if (typeof actual === 'number' && typeof forecast === 'number' && actual !== 0) {
            const mape = ((Math.abs(actual - forecast) / actual) * 100).toFixed(0);
            rowValues.push(Number(mape));
            return;
          }
        }
        rowValues.push(null);
      } else if (key === 'Week') {
        const date = new Date(getLocalDateString(row[key]));
        rowValues.push(isNaN(date.getTime()) ? null : date);
      } else {
        const val = row[key];
        if (typeof val === 'number') {
          rowValues.push(val);
        } else {
          rowValues.push(val ?? null);
        }
      }
    });

    worksheet.addRow(rowValues);
  });

  // Step 7: Column formatting
  worksheet.columns.forEach((col, index) => {
    const key = allHeaders[index];

    if (key === 'Week') {
      col.numFmt = 'yyyy-mm-dd';
      col.width = 15;
    } else if (typeof data[0][key] === 'number' || key.endsWith('_MAPE')) {
      col.numFmt = '0'; // Integer format
      col.width = 18;
    } else {
      col.width = 25;
    }
  });

  // Step 8: Export
  const buffer = await workbook.xlsx.writeBuffer();
  const blob = new Blob([buffer], {
    type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  });

  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.endsWith('.xlsx') ? filename : `${filename}.xlsx`;
  a.click();
  URL.revokeObjectURL(url);
};



/* const allModels = [
  { label: "Prophet", value: "Prophet" },
  { label: "ARIMA", value: "ARIMA" },
  { label: "SARIMAX", value: "SARIMAX" },
    { label: "Weighted Moving Average", value: "Weighted Moving Average" },
      { label: "ETS", value: "ETS" },
  { label: "CatBoost", value: "CatBoost" },
  { label: "XGBoost", value: "XGBoost" },
  { label: "LightGBM", value: "LightGBM" },
  { label: "ARIMA + LightGBM", value: "ARIMA+LightGBM" },
  { label: "Prophet + XGBoost", value: "Prophet+XGBoost" },
  { label: "SARIMAX + CatBoost", value: "SARIMAX+CatBoost" },
]; */

/* const modelColors = {
  "Prophet": "#1f77b4",             // Blue
  "ARIMA": "#2ca02c",               // Green
  "SARIMAX": "#ff7f0e",             // Orange
  "Weighted Moving Average": "#9467bd", // Purple
  "ETS": "#8c564b",                 // Brown
  "CatBoost": "#d62728",            // Red
  "XGBoost": "#17becf",             // Teal
  "LightGBM": "#bcbd22",            // Olive
  "ARIMA+LightGBM": "#7f7f7f",      // Gray
  "Prophet+XGBoost": "#e377c2",     // Pink
  "SARIMAX+CatBoost": "#aec7e8"     // Light Blue
}; */

function getISOWeekNumber(date: Date): number {
  const tempDate = new Date(date.getTime());
  tempDate.setHours(0, 0, 0, 0);
  tempDate.setDate(tempDate.getDate() + 4 - (tempDate.getDay() || 7));
  const yearStart = new Date(tempDate.getFullYear(), 0, 1);
  const weekNo = Math.ceil((((tempDate.getTime() - yearStart.getTime()) / 86400000) + 1) / 7);
  return weekNo;
}

export const ModelValidationTab = ({ loading, forecastVersions, selectedRange, versionsLoading, foreCastLoading, defaultDecimalPrecisions, userData, businessData, trigger, setTriger, algorithmModels, algorithmModelsData, lobData, foreCastdata, setParentSelectedModels, parentSelectedMetric, parentSelectedModels, setParentSelectedMetric, metricData, children, aggregationType = "Weekly", data, modelType, plotAggregationType: propPlotAggregationType, setPlotAggregationType: propSetPlotAggregationType }) => {

  const [showForecastSettings, setShowForecastSettings] = useState(false);
  const [metrics, setMetrics] = useState([]);
  const [localPlotAggregationType, setLocalPlotAggregationType] = useState('Weekly');
  const plotAggregationType = propPlotAggregationType ?? localPlotAggregationType;
  const setPlotAggregationType = propSetPlotAggregationType ?? setLocalPlotAggregationType;
  const [hoveredLegend, setHoveredLegend] = useState<string | null>(null);
  const [chartType, setChartType] = useState('line');

  const [isModelOpen, setModelOpen] = useState(false);
  const [currentModelForeCast, setCurrentModelForeCast] = useState({});
  const [updateLoading, setUpdateLoading] = useState(false);
  const [modelMetrics, setModelMetrics] = useState(modelMetricsStatic);
  const [aiModal, showAIModal] = useState(false);

  const formatNumber = (value: number) => {
    if (value >= 1000000) return defaultDecimalPrecisions > 0 ? `${(value / 1000000).toFixed(defaultDecimalPrecisions)}M` : `${Math.round(value / 1_000_000)}M`;
    if (value >= 1000) return defaultDecimalPrecisions > 0 ? `${(value / 1000).toFixed(1)}K` : `${Math.round(value / 1_000)}K`;
    return value.toLocaleString();
  };

  const selectedMetric = parentSelectedMetric;

  const selectedModels = useMemo(
    () => parentSelectedModels.filter((m) => m !== ''),
    [parentSelectedModels]
  );
  const allModels = [...algorithmModels];

  useEffect(() => {
    if (selectedModels?.length && forecastVersions?.length) {
      const newMetrics: Record<string, any> = {};

      selectedModels.forEach((modelName) => {
        const metrics = forecastVersions.find((item) => {
          const extractedModel = extractModelNameFromItemName(item.name);
          return extractedModel === modelName;
        });

        if (!metrics) return;

        newMetrics[modelName] = {
          MAPE: metrics.mape,
          RMSE: metrics.rmse,
          MAE: metrics.mse ?? 'N/A',
          'R²': metrics.r_squared,
        };
      });
      setModelMetrics(newMetrics);
    }
  }, [selectedModels, forecastVersions]);

  const groupedModels = allModels.reduce((acc: Record<string, any[]>, model: any) => {
    const category = model.category || "Uncategorized";
    acc[category] = acc[category] || [];
    acc[category].push(model);
    return acc;
  }, {});

  const fallbackColors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#17a768", "#e6550d", "#6b6ecf", "#b5cf6b", "#9c9ede"
  ];

  const assignedColors: Record<string, string> = {};
  let colorIndex = 0;

  algorithmModels.forEach((model) => {
    const name = model.name?.trim();
    if (name && !assignedColors[name]) {
      assignedColors[name] = fallbackColors[colorIndex % fallbackColors.length];
      colorIndex++;
    }
  });

  const modelColors = assignedColors;



  const { toast } = useToast();

  const handleModelToggle = (modelValue: string) => {
    setParentSelectedModels((current) =>
      current.includes(modelValue)
        ? current.filter((v) => v !== modelValue)
        : [...current, modelValue]
    );
  };

  const onModelForecastChange = (model: any) => {
    setCurrentModelForeCast(model);
    setModelOpen(true)
  };

  type LobItem = {
    id: number;
    preferred_algorithm?: [number, string]; // [id, name]
    [key: string]: any;
  };

  type SelectedItem = {
    lob_id: number;
    [key: string]: any;
  };

  type DefaultAlgorithm = {
    id: number;
    name: string;
  } | null;

  function getDefaultAlgorithmByLob(
    lobList: LobItem[],
    selected: SelectedItem
  ): DefaultAlgorithm {
    // Case 1: Match by lob_id
    if (selected?.lob_id && selected?.source === 'lob') {
      const match = lobList.find((lob) => lob.id === selected.lob_id);

      if (
        match &&
        Array.isArray(match.preferred_algorithm) &&
        match.preferred_algorithm.length === 2
      ) {
        return {
          id: match.preferred_algorithm[0],
          name: match.preferred_algorithm[1],
        };
      }
    }

    // Case 2: Fallback to parameter_id and use businessData
    if (
      selected?.parameter_id &&
      selected?.source === 'parameter' &&
      businessData &&
      Array.isArray(businessData.preferred_algorithm) &&
      businessData.preferred_algorithm.length === 2
    ) {
      return {
        id: businessData.preferred_algorithm[0],
        name: businessData.preferred_algorithm[1],
      };
    }

    // Case 3: No match found
    return null;
  }

  // Initial model selection - runs only once when component mounts
  useEffect(() => {
    const defaultModel = getDefaultAlgorithmByLob(lobData, metrics.find((m) => m.id === selectedMetric) || {})?.name || '';
    setParentSelectedModels([defaultModel])
  }, [lobData, businessData, selectedMetric, metrics]);

  const defaultColorPalette = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#1F77B4", "#AEC7E8", "#FF7F0E", "#FFBB78", "#2CA02C",
    "#98DF8A", "#D62728", "#FF9896", "#9467BD", "#C5B0D5",
    "#8C564B", "#C49C94", "#E377C2", "#F7B6D2", "#7F7F7F",
    "#C7C7C7", "#BCBD22", "#DBDB8D", "#17BECF", "#9EDAE5",
    "#393B79", "#5254A3", "#6B6ECF", "#9C9EDE", "#637939",
    "#8CA252", "#B5CF6B", "#CEDB9C", "#8C6D31", "#BD9E39",
    "#E7BA52", "#E7CB94", "#843C39", "#AD494A", "#D6616B",
    "#E7969C", "#7B4173", "#A55194", "#CE6DBD", "#DE9ED6"
  ];

  const getColorByIndex = (index: number) =>
    defaultColorPalette[index];

  useMemo(() => {
    const newData = metricData.map((cat, index) => ({
      id: cat.title,
      name: cat.title,
      color: getColorByIndex(index),
      source: cat.source,
      parameter_id: cat?.parameterId || false,
      lob_id: cat?.lobId || false,
    }));
    setMetrics(newData)
    if (!trigger || !parentSelectedMetric) {
      setParentSelectedMetric(newData?.length && newData[0].name ? newData[0].name : '')
    }
  }, [metricData, trigger]);

  const adjustment = selectedModels.length === 1 ? modelAdjustments[selectedModels[0]] || 0 : 0;

  const combinedData = useMemo(() => {
    const rawCombinedData = pivotActualData(data || []);
    const rawForecastData = pivotForecastData(foreCastdata || [], plotAggregationType);

    const rawTranformData = transformData(rawCombinedData, plotAggregationType);
    const rawTranformForecastData = transformData(rawForecastData, plotAggregationType);

    if (!rawTranformData || typeof rawTranformData !== "object") {
      console.warn("Invalid combined format: Expected an object, but got:", rawTranformData);
      return [];
    }
    const flattenedData = flattenCombinedData(rawTranformData, rawTranformForecastData, selectedModels, selectedMetric, algorithmModelsData);

    return flattenedData;
  }, [data, foreCastdata, selectedModels, selectedMetric, algorithmModelsData, plotAggregationType]);

  const filteredData = useMemo(() => {

    if (!combinedData?.length) return [];
    //const oneYearAgo = new Date();
    // oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);
    const filtered = combinedData;

    return filtered;
  }, [combinedData]);

  const processedData = useMemo(() => {
    return filteredData.map(d => {
      const extended: any = { ...d };

      selectedModels.forEach(model => {
        const lower = d[`${selectedMetric}_${model}_yhat_lower`];
        const upper = d[`${selectedMetric}_${model}_yhat_upper`];

        if (lower !== undefined && upper !== undefined) {
          extended[`${selectedMetric}_${model}_yhat_range`] = upper - lower; // bar height
          extended[`${selectedMetric}_${model}_yhat_base`] = lower;          // bar base
        }
      });

      return extended;
    });
  }, [filteredData]);


  async function updateForm(id: number, model: string, values: Record<string, unknown>) {
    const formData = new FormData();

    formData.append("ids", `[${id}]`);
    formData.append("model", model);
    formData.append("values", JSON.stringify(values)); // nested object as JSON string

    const url = `${WEBAPPAPIURL}write`;
    setUpdateLoading(true);
    try {
      const response = await axios.put(url, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      setUpdateLoading(false);
      setModelOpen(false);
      setTriger(Math.random())
      toast({
        title: "Settings Applied",
        description: "Forecast settings have been updated successfully.",
      });
      return response.data;
    } catch (e) {
      setUpdateLoading(false);
      setModelOpen(false)
    }
  }

  const onSaveModelChanges = async () => {
    if (!currentModelForeCast?.id) return;
    const currentMetricForecast = metrics.find(m => m.id === selectedMetric);
    if (currentMetricForecast && currentMetricForecast?.lob_id) {
      const response = await updateForm(Number(currentMetricForecast?.lob_id), 'line_business_lob', {
        preferred_algorithm: currentModelForeCast?.id,
      });
    } else if (currentMetricForecast && currentMetricForecast?.parameter_id && businessData && businessData?.id) {
      const response = await updateForm(Number(businessData?.id), 'business.unit', {
        preferred_algorithm: currentModelForeCast?.id,
      });
    }
  };

  const [sortKey, setSortKey] = useState<"Model" | "MAPE" | "RMSE" | "MAE" | "R²">("MAPE");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");
  const [isClear, setClear] = useState(false);


  const onClearHistory = () => {
    localStorage.setItem("ai_messages", JSON.stringify([]));
    setClear(Math.random());
  };

  const handleSort = (key: typeof sortKey) => {
    if (key === sortKey) {
      // Toggle order if same key is clicked
      setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortOrder("asc");
    }
  };

  const sortedModels = useMemo(() => {
    return [...selectedModels].sort((a, b) => {
      let aVal: string | number | null = null;
      let bVal: string | number | null = null;

      if (sortKey === "Model") {
        aVal = a.toLowerCase();
        bVal = b.toLowerCase();
      } else if (sortKey === "MAPE") {
        aVal = getModelMAPE(filteredData, a);
        bVal = getModelMAPE(filteredData, b);
      } else if (sortKey === "RMSE") {
        aVal = getModelRMSE(filteredData, a);
        bVal = getModelRMSE(filteredData, b);
      } else if (sortKey === "R²") {
        aVal = getModelRSquared(filteredData, a);
        bVal = getModelRSquared(filteredData, b);
      } else {
        aVal = modelMetrics[a]?.[sortKey] ?? null;
        bVal = modelMetrics[b]?.[sortKey] ?? null;
      }

      if (aVal == null) return 1;
      if (bVal == null) return -1;

      if (typeof aVal === "string" && typeof bVal === "string") {
        return sortOrder === "asc" ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }

      return sortOrder === "asc" ? (aVal as number) - (bVal as number) : (bVal as number) - (aVal as number);
    });
  }, [selectedModels, modelMetrics, sortKey, sortOrder, filteredData]);



  const actualKey = `${selectedMetric}_actual`;
  const yhatKey = `${selectedMetric}_yhat`;
  const lowerKey = `${selectedMetric}_yhat_lower`;
  const upperKey = `${selectedMetric}_yhat_upper`;

  const metricValues = data.metrics?.[selectedMetric] || {};

  const jsonTableData = useMemo(() => {
    return sortedModels.map((model) => {
      const mape = getModelMAPE(filteredData, model);
      const rmse = getModelRMSE(filteredData, model);
      const r2 = getModelRSquared(filteredData, model);

      return {
        Model: model,
        MAPE: mape != null ? `${mape}%` : "N/A",
        RMSE: rmse != null ? formatWithPrecision(rmse, defaultDecimalPrecisions) : "N/A",
        "R²": r2 != null ? formatWithPrecision(r2, defaultDecimalPrecisions) : "N/A",
        Rating: getRatingStars(mape),
      };
    });
  }, [sortedModels, filteredData]);


  const cards = [
    {
      title: "Top MAPE Values",
      description: "Show me the top 3 MAPE values for all LOBs.",
    },
    {
      title: "MAPE Insights",
      description: "What is the least MAPE value for Orders?",
    },
    {
      title: "Model Comparison",
      description: "Compare Prophet vs SARIMAX and show me the MAPE% for Orders this year.",
    }
  ]

  return (
    <TooltipProvider>
      <div className="flex flex-row gap-4 w-full items-start">
        {!showForecastSettings && (
          <Card className="flex-1  dark:border-gray-700 relative overflow-visible">
            <div className="relative pt-2 mt-[-10px] ">
              <h3 className="text-xl font-semibold">Model Validation</h3>
              <CardContent className="p-6">
                <div className="absolute top-2 right-2 z-10 flex items-center space-x-2">
                  {children}
                  <div className="gap-2 border rounded-md p-1 ">
                    <Button
                      variant={plotAggregationType === "Weekly" ? "default" : "ghost"}
                      size="sm"
                      onClick={() => setPlotAggregationType("Weekly")}
                      disabled={!(data?.length > 0)}
                      className="h-7 px-3"
                    >
                      Weekly
                    </Button>
                    <Button
                      variant={plotAggregationType === "Monthly" ? "default" : "ghost"}
                      size="sm"
                      disabled={!(data?.length > 0)}
                      onClick={() => setPlotAggregationType("Monthly")}
                      className="h-7 px-3"
                    >
                      Month
                    </Button>
                  </div>
                  <Select disabled={!(data?.length > 0)} value={selectedMetric} onValueChange={setParentSelectedMetric}>
                    <SelectTrigger className="w-[180px]" aria-label="Select metric">
                      <SelectValue placeholder="Select metric" />
                    </SelectTrigger>
                    <SelectContent>
                      {metrics.map(metric => (
                        <SelectItem key={metric.id} value={metric.id}>
                          {metric.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button disabled={!(data?.length > 0)} variant="outline" className="w-[180px] whitespace-nowrap justify-between">
                        {selectedModels.length === 1
                          ? allModels.find(m => m.name === selectedModels[0])?.name
                          : `${selectedModels.length} models selected`}
                        <ChevronDown className="ml-2 h-4 w-4 opacity-50" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent className="w-[350px]" onCloseAutoFocus={(e) => e.preventDefault()} style={{ maxHeight: "450px", overflowY: "auto" }}>
                      <DropdownMenuLabel>Select Models</DropdownMenuLabel>
                      <DropdownMenuSeparator />
                      <DropdownMenuCheckboxItem
                        key="select-all"
                        disabled={!(data?.length > 0) || foreCastLoading}
                        checked={selectedModels.length === allModels.length}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            setParentSelectedModels(allModels.map(m => m.name));
                          } else {
                            setParentSelectedModels([]);
                          }
                        }}
                        onSelect={(e) => e.preventDefault()}
                      >
                        {foreCastLoading ? 'Loading...' : 'Select All'}
                      </DropdownMenuCheckboxItem>
                      <DropdownMenuSeparator />
                      {Object.entries(groupedModels).map(([category, models]) => (
                        <div key={category}>
                          <DropdownMenuLabel className="text-sm font-semibold opacity-70">{category}</DropdownMenuLabel>
                          <DropdownMenuSeparator />

                          {models.map((model: any) => {
                            const isDefault =
                              getDefaultAlgorithmByLob(lobData, metrics.find((m) => m.id === selectedMetric) || {})?.id === Number(model.id);

                            return (
                              <DropdownMenuCheckboxItem
                                key={model.id}
                                checked={selectedModels.includes(model.name)}
                                onCheckedChange={() => handleModelToggle(model.name)}
                                onSelect={(e) => e.preventDefault()}
                                className="group"
                                disabled={foreCastLoading}
                              >
                                <div className="flex justify-between items-center w-full">
                                  {/* Left: Name + Default Chip */}
                                  <div className="text-left pr-2 whitespace-normal break-words max-w-[200px]">
                                    {model.name}
                                    {isDefault && (
                                      <span className="ml-1 text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
                                        Default
                                      </span>
                                    )}
                                  </div>

                                  {/* Right: Icons */}
                                  <div className="flex space-x-1 items-center">
                                    <TooltipProvider>
                                      <UITooltip>
                                        <TooltipTrigger asChild>
                                          <div className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-muted cursor-pointer">
                                            <Info className="w-4 h-4 text-muted-foreground hover:text-primary" />
                                          </div>
                                        </TooltipTrigger>
                                        <TooltipContent side="top" className="text-md max-w-[300px]">
                                          <p className="font-md">{model.description || "No description available."}</p>
                                        </TooltipContent>
                                      </UITooltip>
                                    </TooltipProvider>
                                    {!isDefault && (
                                      <UITooltip>
                                        <TooltipTrigger asChild>
                                          <div
                                            className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-muted cursor-pointer"
                                            onClick={(e) => {
                                              e.stopPropagation();
                                              onModelForecastChange(model);
                                            }}
                                          >

                                            <Settings fontSize="small" className="w-4 h-4 text-muted-foreground hover:text-primary" />
                                          </div>
                                        </TooltipTrigger>
                                        <TooltipContent>
                                          <p>Set as Default</p>
                                        </TooltipContent>
                                      </UITooltip>
                                    )}
                                  </div>
                                </div>
                              </DropdownMenuCheckboxItem>
                            );
                          })}
                        </div>
                      ))}

                    </DropdownMenuContent>
                  </DropdownMenu>

                  <Sheet open={isModelOpen} onOpenChange={setModelOpen}>
                    <SheetContent
                      side="right"
                      className="w-[1000px] h-screen bg-card text-card-foreground shadow-lg border border-border overflow-y-auto fixed top-0 right-0 z-[1000]"
                    >
                      <SheetHeader>
                        <SheetTitle>Forecast Settings</SheetTitle>
                      </SheetHeader>
                      {/* You may want to add loading state/logic here if needed */}
                      <div className="space-y-4 mt-3">
                        {/* Forecast Period */}
                        <div className="p-4 space-y-4">
                          <div>
                            <h4 className="text-sm font-semibold text-muted-foreground">Selected Metric</h4>
                            <p className="text-base pl-1 font-medium">{metrics.find(m => m.id === selectedMetric)?.name || '—'}</p>
                          </div>

                          <div>
                            <h4 className="text-sm font-semibold text-muted-foreground">Your Preferred Model</h4>
                            <p className="text-base pl-1 font-medium inline-flex items-center">
                              {getDefaultAlgorithmByLob(lobData, metrics.find((m) => m.id === selectedMetric) || {})?.name || '-'}
                              <TooltipProvider>
                                <UITooltip>
                                  <TooltipTrigger asChild>
                                    <span className="ml-1 cursor-pointer">
                                      <Info className="w-4 h-4 text-primary mr-2" />
                                    </span>
                                  </TooltipTrigger>
                                  <TooltipContent side="top" className="text-md max-w-[300px]">
                                    <p className="font-md">
                                      {allModels.find(m => m.id === getDefaultAlgorithmByLob(lobData, metrics.find((m) => m.id === selectedMetric) || {})?.id)?.description}
                                    </p>
                                  </TooltipContent>
                                </UITooltip>
                              </TooltipProvider>
                            </p>
                          </div>

                          <div className="pt-2 border-t border-border">
                            <p className="text-sm">
                              Would you like to save
                              <span className="font-semibold text-primary inline-flex items-center">
                                {`"${currentModelForeCast?.name}"`}
                                {currentModelForeCast?.description && (
                                  <TooltipProvider>
                                    <UITooltip>
                                      <TooltipTrigger asChild>
                                        <span className="ml-1 cursor-pointer">
                                          <Info className="w-4 h-4 text-primary mr-2" />
                                        </span>
                                      </TooltipTrigger>
                                      <TooltipContent side="top" className="text-md max-w-[300px]">
                                        <p className="font-md">
                                          {currentModelForeCast?.description}
                                        </p>
                                      </TooltipContent>
                                    </UITooltip>
                                  </TooltipProvider>
                                )}
                              </span>
                              as your preferred model for forecasting?
                            </p>
                          </div>
                        </div>

                        {/* Apply Changes Button */}
                        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                          <Button
                            size="sm"
                            className="w-full"
                            disabled={updateLoading}
                            onClick={() => onSaveModelChanges()}
                          >
                            {updateLoading ? 'Saving...' : 'Save Changes'}
                          </Button>
                        </div>
                      </div>
                    </SheetContent>
                  </Sheet>

                  <AskAIButton mainData={data || []} aiParams={{ graph: filteredData, table: jsonTableData }} parametersList={[]} />
                  <TooltipProvider>
                    <UITooltip>
                      <TooltipTrigger asChild>
                        <div>
                          <Popover>
                            <PopoverTrigger asChild>
                              <Button disabled={!(data?.length > 0)} variant="outline" size="icon">
                                <AreaChartOutlinedIcon className="h-5 w-5 text-muted-foreground hover:text-foreground" />
                              </Button>
                            </PopoverTrigger>
                            <PopoverContent className="w-[200px] p-0" align="end">
                              <div className="p-2">
                                <p className="text-sm font-medium">Select Chart</p>
                              </div>
                              <Separator />
                              <div className="p-1">
                                <button
                                  onClick={() => setChartType('area')}
                                  disabled={chartType === 'area' || !(data?.length > 0)}
                                  className="w-full text-left p-2 text-sm rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
                                >
                                  Area Chart
                                </button>
                                <button
                                  onClick={() => setChartType('line')}
                                  disabled={chartType === 'line' || !(data?.length > 0)}
                                  className="w-full text-left p-2 text-sm rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
                                >
                                  Line Chart
                                </button>
                                <button
                                  onClick={() => setChartType('bar')}
                                  disabled={chartType === 'bar' || !(data?.length > 0)}
                                  className="w-full text-left p-2 text-sm rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
                                >
                                  Bar Chart
                                </button>
                              </div>
                            </PopoverContent>
                          </Popover>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Select Chart</p>
                      </TooltipContent>
                    </UITooltip>
                  </TooltipProvider>

                  <TooltipProvider>
                    <UITooltip>
                      <TooltipTrigger asChild>
                        <div>
                          <Popover>
                            <PopoverTrigger asChild>
                              <Button disabled={!(data?.length > 0)} variant="outline" size="icon">
                                <Download className="h-4 w-4 text-muted-foreground hover:text-foreground" />
                              </Button>
                            </PopoverTrigger>
                            <PopoverContent className="w-[200px] p-0" align="end">
                              <div className="p-2">
                                <p className="text-sm font-medium">Select Export Type</p>
                              </div>
                              <Separator />
                              <div className="p-1">
                                <button
                                  onClick={() => exportToCSV(filteredData, `Model_Validation_${aggregationType.toLowerCase()}_${new Date().toISOString().split("T")[0]}.csv`)}
                                  disabled={!(data?.length > 0)}
                                  className="w-full text-left p-2 text-sm rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
                                >
                                  CSV
                                </button>
                                <button
                                  onClick={() => exportToXLSX(filteredData, `Model_Validation_${aggregationType.toLowerCase()}_${new Date().toISOString().split("T")[0]}.xlsx`)}
                                  disabled={!(data?.length > 0)}
                                  className="w-full text-left p-2 text-sm rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
                                >
                                  XLSX
                                </button>
                              </div>
                            </PopoverContent>
                          </Popover>
                        </div>
                      </TooltipTrigger>
                    </UITooltip>
                  </TooltipProvider>

                </div>
                {!loading && !data?.length && (
                  <NoDataFullPage message="No data found." />
                )}
                {data?.length > 0 && (
                  <div className="h-[400px]">
                    {chartType === "line" && (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={filteredData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            dataKey="Week"
                            tickFormatter={(tick) => {
                              const date = new Date(tick);
                              if (isNaN(date.getTime())) return tick;
                              let label = '';
                              if (plotAggregationType === 'Weekly') {
                                const weekNumber = getISOWeekNumber(date);
                                label = `WK${weekNumber.toString().padStart(2, '0')}`;
                              } else {
                                const month = date.toLocaleString('default', { month: 'short' });
                                const year = date.getFullYear();
                                label = `${month}\n${year}`;
                              }
                              return label;
                            }}
                          />
                          <YAxis tickFormatter={formatNumber} />
                          <RechartsTooltip
                            content={({ payload, label }) => {
                              if (!payload || payload.length === 0) return null;
                              const date = new Date(label);
                              const week = getISOWeekNumber(date);
                              const formattedDate = isNaN(date.getTime())
                                ? label
                                : `${date.getDate().toString().padStart(2, '0')}-${date.toLocaleString('en-US', { month: 'short' })}-${date.getFullYear()} (W${week})`;

                              const actual = payload.find(p => p.name === 'Actual');
                              const modelPredictions = selectedModels.map(model => ({
                                model,
                                value: payload.find(p => p.name === model)?.value,
                                lower: payload[0]?.payload?.[`${selectedMetric}_${model}_yhat_lower`],
                                upper: payload[0]?.payload?.[`${selectedMetric}_${model}_yhat_upper`]
                              }));

                              return (
                                <div className="p-2 rounded-lg border text-sm bg-white text-gray-900 border-gray-300 dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">
                                  <div className="mb-2 font-semibold">{formattedDate}</div>

                                  {actual && (
                                    <div className="flex justify-between text-[13px] font-medium mb-1">
                                      <span className="flex items-center gap-1.5">
                                        <span className="w-[10px] h-[10px] rounded-sm" style={{ backgroundColor: '#8884d8' }}></span>
                                        <span className="text-gray-600 dark:text-gray-300">Actual:</span>
                                      </span>
                                      <span className="text-gray-700 dark:text-gray-200">
                                        {typeof actual.value === 'number' ? formatNumber(actual.value) : actual.value ?? 'N/A'}
                                      </span>
                                    </div>
                                  )}

                                  {modelPredictions.map(({ model, value, lower, upper }) => (
                                    <div key={model} className="mb-1">
                                      <div className="flex justify-between text-[13px] font-medium">
                                        <span className="flex items-center gap-1.5">
                                          <span
                                            className="w-[10px] h-[10px] rounded-sm"
                                            style={{ backgroundColor: modelColors[model] || '#000' }}
                                          ></span>
                                          <span className="text-gray-600 dark:text-gray-300">{model}:</span>
                                        </span>
                                        <span className="text-gray-700 dark:text-gray-200">
                                          {typeof value === 'number' ? formatNumber(value) : value ?? 'N/A'}
                                        </span>
                                      </div>
                                      {lower !== undefined && upper !== undefined && (
                                        <div className="text-[11px] text-gray-500 dark:text-gray-400 ml-4 mt-0.5">
                                          Confidence bounds: {formatNumber(lower)} – {formatNumber(upper)}
                                        </div>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              );
                            }}
                          />

                          <Legend
                            onMouseEnter={(e: any) => setHoveredLegend(e?.value)}
                            onMouseLeave={() => setHoveredLegend(null)}
                            formatter={(value: string) => {
                              const isActive = hoveredLegend === null || hoveredLegend === value;
                              return (
                                <span
                                  style={{
                                    opacity: isActive ? 1 : 0.4,
                                    fontWeight: isActive ? 600 : 400,
                                    color: '#94a3b8',
                                    cursor: 'pointer',
                                  }}
                                >
                                  {value}
                                </span>
                              );
                            }}
                            payload={[
                              ...selectedModels.map(model => ({
                                value: model,
                                type: 'line' as const,
                                color: modelColors[model] || '#000',
                              })),
                              {
                                value: 'Actual',
                                type: 'line' as const,
                                color: '#8884d8',
                              }
                            ]}
                          />
                          <Line
                            type="monotone"
                            dataKey={actualKey}
                            name="Actual"
                            stroke="#8884d8"
                            strokeWidth={2}
                            dot={{ r: 3 }}
                            activeDot={{ r: 5 }}
                            opacity={hoveredLegend === null || hoveredLegend === 'Actual' ? 1 : 0.2}
                          />
                          {selectedModels.map(model => {
                            const isActive = hoveredLegend === null || hoveredLegend === model;
                            const strokeColor = modelColors[model] || '#888';

                            return (
                              <React.Fragment key={model}>
                                <Line
                                  type="monotone"
                                  dataKey={`${selectedMetric}_${model}_yhat`}
                                  name={model}
                                  stroke={strokeColor}
                                  strokeWidth={2}
                                  dot={{ r: 3 }}
                                  opacity={isActive ? 1 : 0.2}
                                />
                                <Line
                                  type="monotone"
                                  dataKey={`${selectedMetric}_${model}_yhat_lower`}
                                  name={`${model} Lower Bound`}
                                  stroke={strokeColor}
                                  strokeWidth={1}
                                  strokeDasharray="3 3"
                                  dot={false}
                                  opacity={isActive ? 0.5 : 0.1}
                                />
                                <Line
                                  type="monotone"
                                  dataKey={`${selectedMetric}_${model}_yhat_upper`}
                                  name={`${model} Upper Bound`}
                                  stroke={strokeColor}
                                  strokeWidth={1}
                                  strokeDasharray="3 3"
                                  dot={false}
                                  opacity={isActive ? 0.5 : 0.1}
                                />
                              </React.Fragment>
                            );
                          })}
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                    {chartType === "area" && (
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={filteredData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis
                            dataKey="Week"
                            tickFormatter={(tick) => {
                              const date = new Date(tick);
                              if (isNaN(date.getTime())) return tick;
                              let label = '';
                              if (plotAggregationType === 'Weekly') {
                                const weekNumber = getISOWeekNumber(date);
                                label = `WK${weekNumber.toString().padStart(2, '0')}`;
                              } else {
                                const month = date.toLocaleString('default', { month: 'short' });
                                const year = date.getFullYear();
                                label = `${month}\n${year}`;
                              }
                              return label;
                            }}
                          />
                          <YAxis tickFormatter={formatNumber} />
                          <RechartsTooltip
                            content={({ payload, label }) => {
                              if (!payload || payload.length === 0) return null;
                              const date = new Date(label);
                              const week = getISOWeekNumber(date);
                              const formattedDate = isNaN(date.getTime())
                                ? label
                                : `${date.getDate().toString().padStart(2, '0')}-${date.toLocaleString('en-US', { month: 'short' })}-${date.getFullYear()} (W${week})`;

                              const actual = payload.find(p => p.name === 'Actual');
                              const modelPredictions = selectedModels.map(model => ({
                                model,
                                value: payload.find(p => p.name === model)?.value,
                                lower: payload[0]?.payload?.[`${selectedMetric}_${model}_yhat_lower`],
                                upper: payload[0]?.payload?.[`${selectedMetric}_${model}_yhat_upper`]
                              }));

                              return (
                                <div className="p-2 rounded-lg border text-sm bg-white text-gray-900 border-gray-300 dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">
                                  <div className="mb-2 font-semibold">{formattedDate}</div>

                                  {actual && (
                                    <div className="flex justify-between text-[13px] font-medium mb-1">
                                      <span className="flex items-center gap-1.5">
                                        <span className="w-[10px] h-[10px] rounded-sm" style={{ backgroundColor: '#8884d8' }}></span>
                                        <span className="text-gray-600 dark:text-gray-300">Actual:</span>
                                      </span>
                                      <span className="text-gray-700 dark:text-gray-200">
                                        {typeof actual.value === 'number' ? formatNumber(actual.value) : actual.value ?? 'N/A'}
                                      </span>
                                    </div>
                                  )}

                                  {modelPredictions.map(({ model, value, lower, upper }) => (
                                    <div key={model} className="mb-1">
                                      <div className="flex justify-between text-[13px] font-medium">
                                        <span className="flex items-center gap-1.5">
                                          <span
                                            className="w-[10px] h-[10px] rounded-sm"
                                            style={{ backgroundColor: modelColors[model] || '#000' }}
                                          ></span>
                                          <span className="text-gray-600 dark:text-gray-300">{model}:</span>
                                        </span>
                                        <span className="text-gray-700 dark:text-gray-200">
                                          {typeof value === 'number' ? formatNumber(value) : value ?? 'N/A'}
                                        </span>
                                      </div>
                                      {lower !== undefined && upper !== undefined && (
                                        <div className="text-[11px] text-gray-500 dark:text-gray-400 ml-4 mt-0.5">
                                          Confidence bounds: {formatNumber(lower)} – {formatNumber(upper)}
                                        </div>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              );
                            }}
                          />

                          <Legend
                            onMouseEnter={(e: any) => setHoveredLegend(e?.value)}
                            onMouseLeave={() => setHoveredLegend(null)}
                            formatter={(value: string) => {
                              const isActive = hoveredLegend === null || hoveredLegend === value;
                              return (
                                <span
                                  style={{
                                    opacity: isActive ? 1 : 0.4,
                                    fontWeight: isActive ? 600 : 400,
                                    color: '#94a3b8',
                                    cursor: 'pointer',
                                  }}
                                >
                                  {value}
                                </span>
                              );
                            }}
                            payload={[
                              ...selectedModels.map(model => ({
                                value: model,
                                type: 'line' as const,
                                color: modelColors[model] || '#000',
                              })),
                              {
                                value: 'Actual',
                                type: 'line' as const,
                                color: '#8884d8',
                              }
                            ]}
                          />
                          <Area
                            type="monotone"
                            dataKey={actualKey}
                            name="Actual"
                            stroke="#8884d8"
                            strokeWidth={2}
                            dot={{ r: 3, strokeWidth: 1 }}
                            activeDot={{ r: 5 }}
                            fill="#8884d8"
                            fillOpacity={hoveredLegend === null || hoveredLegend === 'Actual' ? 0.3 : 0.03}
                          />
                          {selectedModels.map(model => {
                            const isActive = hoveredLegend === null || hoveredLegend === model;
                            const strokeColor = modelColors[model] || '#888';

                            return (
                              <React.Fragment key={model}>
                                <Area
                                  type="monotone"
                                  dataKey={`${selectedMetric}_${model}_yhat`}
                                  name={model}
                                  stroke={strokeColor}
                                  strokeWidth={2}
                                  dot={{ r: 3, strokeWidth: 1 }}
                                  activeDot={{ r: 5 }}
                                  opacity={isActive ? 1 : 0.2}
                                  fill={strokeColor}
                                  fillOpacity={isActive ? 0.3 : 0.03}
                                />
                                <Area
                                  type="monotone"
                                  dataKey={`${selectedMetric}_${model}_yhat_lower`}
                                  name={`${model} Lower Bound`}
                                  stroke={isActive ? strokeColor : 'none'}
                                  strokeWidth={isActive ? 1 : 0}
                                  dot={{ r: 3, strokeWidth: 1 }}
                                  activeDot={{ r: 5 }}
                                  fill={strokeColor}
                                  fillOpacity={isActive ? 0.25 : 0.02}
                                />
                                <Area
                                  type="monotone"
                                  dataKey={`${selectedMetric}_${model}_yhat_upper`}
                                  name={`${model} Upper Bound`}
                                  stroke={isActive ? strokeColor : 'none'}
                                  strokeWidth={isActive ? 1 : 0}
                                  dot={{ r: 3, strokeWidth: 1 }}
                                  activeDot={{ r: 5 }}
                                  fill={strokeColor}
                                  fillOpacity={isActive ? 0.25 : 0.02}
                                />
                              </React.Fragment>
                            );
                          })}
                        </AreaChart>
                      </ResponsiveContainer>
                    )}
                    {chartType === "bar" && (
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={processedData} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} className="opacity-30" />
                          <XAxis
                            dataKey="Week"
                            tickFormatter={(tick) => {
                              const date = new Date(tick);
                              if (isNaN(date.getTime())) return tick;
                              let label = '';
                              if (plotAggregationType === 'Weekly') {
                                const weekNumber = getISOWeekNumber(date);
                                label = `WK${weekNumber.toString().padStart(2, '0')}`;
                              } else {
                                const month = date.toLocaleString('default', { month: 'short' });
                                const year = date.getFullYear();
                                label = `${month}\n${year}`;
                              }
                              return label;
                            }}
                          />
                          <YAxis tickFormatter={formatNumber} />
                          <RechartsTooltip
                            content={({ payload, label }) => {
                              if (!payload || payload.length === 0) return null;
                              const date = new Date(label);
                              const week = getISOWeekNumber(date);
                              const formattedDate = isNaN(date.getTime())
                                ? label
                                : `${date.getDate().toString().padStart(2, '0')}-${date.toLocaleString('en-US', { month: 'short' })}-${date.getFullYear()} (W${week})`;

                              const actual = payload.find(p => p.name === 'Actual');
                              const modelPredictions = selectedModels.map(model => ({
                                model,
                                value: payload.find(p => p.name === model)?.value,
                                lower: payload[0]?.payload?.[`${selectedMetric}_${model}_yhat_lower`],
                                upper: payload[0]?.payload?.[`${selectedMetric}_${model}_yhat_upper`]
                              }));

                              return (
                                <div className="p-2 rounded-lg border text-sm bg-white text-gray-900 border-gray-300 dark:bg-gray-800 dark:text-gray-100 dark:border-gray-600">
                                  <div className="mb-2 font-semibold">{formattedDate}</div>

                                  {actual && (
                                    <div className="flex justify-between text-[13px] font-medium mb-1">
                                      <span className="flex items-center gap-1.5">
                                        <span className="w-[10px] h-[10px] rounded-sm" style={{ backgroundColor: '#8884d8' }}></span>
                                        <span className="text-gray-600 dark:text-gray-300">Actual:</span>
                                      </span>
                                      <span className="text-gray-700 dark:text-gray-200">
                                        {typeof actual.value === 'number' ? formatNumber(actual.value) : actual.value ?? 'N/A'}
                                      </span>
                                    </div>
                                  )}

                                  {modelPredictions.map(({ model, value, lower, upper }) => (
                                    <div key={model} className="mb-1">
                                      <div className="flex justify-between text-[13px] font-medium">
                                        <span className="flex items-center gap-1.5">
                                          <span
                                            className="w-[10px] h-[10px] rounded-sm"
                                            style={{ backgroundColor: modelColors[model] || '#000' }}
                                          ></span>
                                          <span className="text-gray-600 dark:text-gray-300">{model}:</span>
                                        </span>
                                        <span className="text-gray-700 dark:text-gray-200">
                                          {typeof value === 'number' ? formatNumber(value) : value ?? 'N/A'}
                                        </span>
                                      </div>
                                      {lower !== undefined && upper !== undefined && (
                                        <div className="text-[11px] text-gray-500 dark:text-gray-400 ml-4 mt-0.5">
                                          Confidence bounds: {formatNumber(lower)} – {formatNumber(upper)}
                                        </div>
                                      )}
                                    </div>
                                  ))}
                                </div>
                              );
                            }}
                          />

                          <Legend
                            onMouseEnter={(e: any) => setHoveredLegend(e?.value)}
                            onMouseLeave={() => setHoveredLegend(null)}
                            formatter={(value: string) => {
                              const isActive = hoveredLegend === null || hoveredLegend === value;
                              return (
                                <span
                                  style={{
                                    opacity: isActive ? 1 : 0.4,
                                    fontWeight: isActive ? 600 : 400,
                                    color: '#94a3b8',
                                    cursor: 'pointer',
                                  }}
                                >
                                  {value}
                                </span>
                              );
                            }}
                            payload={[
                              ...selectedModels.map(model => ({
                                value: model,
                                type: 'line' as const,
                                color: modelColors[model] || '#000',
                              })),
                              {
                                value: 'Actual',
                                type: 'line' as const,
                                color: '#8884d8',
                              }
                            ]}
                          />
                          <Bar
                            type="monotone"
                            dataKey={actualKey}
                            name="Actual"
                            fill="#8884d8"
                            fillOpacity={hoveredLegend === null || hoveredLegend === 'Actual' ? 1 : 0.4}
                          />
                          {selectedModels.map(model => {
                            const isActive = hoveredLegend === null || hoveredLegend === model;
                            const strokeColor = modelColors[model] || '#888';

                            return (
                              <React.Fragment key={model}>
                                <Bar
                                  dataKey={`${selectedMetric}_${model}_yhat_base`}
                                  stackId={`${model}_bounds`}
                                  fill="transparent"
                                  isAnimationActive={false}
                                />
                                <Bar
                                  dataKey={`${selectedMetric}_${model}_yhat_range`}
                                  stackId={`${model}_bounds`}
                                  name={`${model} Bounds`}
                                  fill={strokeColor}
                                  fillOpacity={isActive ? 0.6 : 0.1}
                                  barSize={22}
                                  isAnimationActive={false}
                                />

                                {/* Prediction bar */}
                                <Bar
                                  dataKey={`${selectedMetric}_${model}_yhat`}
                                  name={model}
                                  fill={strokeColor}
                                  fillOpacity={isActive ? 0.9 : 0.3}
                                  barSize={10}
                                  isAnimationActive={false}
                                />
                              </React.Fragment>
                            );
                          })}
                        </ComposedChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                )}
                {data?.length > 0 && (
                  <div className="mt-10">
                    <h3 className="text-lg font-medium text-gray-800 dark:text-gray-200 mb-2">
                      Model Performance Comparison
                    </h3>
                    <Table>
                      <TableHeader>
                        <TableRow className="bg-gray-100 dark:bg-gray-800">
                          <TableHead
                            id="Model"
                            className="font-bold cursor-pointer"
                            onClick={() => handleSort("Model")}
                          >
                            Model {sortKey === "Model" && (sortOrder === "asc" ? "↑" : "↓")}
                          </TableHead>
                          <TableHead
                            id="MAPE"
                            className="font-bold cursor-pointer"
                            onClick={() => handleSort("MAPE")}
                          >
                            MAPE {sortKey === "MAPE" && (sortOrder === "asc" ? "↑" : "↓")}
                          </TableHead>
                          <TableHead
                            id="RMSE"
                            className="font-bold cursor-pointer"
                            onClick={() => handleSort("RMSE")}
                          >
                            RMSE {sortKey === "RMSE" && (sortOrder === "asc" ? "↑" : "↓")}
                          </TableHead>
                          <TableHead
                            id="R²"
                            className="font-bold cursor-pointer"
                            onClick={() => handleSort("R²")}
                          >
                            R² {sortKey === "R²" && (sortOrder === "asc" ? "↑" : "↓")}
                          </TableHead>
                          <TableHead className="font-bold">Rating</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {sortedModels.map((model) => {
                          const mape = getModelMAPE(filteredData, model);
                          const rmse = getModelRMSE(filteredData, model);
                          const r2 = getModelRSquared(filteredData, model);
                          return (
                            <TableRow key={model}>
                              <TableCell>{model}</TableCell>
                              <TableCell>{mape != null ? `${mape}%` : "N/A"}</TableCell>
                              <TableCell>{rmse != null ? formatWithPrecision(rmse, defaultDecimalPrecisions) : "N/A"}</TableCell>
                              <TableCell>{rmse != null ? formatWithPrecision(r2, defaultDecimalPrecisions) : "N/A"}</TableCell>
                              <TableCell>
                                <ModelRating score={mape} />
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </CardContent>
            </div>
          </Card>
        )}
      </div>
      <Drawer
        PaperProps={{
          sx: { width: '30%' },
        }}
        anchor="right"
        open={aiModal}
        transitionDuration={0}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b bg-gray-100 dark:bg-gray-800">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Ask Gauri AI (Beta)
          </h2>
          <div className="flex gap-2">
            {localStorage.getItem("ai_messages") && JSON.parse(localStorage.getItem("ai_messages") || "[]").length > 0 && (
              <button
                onClick={onClearHistory}
                className="px-3 py-1 text-sm rounded-md bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200"
              >
                Clear
              </button>
            )}
            {/* Close */}
            <button
              onClick={() => showAIModal(false)}
              className="p-1 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700"
            >
              <X className="w-5 h-5 text-gray-700 dark:text-gray-200" />
            </button>
          </div>
        </div>
        <div
          className="h-full overflow-y-auto hidden-scrollbar bg-gray-50 dark:bg-gray-900"
        >
          <AiChat
            model="model_validation"
            modelName="Model Validation"
            isClear={isClear}
            businessData={businessData}
            dateRange={selectedRange}
            onView={() => setTriger(Math.random())}
            userData={userData}
            models={parentSelectedModels}
            lob={parentSelectedMetric}
            cards={cards}
          />
        </div>

      </Drawer>
      <Fab
        color="primary"
        aria-label="ask-ai"
        onClick={() => showAIModal(true)}
        sx={{
          position: "fixed",
          bottom: 70,
          right: 28,
          zIndex: 1500,
        }}
      >

        <UITooltip>
          <TooltipTrigger asChild><Bot size={20} /></TooltipTrigger>
          <TooltipContent>Ask Gauri AI</TooltipContent>
        </UITooltip>
      </Fab>
    </TooltipProvider>
  );
};

const RenderPerformanceMetrics = ({ selectedModels, modelAdjustments, data, selectedMetric, metrics }) => {
  if (selectedModels.length !== 1) return null;

  const modelType = selectedModels[0];
  let metricValues = data?.metrics?.[selectedMetric] || {};
  if (modelType && modelAdjustments[modelType] < 0) {
    metricValues = Object.fromEntries(Object.entries(metricValues).map(([k, v]) => [k, Number(v) * (1 + modelAdjustments[modelType])]));
  }

  return (
    <Card className="w-[300px] shadow-sm border border-gray-200 dark:border-gray-700 rounded-lg p-3 flex flex-col gap-2">
      <div className="flex items-center justify-between mb-1">
        <h4 className="font-semibold text-sm">Performance Metrics</h4>
        <span className="text-green-700 bg-green-100 px-2 py-0.5 rounded-full text-[10px] font-semibold">Good</span>
      </div>
      <p className="text-[10px] text-muted-foreground mb-1">For {metrics.find(m => m.id === selectedMetric)?.name}</p>
      {[
        { key: "MAPE", label: "Mean Absolute Percentage Error" },
        { key: "RMSE", label: "Root Mean Squared Error" },
        { key: "MAE", label: "Mean Absolute Error" },
        { key: "MSE", label: "Mean Squared Error" },
        { key: "R²", label: "R-squared" },
      ].map(({ key, label }) => (
        <div key={key} className="flex justify-between items-center rounded-lg p-2 bg-gray-50 dark:bg-gray-800">
          <div className="flex flex-col">
            <div className="flex items-center gap-1 mb-0.5">
              <span className="font-semibold text-xs">{key}</span>
            </div>
            <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
              {label}
              <TooltipProvider>
                <div className="relative group inline-block">
                  <div className="flex items-center justify-center rounded-full border border-gray-400 dark:border-gray-500 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 text-[9px] font-semibold w-4 h-4 cursor-default">
                    i
                  </div>
                  <div className="absolute z-50 hidden group-hover:block bg-white dark:bg-gray-800 text-[10px] text-gray-800 dark:text-gray-200 border border-gray-300 dark:border-gray-600 rounded p-2 w-56 shadow-lg -left-1/2 top-5">
                    <div className="font-semibold mb-1">{label}</div>
                    {(() => {
                      const val = metricValues[key];
                      if (key === "MSE") {
                        return (
                          <>
                            <p>Mean Squared Error measures average squared difference between predicted and actual values. Lower is better.</p>
                            <p className="mt-1 font-medium">{val !== undefined ? (val < 1 ? "Excellent" : val < 10 ? "Good" : "High error") : "N/A"}</p>
                          </>
                        );
                      } else if (key === "MAE") {
                        return (
                          <>
                            <p>Mean Absolute Error measures average absolute difference. Lower is better.</p>
                            <p className="mt-1 font-medium">{val !== undefined ? (val < 1 ? "Excellent" : val < 10 ? "Good" : "High error") : "N/A"}</p>
                          </>
                        );
                      } else if (key === "RMSE") {
                        return (
                          <>
                            <p>Root Mean Squared Error, in same units as data. Lower is better.</p>
                            <p className="mt-1 font-medium">{val !== undefined ? (val < 1 ? "Excellent" : val < 10 ? "Good" : "High error") : "N/A"}</p>
                          </>
                        );
                      } else if (key === "MAPE") {
                        return (
                          <>
                            <p>Mean Absolute Percentage Error. Lower is better.</p>
                            <p className="mt-1 font-medium">{val !== undefined ? (val < 5 ? "Excellent" : val < 20 ? "Good" : "High error") : "N/A"}</p>
                          </>
                        );
                      } else if (key === "R²") {
                        return (
                          <>
                            <p>R-squared indicates proportion of variance explained. Closer to 1 is better.</p>
                            <p className="mt-1 font-medium">{val !== undefined ? (val > 0.9 ? "Excellent" : val > 0.7 ? "Good" : "Poor fit") : "N/A"}</p>
                          </>
                        );
                      } else {
                        return null;
                      }
                    })()}
                  </div>
                </div>
              </TooltipProvider>
            </div>
          </div>
          <span className="text-sm font-semibold">
            {key === "R²" && metricValues[key] !== undefined
              ? Math.abs(Number(metricValues[key])).toFixed(2)
              : metricValues[key] !== undefined
                ? Number(metricValues[key]).toFixed(2)
                : "N/A"}
          </span>
        </div>
      ))}
      <div className="text-[10px] p-2 rounded-md bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 mt-1">
        Accuracy interpretation: <span className="font-medium">Good</span> prediction accuracy
      </div>
    </Card>
  );
};
