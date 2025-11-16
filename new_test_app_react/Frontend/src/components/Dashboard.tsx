"use client";
import React, { useState, useEffect, useMemo } from "react";
import axios from "axios";
import { parseISO, parse, differenceInWeeks, format, startOfWeek, endOfWeek, subWeeks, addWeeks, isAfter, isBefore, isEqual } from 'date-fns';
import { toZonedTime } from 'date-fns-tz';
import API_CONFIG from '../lib/apiConfig'; // Adjusted path
import { Settings, Home as HomeIcon, FileText, BarChart, Info, LogOut, RefreshCw, Upload } from "lucide-react";
import Home from "./Home";
import { DashboardHeader } from "./DashboardHeader";
import CustomSidebar from "./Sidebar";
import { SidebarProvider } from "@/components/ui/sidebar";
import { ModelConfiguration } from "./ModelConfiguration";
import CircularProgress from '@mui/material/CircularProgress';
import LoadingSkeleton from "./LoadingSkeleton";
import { UploadDataForm } from "./UploadDataForm";
import { TabNavigation } from "./TabNavigation";
import { ActualDataTab } from "./ActualDataTab";
import { ForecastTab } from "./ForecastTab";
import { ModelValidationTab } from "./ModelValidationTab";
import { InsightsTab } from "./InsightsTab";
import { useToast } from "@/hooks/use-toast";
import { KPIMetricsCard } from "./KPIMetricsCard";
import KPIMetrics from "./KPIMetrics";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import TacticalPlanSelection from "./landingPage/TacticalPlanSelection";
import Simulators from "./Simulators";
import IntraDays from "./IntraDays";
import CapacityLook from './CapacityLook'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider
} from "@/components/ui/tooltip";
import { TabContext, TabList, TabPanel } from '@mui/lab';
import { Tab } from '@mui/material';
import { Checkbox } from "@/components/ui/checkbox";
import AuthService from "@/auth/utils/authService";
import { CardContent } from "@/components/ui/card"
import { DateRange } from "react-day-picker";

import DateRangePicker from "./DateRangePicker";

import AppConfig from '../auth/config.js';
import PlanSelection from "./PlanSelection";
import { ContactCenterApp } from "./occupancyModeling/ContactCenterApp";

function formatDateToUTCString(date: Date, formats: string): string {
  const utcDate = toZonedTime(date, 'UTC');
  return format(utcDate, formats);
}

async function tryApiUrls(endpointKey: keyof typeof API_CONFIG.endpoints, formData: FormData) {
  const endpoint = API_CONFIG.endpoints[endpointKey];
  for (const url of API_CONFIG.baseUrls) {
    try {
      // For 'analyze_forecasts', it might be a GET request or POST with empty body
      // Assuming POST for consistency with original code, but this might need adjustment
      const response = await axios.post(`${url}/${endpoint}`, formData, {
        headers: {
          "Content-Type": "multipart/form-data", // This header might not be needed for all endpoints
        },
      });
      return response;
    } catch (error) {
      console.error(`Failed to connect to ${url}/${endpoint}:`, error);
    }
  }
  throw new Error(`All API URLs failed for endpoint: ${endpoint}`);
}

// Helper to get ISO week number from a date string
function getISOWeekNumber(dateStr: string | undefined): number | null {
  if (!dateStr) return null;
  const date = new Date(dateStr);
  if (isNaN(date.getTime())) return null;

  // Set to nearest Thursday: current date + 4 - current day number
  const target = new Date(date.valueOf());
  const dayNr = (date.getDay() + 6) % 7; // Monday=0, Sunday=6
  target.setDate(target.getDate() - dayNr + 3);

  // January 4th is always in week 1
  const jan4 = new Date(target.getFullYear(), 0, 4);
  const dayDiff = (target.getTime() - jan4.getTime()) / 86400000;
  return 1 + Math.floor(dayDiff / 7);
}

type WeekRange = {
  start_week: string; // format: DD/MM/YYYY
  end_week: string;
};

const getWeekCount = ({ start_week, end_week }: WeekRange): number => {
  const startDate = parse(start_week, "dd/MM/yyyy", new Date());
  const endDate = parse(end_week, "dd/MM/yyyy", new Date());

  return differenceInWeeks(endDate, startDate) + 1; // inclusive
};

interface DashboardProps {
  onReset: () => void;
  apiResponse: any;
}

export const Dashboard = ({ onReset, apiResponse }: DashboardProps) => {
  const [modelType, setModelType] = useState("Prophet");
  const [tempModelType, setTempModelType] = useState("Prophet");
  const [forecastPeriod, setForecastPeriod] = useState(4);
  const [tempForecastPeriod, setTempForecastPeriod] = useState(4);
  const [aggregationType, setAggregationType] = useState("Weekly");
  const [tempAggregationType, setTempAggregationType] = useState("Weekly");
  const [trigger, setTriger] = useState('');
  const [trigger1, setTrigger1] = useState('');
  const [modelValidationForecastPeriod, setModelValidationForecastPeriod] = useState(4);
  const [activeTab, setActiveTab] = useState("House");
  const { toast } = useToast();
  const [file, setFile] = useState(null);
  const handleFileUpload = (file) => {
    setFile(file);
    setActiveTab("businessPerformance");
  };
  const [analysisData, setAnalysisData] = useState([]);
  const [futureData, setFutureData] = useState([]);
  const [kpiSourceData, setKpiSourceData] = useState([]);
  const [kpiInitialized, setKpiInitialized] = useState(false);
  const [formData, setFormData] = useState(null);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [loadingIcons, setLoadingIcons] = useState({ forecast: false, insights: false });
  const [insights, setInsights] = useState({});
  const [userData, setUserData] = useState<{ id?: number;[key: string]: any }>({});
  const [headerText, setTitle] = useState('');
  const [defaultWeeks, setDefaultWeeks] = useState(8);
  const [defaultForecastWeeks, setDefaultForecastWeeks] = useState(4);
  const [defaultDecimalPrecisions, setDefaultDecimalPrecision] = useState(0);
  const [defaultConfigId, setDefaultBusinessConfigId] = useState(false);
  const [selectedRange, setSelectedRange] = useState<DateRange | undefined>(undefined);
  const [weekStartsOn, setWeekConfig] = useState(1);
  const [parametersList, setParametersList] = useState([]);
  const [forecastVersions, setForecastVersions] = useState([]);
  const [versionsLoading, setVersionsLoading] = useState(false);
  const [algorithmModelsData, setAlgorithmModels] = useState([]);
  const [forecastModeldata, setForecastModelData] = useState([]);

  const [algForecastDataLoading, setAlgForecastDataLoading] = useState(false);
  const [algorithmForecastModels, setAlgorithmForecastModels] = useState([]);

  const [simulatorsLoading, setSimulatorsLoading] = useState(false);
  const [simulatorsList, setSimulatorsList] = useState([]);

  const [intraDaysLoading, setIntraDayLoading] = useState(true);
  const [intraTrigger, setIntraTrigger] = useState(0);
  const [intraDaysList, setIntraDaysList] = useState([]);



  const today = new Date();


  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  const WEBAPPAPIURL = '/api/v2/';


  const handleLogout = () => {
    AuthService.clearToken();
    window.location.href = "/";
  };


  const [kpiData, setKpiData] = useState([]);
  const [kpiOldData, setKpiOldData] = useState([]);
  const [lastWeekNumber, setLastWeekNumber] = useState(null);
  const [kpiLoading, setKpiLoading] = useState(false);
  const [kpiForeCastLoading, setKpiForeCastLoading] = useState(false);
  const [algDataLoading, setAlgDataLoading] = useState(false)
  const [overallData, setOverallData] = useState([])
  const [overallCurrentData, setOverallCurrentData] = useState([])
  const [overallOldData, setOverallOldData] = useState([])
  const [overallLoading, setOverallLoading] = useState(false)
  const [selectedMetrics, setSelectedMetrics] = useState([]);

  const [selectedParentMetric, setParentSelectedMetric] = useState("");
  const [selectedParentModels, setParentSelectedModels] = useState<string[]>(['Prophet']);

  const [apiData, setData] = useState([]);
  const [apiForecastData, setForecastData] = useState([]);
  interface BusinessData {
    id?: number;
    code?: string;
    display_name?: string;
    [key: string]: unknown;
  }
  const [businessData, setBusinessData] = useState<BusinessData>({});
  const [businessDataList, setBusinessDataList] = useState([]);
  const [defaultConfig, setDefaultConfig] = useState([]);
  const [configLoading, setConfigLoading] = useState(true);
  const [algorithmList, setAlgaorithms] = useState([]);
  const [lobData, setLobdata] = useState([]);

  useEffect(() => {
    if (!AuthService.getAccessToken()) {
      toast({
        title: "Logout",
        description: "Session Expired",
      });
      setTimeout(() => {
        handleLogout();
      }, 3000);
    }
  }, [AuthService.getAccessToken()]);


  // Initial forecast API call on mount
  /* useEffect(() => {
     const fetchInitialForecast = async () => {
       const formData = new FormData();
       formData.append("weeks", "4");
       formData.append("model_type", modelType);
 
       try {
         const response = await tryApiUrls("forecast", formData);
         if (response.data?.dz_df && response.data?.future_df) {
           setAnalysisData(response.data.dz_df);
           setFutureData(response.data.future_df);
           if (!kpiInitialized) {
             setKpiSourceData(response.data.dz_df);
             setKpiInitialized(true);
           }
         } else if (response.data?.dz_df) {
           setAnalysisData(response.data.dz_df);
           setFutureData([]);
           if (!kpiInitialized) {
             setKpiSourceData(response.data.dz_df);
             setKpiInitialized(true);
           }
         } else {
           setAnalysisData(response.data);
           setFutureData([]);
         }
       } catch (error) {
         toast({
           title: "Error",
           description: "Failed to fetch initial forecast data.",
         });
       }
     };
 
     fetchInitialForecast();
   }, []);
 
   useEffect(() => {
     const fetchAnalyzeForecasts = async () => {
       try {
         const response = await tryApiUrls("analyze_forecasts", new FormData());
         console.log("Insights API response:", response.data); // Add debug logging
         setInsights(response.data.insights || {});
         toast({
           title: "Insights fetched",
           description: "Insights data has been successfully loaded.",
         });
       } catch (error) {
         console.error("Error fetching insights:", error); // Add error logging
         toast({
           title: "Error",
           description: "Failed to fetch insights data.",
         });
       }
     };
 
     fetchAnalyzeForecasts(); // Don't forget to call the function
   }, []); */

  const handleApplyChanges = async (forecastPeriod, modelType) => {
    setDefaultForecastWeeks(forecastPeriod);
    /* setLoadingIcons(prev => ({ ...prev, forecast: true }));
     if (!file && forecastType === "external") {
       toast({
         title: "No file uploaded",
         description: "Please upload a file for external factors.",
       });
       setLoadingIcons(prev => ({ ...prev, forecast: false }));
       return;
     }

     const formData = new FormData();
     formData.append("weeks", tempForecastPeriod.toString());

     let model_type = "";
     if (forecastType === "single") {
       model_type = selectedModel;
     } else if (forecastType === "hybrid") {
       model_type = "hybrid";
       formData.append("hybrid_models", JSON.stringify(selectedHybridModels));
     } else if (forecastType === "external") {
       model_type = "external";
       // Handle external factors if needed
     }
     formData.append("model_type", model_type);

     setFormData(formData);

     const response = await tryApiUrls("forecast", formData);
     toast({
       title: "Analysis complete",
       description: "Your dashboard has been updated successfully.",
     });

     let processedData = response.data;

     if (aggregationType === "Monthly" && Array.isArray(processedData)) {
       const monthlyData: Record<string, any> = {};

       processedData.forEach((item: any) => {
         const dateStr = item.date || item.ds; // support 'date' or 'ds'
         if (!dateStr) return;

         const monthKey = dateStr.slice(0, 7); // 'YYYY-MM'

         if (!monthlyData[monthKey]) {
           monthlyData[monthKey] = { ...item, count: 1 };
           monthlyData[monthKey].date = monthKey + "-01"; // set to first of month
         } else {
           Object.keys(item).forEach((key) => {
             if (typeof item[key] === "number") {
               monthlyData[monthKey][key] += item[key];
             }
           });
           monthlyData[monthKey].count += 1;
         }
       });

       // Average numeric fields
       processedData = Object.values(monthlyData).map((item: any) => {
         const count = item.count;
         const newItem = { ...item };
         Object.keys(newItem).forEach((key) => {
           if (typeof newItem[key] === "number" && key !== "count") {
             newItem[key] = newItem[key] / count;
           }
         });
         delete newItem.count;
         return newItem;
       });
     }

     if (processedData?.dz_df && processedData?.future_df) {
       setAnalysisData(processedData.dz_df);
       setFutureData(processedData.future_df);
       if (!kpiInitialized) {
         setKpiSourceData(processedData.dz_df);
         setKpiInitialized(true);
       }
     } else {
       setAnalysisData(processedData);
       setFutureData([]);
     }
   } catch (error) {
     toast({
       title: "Error",
       description: "An error occurred while updating the dashboard.",
     });
   } finally {
     setLoadingIcons(prev => ({ ...prev, forecast: false }));
   } */
  };

  const tabs = [
    { id: "actualData", name: "Historical Data" },
    { id: "forecast", name: "Trends & Forecast" },
    { id: "modelValidation", name: "Model Validation" },
    { id: "insights", name: "Insights" },
    { id: "planning", name: "Planning" },
  ];

  const handleRefresh = () => {
    toast({
      title: "Refreshing Dashboard",
      description: "Updating with the latest data...",
    });
  };

  // Draggable Forecast Settings widget state
  const [showForecastSettings, setShowForecastSettings] = useState(true);

  useEffect(() => {
    if (activeTab === "forecast" || activeTab === "modelValidation") {
      setShowForecastSettings(true);
    } else {
      setShowForecastSettings(false);
    }
  }, [activeTab]);

  const fetchData = async (type: string, id: number, formattedFrom: string, formattedTo: string) => {
    let params = new URLSearchParams({
      limit: '100',
      offset: '0',
      domain: '[]',
      fields: '["name","code","client_id","start_date","end_date","enterprise_group_id"]',
      model: 'business.unit',
    });

    if (activeTab === 'businessPerformance' || type === 'dowIntraday' || type === 'actualData' || type === 'forecast' || type === 'modelValidation' || activeTab === 'insights') {
      params = new URLSearchParams({
        offset: '0',
        domain: `[["business_unit_id","=",${id}],["parameter_id.type","=","Volume"],["date",">=", "${formatDateToUTCString(new Date(formattedFrom), 'yyyy-MM-dd HH:mm:ss')}"],["date","<=", "${formatDateToUTCString(new Date(formattedTo), 'yyyy-MM-dd HH:mm:ss')}"]]`,
        fields: '["date", "calendar_week_id", "business_unit_id", "lob_id", "bpo_id", "week_no", "parameter_id", "value"]',
        model: 'data_feeds',
      });
    }

    setData([]);

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      setKpiLoading(false);
      return response.data;
    } catch (error) {
      setKpiLoading(false);
      if (error.response) {
        console.error('Error response:', error.response);
      } else if (error.request) {
        console.error('No response received:', error.request);
      } else {
        console.error('Error setting up request:', error.message);
      }
      throw error;
    }
  };

  const fetchOverallData = async (type: string, formattedFrom: string, formattedTo: string) => {
    let params = new URLSearchParams({
      limit: '100',
      offset: '0',
      domain: '[]',
      fields: '["name","code","client_id","start_date","end_date","enterprise_group_id"]',
      model: 'business.unit',
    });
    setOverallLoading(true);
    if (activeTab === 'House') {
      params = new URLSearchParams({
        offset: '0',
        domain: `[["date",">=", "${formatDateToUTCString(new Date(formattedFrom), 'yyyy-MM-dd HH:mm:ss')}"],["date","<=", "${formatDateToUTCString(new Date(formattedTo), 'yyyy-MM-dd HH:mm:ss')}"]]`,
        fields: '["date", "calendar_week_id", "business_unit_id", "lob_id", "bpo_id", "week_no", "parameter_id", "value"]',
        model: 'data_feeds',
      });
    }

    setOverallData([]);

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      setOverallLoading(false);
      return response.data;
    } catch (error) {
      setOverallLoading(false);
      console.error('Error fetching data:', error);
    }
  };

  const fetchForecastData = async (type: string, id: number, modelIds: string, formattedFrom: string, formattedTo: string) => {
    let params = new URLSearchParams({
      limit: '100',
      offset: '0',
      domain: '[]',
      fields: '["name","code","client_id","start_date","end_date","enterprise_group_id"]',
      model: 'business.unit',
    });
    // setKpiForeCastLoading(true);
    if (type === 'forecast') {
      params = new URLSearchParams({
        offset: '0',
        domain: `[["business_unit_id","=",${id}],["forecast_model_version","in",${modelIds}],["date",">=", "${formatDateToUTCString(new Date(formattedFrom), 'yyyy-MM-dd HH:mm:ss')}"],["date","<=", "${formatDateToUTCString(new Date(formattedTo), 'yyyy-MM-dd HH:mm:ss')}"]]`,
        fields: '["date","calendar_week_id","forecast_model_version","lower_bound","upper_bound","business_unit_id", "lob_id", "bpo_id", "week_no", "parameter_id", "value"]',
        model: 'forecast_feeds',
      });
    }

    setForecastData([]);

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      setKpiLoading(false);
      return response.data;
    } catch (error) {
      setKpiLoading(false);
      console.error('Error fetching data:', error);
    }
  };

  const fetchForecastModelsData = async (type: string, id: number, modelIds: string, formattedFrom: string, formattedTo: string) => {
    let params = new URLSearchParams({
      limit: '100',
      offset: '0',
      domain: '[]',
      fields: '["name","code","client_id","start_date","end_date","enterprise_group_id"]',
      model: 'business.unit',
    });
    setAlgDataLoading(true);
    if (type === 'modelValidation') {
      params = new URLSearchParams({
        offset: '0',
        domain: `[["business_unit_id","=",${id}],["forecast_model_version","in",${modelIds}],["date",">=", "${formatDateToUTCString(new Date(formattedFrom), 'yyyy-MM-dd HH:mm:ss')}"],["date","<=", "${formatDateToUTCString(new Date(formattedTo), 'yyyy-MM-dd HH:mm:ss')}"]]`,
        fields: '["date","forecast_model_version","calendar_week_id","lower_bound","upper_bound","business_unit_id", "lob_id", "bpo_id", "week_no", "parameter_id", "value"]',
        model: 'forecast_feeds',
      });
    }

    setForecastModelData([]);

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      setAlgDataLoading(false);
      return response.data;
    } catch (error) {
      setAlgDataLoading(false);
      console.error('Error fetching data:', error);
    }
  };

  const fetchAlgorithmModelsData = async (paramName: string, type: string, algs: string) => {
    let params = new URLSearchParams({
      limit: '100',
      offset: '0',
      domain: '[]',
      fields: '["name","code","client_id","start_date","end_date","enterprise_group_id"]',
      model: 'business.unit',
    });
    setAlgDataLoading(true);
    if (type === 'modelValidation') {
      params = new URLSearchParams({
        offset: '0',
        domain: `["|","&",["target_parameter_id.name", "=", "${paramName}"], ["lob_id", "=", false], ["lob_id.name", "=", "${paramName}"], ["algorithms_id","in",${algs}]]`,
        fields: '["id","target_parameter_id","algorithms_id","current_forecast_version","lob_id"]',
        model: 'forecast_model',
      });
    }

    setAlgorithmModels([]);

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      setAlgDataLoading(false);
      return response.data;
    } catch (error) {
      setAlgDataLoading(false);
      console.error('Error fetching data:', error);
    }
  };

  const fetchAlgorithmForecastData = async (paramsName: string, type: string, algs: string) => {
    let params = new URLSearchParams({
      limit: '100',
      offset: '0',
      domain: '[]',
      fields: '["name","code","client_id","start_date","end_date","enterprise_group_id"]',
      model: 'business.unit',
    });
    setAlgForecastDataLoading(true);
    if (type === 'forecast') {
      params = new URLSearchParams({
        offset: '0',
        domain: `["|","&",["target_parameter_id.name", "in", ${paramsName}], ["lob_id", "=", false], ["lob_id.name", "in", ${paramsName}], ["algorithms_id","in",${algs}]]`,
        fields: '["id","target_parameter_id","algorithms_id","current_forecast_version","lob_id"]',
        model: 'forecast_model',
      });
    }

    setAlgorithmForecastModels([]);

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      setAlgForecastDataLoading(false);
      return response.data;
    } catch (error) {
      setAlgForecastDataLoading(false);
      console.error('Error fetching data:', error);
    }
  };


  const fetchModelVersionsData = async (type: string, models: string) => {
    let params = new URLSearchParams({
      limit: '100',
      offset: '0',
      domain: '[]',
      fields: '["name","code","client_id","start_date","end_date","enterprise_group_id"]',
      model: 'business.unit',
    });
    setVersionsLoading(true);
    if (type === 'modelValidation') {
      params = new URLSearchParams({
        offset: '0',
        domain: `[["id","in",${models}]]`,
        fields: '["id","r_squared","mape","mse","rmse","variance","bias","name"]',
        model: 'forecast_model_verison',
      });
    }

    setForecastVersions([]);

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      setVersionsLoading(false);
      return response.data;
    } catch (error) {
      setVersionsLoading(false);
      setForecastVersions([]);
      console.error('Error fetching data:', error);
    }
  };

  const fetchBusinessData = React.useCallback(async () => {
    const params = new URLSearchParams({
      offset: '0',
      domain: '[]',
      order: 'id ASC',
      fields: '["display_name","code","description","sequence","preferred_algorithm"]',
      model: 'business.unit',
    });
    setKpiLoading(true);

    setBusinessDataList([]);

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      return response?.data ?? '';
    } catch (error) {
      setBusinessDataList([]);
      console.error('Error fetching data:', error);
    }
  }, []);


  const fetchParameterData = React.useCallback(async () => {
    const params = new URLSearchParams({
      offset: '0',
      domain: '[]',
      order: 'id ASC',
      fields: '["direction","name"]',
      model: 'parameters',
    });
    // setKpiLoading(true);

    setParametersList([]);

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      return response?.data ?? '';
    } catch (error) {
      setParametersList([]);
      console.error('Error fetching data:', error);
    }
  }, []);


  const fetchDefaults = React.useCallback(async (id) => {
    const params = new URLSearchParams({
      offset: '0',
      domain: `[["user_id","=",${id}]]`,
      order: 'id ASC',
      fields: '["name","value"]',
      model: 'user.default_preferences',
    });
    setConfigLoading(true);
    setDefaultConfig([]);

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      // Ensure the data is always an array
      if (Array.isArray(response?.data)) {
        return response.data;
      } else {
        return [];
      }
    } catch (error) {
      setDefaultConfig([]);
      setConfigLoading(false);
      console.error('Error fetching data:', error);
      return [];
    }
  }, []);

  const fetchWeekConfig = React.useCallback(async () => {
    const params = new URLSearchParams({
      limit: '1',
      offset: '0',
      domain: '[]',
      order: 'id ASC',
      fields: '["anchor_weekday"]',
      model: 'settings',
    });
    // setKpiLoading(true);
    setWeekConfig(1);

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      return response.data?.[0]?.anchor_weekday > - 1 ? response.data?.[0]?.anchor_weekday : 1; // Default to 1 if no data found
    } catch (error) {
      setWeekConfig(1);
      console.error('Error fetching data:', error);
    }
  }, []);


  const fetchLobData = React.useCallback(async (id) => {
    const params = new URLSearchParams({
      offset: '0',
      domain: `[["business_unit","=",${id}]]`,
      order: 'sequence ASC',
      fields: '["description","name","sequence","id","color_code","preferred_algorithm","contributors_ids"]',
      model: 'line_business_lob',
    });
    setLobdata([]);

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      return response?.data || []; // Default to 1 if no data found
    } catch (error) {
      setLobdata([]);
      console.error('Error fetching data:', error);
    }
  }, []);


  const fetchAlagorithmsList = React.useCallback(async () => {
    const params = new URLSearchParams({
      offset: '0',
      domain: `[]`,
      order: 'sequence ASC',
      fields: '["description","name","sequence","id","category"]',
      model: 'forecast_algorithm',
    });
    setAlgaorithms([]);

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      return response?.data || []; // Default to 1 if no data found
    } catch (error) {
      setAlgaorithms([]);
      console.error('Error fetching data:', error);
    }
  }, []);

  const fetchSimulatorsList = React.useCallback(async (id) => {
    const params = new URLSearchParams({
      offset: '0',
      domain: `[["business_unit","=",${id}]]`,
      order: 'id DESC',
      fields: '["description","name","start_date","end_date","state","reference","create_uid","create_date","write_date"]',
      model: 'what_if_simulator',
    });
    setSimulatorsList([]);

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      return response?.data || []; // Default to 1 if no data found
    } catch (error) {
      setSimulatorsList([]);
      console.error('Error fetching data:', error);
    }
  }, []);

  const fetchIntraDaysList = React.useCallback(async (id) => {
    const params = new URLSearchParams({
      offset: '0',
      domain: `[["business_unit","=",${id}]]`,
      order: 'id DESC',
      fields: '["reference","name","business_unit","forecast_end_week","data","forecast_start_week","lob_id","parameter_id","create_uid","create_date","reference_start_week","reference_end_week","remarks"]',
      model: 'dow_intraday',
    });
    setIntraDaysList([]);

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      return response?.data || []; // Default to 1 if no data found
    } catch (error) {
      setSimulatorsList([]);
      console.error('Error fetching data:', error);
    }
  }, []);


  const fetchUser = React.useCallback(async () => {
    // setKpiLoading(true);
    setUserData({});

    try {
      const response = await axios.get(`${WEBAPPAPIURL}userinfo`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      return response?.data ?? '';
    } catch (error) {
      setUserData({});
      console.error('Error fetching data:', error);
    }
  }, []);

  function getDateFnsWeekStart(weekEndDay: number): number {
    return ((weekEndDay + 1) % 7 + 1) % 7;
  }

  useEffect(() => {
    const runUser = async () => {
      try {
        const result = await fetchUser();
        setUserData(result);
      } catch (err) {
        setUserData({});
        console.error('Failed to fetch:', err);
      }
    };

    const runBusinessList = async () => {
      try {
        const result = await fetchBusinessData();
        setBusinessDataList(result);
        setKpiLoading(false)
      } catch (err) {
        setBusinessDataList([]);
        setKpiLoading(false)
        console.error("Failed to fetch:", err);
      }
    };


    const runParameters = async () => {
      try {
        const result = await fetchParameterData();
        setParametersList(result);
      } catch (err) {
        setParametersList([]);
        console.error("Failed to fetch:", err);
      }
    };



    const runWeekConfig = async () => {
      try {
        const result = await fetchWeekConfig();
        const weekEndDay = parseInt(result) + 1;
        // const weekStartDay = getDateFnsWeekStart(weekEndDay);
        setWeekConfig(weekEndDay);
      } catch (err) {
        setWeekConfig(1);
        console.error("Failed to fetch:", err);
      }
    };

    runUser();
    runBusinessList();
    runParameters();
    runWeekConfig();
  }, [fetchUser, fetchBusinessData, fetchParameterData, fetchWeekConfig]);

  useEffect(() => {

    const runDefault = async () => {
      if (userData && userData.sub) {
        try {
          const result = await fetchDefaults(userData.sub);
          setDefaultConfig(result);
          setConfigLoading(false);
        } catch (err) {
          setDefaultConfig([]);
          console.error("Failed to fetch:", err);
        }
      }
    };

    runDefault();
  }, [userData, fetchDefaults]);

  useEffect(() => {
    const runFetch = async () => {
      if (
        businessData?.id
      ) {
        try {
          const result = await fetchLobData(businessData?.id); // call the function
          setLobdata(result); // store the data in state
        } catch (err) {
          setLobdata([]);
          console.error("Failed to fetch:", err);
        }
      }
    };

    runFetch();
  }, [businessData, fetchLobData]);

  useEffect(() => {
    const runFetch1 = async () => {
      if (userData && userData.sub) {
        try {
          const result = await fetchBusinessData();
          setBusinessDataList(result);
          setKpiLoading(false)
          setTrigger1(Math.random())
        } catch (err) {
          setBusinessDataList([]);
          setKpiLoading(false)
          console.error("Failed to fetch:", err);
        }
      }
    };

    const runFetch = async () => {
      if (
        businessData?.id
      ) {
        try {
          const result = await fetchLobData(businessData?.id); // call the function
          setLobdata(result); // store the data in state
        } catch (err) {
          setLobdata([]);
          console.error("Failed to fetch:", err);
        }
      }
    };
    runFetch1();
    runFetch();
  }, [trigger, fetchBusinessData, fetchLobData]);

  useEffect(() => {
    const runFetch = async () => {
      if (
        businessData?.id
      ) {
        try {
          const result = await fetchAlagorithmsList(); // call the function
          setAlgaorithms(result); // store the data in state
        } catch (err) {
          setAlgaorithms([]);
          console.error("Failed to fetch:", err);
        }
      }
    };

    runFetch();
  }, [businessData, fetchAlagorithmsList]);

  useEffect(() => {
    const runFetch = async () => {
      if (
        activeTab === 'What-If'
        && businessData?.id
      ) {
        setSimulatorsLoading(true);
        try {
          const result = await fetchSimulatorsList(businessData?.id); // call the function
          setSimulatorsList(result); // store the data in state
          setSimulatorsLoading(false);
        } catch (err) {
          setSimulatorsList([]);
          setSimulatorsLoading(false);
          console.error("Failed to fetch:", err);
        }
      }
    };

    runFetch();
  }, [activeTab, businessData, fetchSimulatorsList]);

  useEffect(() => {
    const runFetch = async () => {
      if (
        activeTab === 'dowIntraday'
        && businessData?.id
      ) {
        setIntraDayLoading(true);
        try {
          const result = await fetchIntraDaysList(businessData?.id); // call the function
          setIntraDaysList(result); // store the data in state
          setIntraDayLoading(false);
        } catch (err) {
          setIntraDaysList([]);
          setIntraDayLoading(false);
          console.error("Failed to fetch:", err);
        }
      }
    };

    runFetch();
  }, [activeTab, businessData, intraTrigger, fetchIntraDaysList]);

  const isJsonString = (str: string) => {
    try {
      JSON.parse(str);
      return true;
    } catch {
      return false;
    }
  };

  const getJsonString = (str: string) => {
    try {
      return JSON.parse(str);;
    } catch {
      return false;
    }
  };

  useMemo(() => {
    let defaultWeeks = 8;
    let defaultWeekD = 8;
    let defaultUnit = 'wfs'
    let defaultForecastWeek = 4
    let defaultBusinessConfigId = false;
    let defaultWeekData = false;
    let defaultDecimalPrecision = 0;
    if (defaultConfig && defaultConfig.length > 0) {
      defaultUnit = defaultConfig.find(item => item.name === 'default_business_unit')?.value || 'wfs';
      defaultWeekD = defaultConfig.find(item => item.name === 'default_week_range')?.value || 8;
      defaultWeekData = defaultConfig.find(item => item.name === 'default_week_range')?.value || false;
      defaultForecastWeek = defaultConfig.find(item => item.name === 'default_forecast_week_range')?.value || 4;
      defaultBusinessConfigId = defaultConfig.find(item => item.name === 'default_business_unit')?.id || false;
      defaultDecimalPrecision = defaultConfig.find(item => item.name === 'decimal_precision')?.value || 0;
    }
    if (
      typeof defaultWeekData === "string" &&
      defaultWeekData &&
      isJsonString(defaultWeekData) &&
      getJsonString(defaultWeekData)?.start_week
    ) {
      defaultWeeks = getWeekCount(getJsonString(defaultWeekData));
    } else {
      defaultWeeks = defaultWeekD;
    }
    setDefaultWeeks(defaultWeeks);
    setDefaultForecastWeeks(defaultForecastWeek);
    setDefaultDecimalPrecision(defaultDecimalPrecision);
    setTempForecastPeriod(defaultForecastWeek);
    setDefaultBusinessConfigId(defaultBusinessConfigId);
    if (!(businessData && businessData?.id)) {
      setBusinessData(businessDataList.find(item => item.code === defaultUnit) || (businessDataList && businessDataList.length > 0 ? businessDataList[0] : {}));
    }
    let defaultFrom = startOfWeek(subWeeks(today, defaultWeeks - 1), { weekStartsOn: weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6 }); // 8 weeks = from 7 weeks ago (inclusive)
    let defaultTo = endOfWeek(today, { weekStartsOn: weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6 });
    if (
      typeof defaultWeekData === "string" &&
      defaultWeekData &&
      isJsonString(defaultWeekData) &&
      getJsonString(defaultWeekData)?.start_week
    ) {
      // Assume getJsonString(defaultWeekData)?.start_week = "26/05/2025"
      const startWeekStr = getJsonString(defaultWeekData)?.start_week;
      const endWeekStr = getJsonString(defaultWeekData)?.end_week;

      // Parse the DD/MM/YYYY string correctly
      const startDate = parse(startWeekStr, "dd/MM/yyyy", new Date());
      const endDate = parse(endWeekStr, "dd/MM/yyyy", new Date());
    
      defaultFrom = startOfWeek(startDate, { weekStartsOn: weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6 }); // 8 weeks = from 7 weeks ago (inclusive)
      defaultTo = endOfWeek(endDate, { weekStartsOn: weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6 });
    }
    setSelectedRange({
      from: defaultFrom,
      to: defaultTo,
    })
  }, [defaultConfig, weekStartsOn]);

  useEffect(() => {
    let defaultUnit = 'wfs'
    if (defaultConfig && defaultConfig.length > 0) {
      defaultUnit = defaultConfig.find(item => item.name === 'default_business_unit')?.value || 'wfs';
    }
    setBusinessData(businessDataList.find(item => item.code === defaultUnit) || (businessDataList && businessDataList.length > 0 ? businessDataList[0] : {}));
  }, [trigger1]);

  const onDateDefaultChange = () => {
    const defaultFrom = startOfWeek(subWeeks(today, defaultWeeks - 1), { weekStartsOn: weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6 }); // 8 weeks = from 7 weeks ago (inclusive)
    const defaultTo = endOfWeek(today, { weekStartsOn: weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6 });
    setSelectedRange({
      from: defaultFrom,
      to: defaultTo,
    })
  };

  useEffect(() => {
    const runFetch = async () => {
      if (
        businessData?.id &&
        selectedRange?.from &&
        selectedRange?.to &&
        (activeTab === "businessPerformance" ||
          activeTab === "actualData" ||
          activeTab === "forecast" || activeTab === 'dowIntraday' || activeTab === "modelValidation" || activeTab === 'insights')
      ) {
        setKpiLoading(true);
        try {
          let formattedFrom = selectedRange.from ? format(selectedRange.from, 'yyyy-MM-dd') : '';
          const formattedTo = selectedRange.to ? format(selectedRange.to, 'yyyy-MM-dd') : '';
          if (activeTab === "businessPerformance") {
            const previousFrom = selectedRange.from
              ? startOfWeek(subWeeks(selectedRange.from, defaultWeeks - 1), { weekStartsOn: weekStartsOn as import('date-fns').Day })
              : null;
            /*  const previousTo = previousFrom
                ? endOfWeek(subWeeks(selectedRange.from!, 1), { weekStartsOn })
                : null; */

            formattedFrom = previousFrom ? format(previousFrom, "yyyy-MM-dd") : "";
          }
          const result = await fetchData(activeTab, businessData?.id, formattedFrom, formattedTo); // call the function
          setData(result); // store the data in state
          setKpiLoading(false);
        } catch (err) {
          setData([]);
          setKpiLoading(false);
          console.error("Failed to fetch:", err);
        }
      }
    };

    runFetch();
  }, [businessData?.id, activeTab, JSON.stringify(selectedRange)]);


  useEffect(() => {
    const runFetch = async () => {
      if (
        selectedRange?.from &&
        selectedRange?.to &&
        activeTab === "House"
      ) {
        try {
          const formattedTo = selectedRange.to ? format(selectedRange.to, 'yyyy-MM-dd') : '';
          const previousFrom = selectedRange.from
            ? startOfWeek(subWeeks(selectedRange.from, defaultWeeks - 1), { weekStartsOn: weekStartsOn as import('date-fns').Day })
            : null;

          const formattedFrom = previousFrom ? format(previousFrom, "yyyy-MM-dd") : "";
          const result = await fetchOverallData(activeTab, formattedFrom, formattedTo); // call the function
          setOverallData(result); // store the data in state
          setOverallLoading(false);
        } catch (err) {
          setOverallData([]);
          setOverallLoading(false);
          console.error("Failed to fetch:", err);
        }
      }
    };

    runFetch();
  }, [activeTab, selectedRange]);


  const selectedModelsKey = useMemo(
    () => JSON.stringify(selectedParentModels.filter((m) => m !== '')),
    [selectedParentModels]
  );

  useEffect(() => {
    const fetchAllData = async () => {
      if (
        !businessData?.id ||
        activeTab !== "modelValidation" ||
        !selectedParentMetric ||
        !selectedRange?.from ||
        !selectedRange?.to ||
        !selectedParentModels?.length
      )
        return;

      try {
        setAlgDataLoading(true);

        const mdata = kpiData.filter(
          (item) => item.parameterName?.trim() === selectedParentMetric
        );

        const newMetric = mdata.length > 0 ? mdata[0].parameterName?.trim() : selectedParentMetric;
        let algoModels = [];
        if (selectedModelsKey && selectedModelsKey.length > 0) {
          // Step 1: Fetch Algorithm Models
          algoModels = await fetchAlgorithmModelsData(
            newMetric,
            activeTab,
            selectedModelsKey
          );
          setAlgorithmModels(algoModels);
        } else {
          setAlgorithmModels([])
        }

        const modelIds = algoModels.map((m) => Array.isArray(m.current_forecast_version) ? m.current_forecast_version[0] : null).filter((id) => typeof id === "number" && id > 0);

        if (!modelIds.length) return;

        // Step 3: Fetch Forecast Data
        const formattedFrom = format(selectedRange.from, "yyyy-MM-dd");
        const formattedTo = format(selectedRange.to, "yyyy-MM-dd");

        const forecastData = await fetchForecastModelsData(
          activeTab,
          businessData.id,
          JSON.stringify(modelIds),
          formattedFrom,
          formattedTo
        );

        setForecastModelData(forecastData);
      } catch (err) {
        console.error("Fetch failed:", err);
        setAlgorithmModels([]);
        setForecastModelData([]);
        setAlgDataLoading(false);
      } finally {
        setAlgDataLoading(false);
      }
    };

    fetchAllData();
  }, [
    businessData?.id,
    activeTab,
    selectedParentMetric,
    selectedModelsKey,
    selectedRange?.from,
    selectedRange?.to,
  ]);

  useEffect(() => {
    const runFetch = async () => {
      if (
        activeTab === 'modelValidation'
        && businessData?.id
        && algorithmModelsData
      ) {
        try {
          const modelIds = algorithmModelsData.map((m) => Array.isArray(m.current_forecast_version) ? m.current_forecast_version[0] : null).filter((id) => typeof id === "number" && id > 0);
          if (modelIds && modelIds.length) {
            const result = await fetchModelVersionsData(activeTab, JSON.stringify(modelIds)); // call the function
            setForecastVersions(result);
          } else {
            setForecastVersions([])
          }
        } catch (err) {
          console.error("Failed to fetch:", err);
        }
      }
    };

    runFetch();
  }, [activeTab, JSON.stringify(algorithmModelsData)]);



  useEffect(() => {
    const fetchAllData = async () => {
      if (
        !businessData?.id ||
        activeTab !== "forecast" ||
        !lobData?.length ||
        !selectedMetrics ||
        !selectedRange?.from ||
        !selectedRange?.to
      )
        return;

      try {
        setAlgForecastDataLoading(true);

        // 1. Extract preferred algorithm names from lobData
        let preferredAlgorithmIds = lobData
          .filter((m) => selectedMetrics.includes(m.name))
          .map((m) => m.preferred_algorithm?.[1])
          .filter((name) => name !== '' && name !== null && name !== undefined);


        // 2. Filter mdata from kpiData where lob_id is falsy and metric is selected
        const mdata = kpiData.filter(
          (item) => item.source === 'parameter' && selectedMetrics.includes(item.parameterName?.trim())
        );

        // 3. If there's matching data and a business-level preferred algorithm, add it
        if (mdata.length && businessData?.preferred_algorithm?.[1]) {
          preferredAlgorithmIds = [
            ...preferredAlgorithmIds,
            businessData.preferred_algorithm[1],
          ];
        }

        // Optional: remove duplicates
        preferredAlgorithmIds = [...new Set(preferredAlgorithmIds)];


        // Step 1: Fetch Algorithm Models
        const algoModels = await fetchAlgorithmForecastData(
          JSON.stringify(selectedMetrics),
          activeTab,
          selectedMetrics && selectedMetrics.length ? JSON.stringify(preferredAlgorithmIds) : []
        );
        setAlgorithmForecastModels(algoModels);

        const modelIds = algoModels.map((m) => Array.isArray(m.current_forecast_version) ? m.current_forecast_version[0] : null).filter((id) => typeof id === "number" && id > 0);

        if (!modelIds.length) { setForecastData([]); return; }

        // Step 3: Fetch Forecast Data
        const formattedFrom = selectedRange.from ? format(selectedRange.from, 'yyyy-MM-dd') : '';
        let formattedTo = selectedRange.to ? format(selectedRange.to, 'yyyy-MM-dd') : '';

        if (selectedRange?.to && activeTab === "forecast") {
          const addedWeeksDate = addWeeks(selectedRange.to, defaultForecastWeeks); // change 4 to how many weeks you want
          const toDate = endOfWeek(addedWeeksDate, { weekStartsOn: weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6 });

          formattedTo = format(toDate, "yyyy-MM-dd");
        }

        const forecastData = await fetchForecastData(
          activeTab,
          businessData.id,
          JSON.stringify(modelIds),
          formattedFrom,
          formattedTo
        );

        setForecastData(forecastData);
      } catch (err) {
        console.error("Fetch failed:", err);
        setAlgorithmForecastModels([]);
        setForecastData([]);
        setKpiForeCastLoading(false);
      } finally {
        setAlgForecastDataLoading(false);
        setKpiForeCastLoading(false);
      }
    };

    fetchAllData();
  }, [
    activeTab,
    selectedMetrics,
    lobData,
    defaultForecastWeeks,
    selectedRange?.from,
    selectedRange?.to,
  ]);



  /*  useEffect(() => {
     const runFetch = async () => {
       if (
         businessData?.id &&
         selectedRange?.from &&
         selectedRange?.to &&
         (activeTab === "forecast")
       ) {
         try {
           const formattedFrom = selectedRange.from ? format(selectedRange.from, 'yyyy-MM-dd') : '';
           let formattedTo = selectedRange.to ? format(selectedRange.to, 'yyyy-MM-dd') : '';
           if (selectedRange?.to && activeTab === "forecast") {
               const addedWeeksDate = addWeeks(selectedRange.to, defaultForecastWeeks); // change 4 to how many weeks you want
               const toDate = endOfWeek(addedWeeksDate, { weekStartsOn: weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6 });

               formattedTo = format(toDate, "yyyy-MM-dd");
           }
           const result = await fetchForecastData(activeTab, businessData?.id, formattedFrom, formattedTo); // call the function
           setForecastData(result); // store the data in state
           setKpiForeCastLoading(false);
         } catch (err) {
           setForecastData([]);
           setKpiForeCastLoading(false);
           console.error("Failed to fetch:", err);
         }
       }
     };

     runFetch();
   }, [businessData, activeTab, defaultForecastWeeks, selectedRange]); */


  useEffect(() => {
    setTitle(businessData?.display_name || 'Walmart Fulfillment Services')
  }, [businessData]);

  useEffect(() => {
    if (
      (activeTab === "businessPerformance" ||
        activeTab === "actualData" || activeTab === 'dowIntraday' || activeTab === 'forecast' || activeTab === 'modelValidation' || activeTab === 'insights') &&
      apiData && apiData?.length &&
      selectedRange
    ) {
      const parameterGroups: Record<string, { title: string; value: number; parameterName?: string; parameterId?: number }> = {};
      const lobGroups: Record<string, { title: string; value: number; parameterName?: string; lobId?: number; parameterId?: number }> = {};

      const parameterOldGroups: Record<string, { title: string; value: number; parameterName?: string; parameterId?: number }> = {};
      const lobOldGroups: Record<string, { title: string; value: number; parameterName?: string; lobId?: number; parameterId?: number }> = {};

      const { from, to } = selectedRange;

      const newData = [...apiData];

      // Calculate previous period range
      const previousFrom = from
        ? startOfWeek(subWeeks(from, defaultWeeks - 1), { weekStartsOn: weekStartsOn as import('date-fns').Day })
        : null;
      const previousTo = previousFrom
        ? endOfWeek(subWeeks(from, 1), { weekStartsOn: weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6 })
        : null;

      // Filter for previous period data
      const filteredOldData =
        previousFrom && previousTo
          ? newData.filter((item) => {
            const itemDate = new Date(item.date);
            return (
              (isAfter(itemDate, previousFrom) || isEqual(itemDate, previousFrom)) &&
              (isBefore(itemDate, previousTo) || isEqual(itemDate, previousTo))
            );
          })
          : [];

      filteredOldData.forEach((item) => {
        const lobName = item.lob_id?.[1]?.trim();
        const paramName = item.parameter_id?.[1]?.trim();

        if (!lobName && paramName) {
          if (!parameterOldGroups[paramName]) {
            parameterOldGroups[paramName] = { title: paramName, value: 0 };
          }
          parameterOldGroups[paramName].value += item.value;
        } else if (lobName) {
          if (!lobOldGroups[lobName]) {
            lobOldGroups[lobName] = { title: lobName, value: 0 };
          }
          lobOldGroups[lobName].value += item.value;
        }
      });

      const groupedOldData = [
        ...Object.values(parameterOldGroups),
        ...Object.values(lobOldGroups),
      ];
      setKpiOldData(groupedOldData);

      // Filter for current period data
      const filteredData =
        from && to
          ? newData.filter((item) => {
            const itemDate = new Date(item.date);
            return (
              (isAfter(itemDate, from) || isEqual(itemDate, from)) &&
              (isBefore(itemDate, to) || isEqual(itemDate, to))
            );
          })
          : [];

      filteredData.forEach((item) => {
        const lobId = item.lob_id?.[0];          // <-- Extract ID
        const lobName = item.lob_id?.[1]?.trim();

        const parameterId = item.parameter_id?.[0];  // <-- Extract ID
        const paramName = item.parameter_id?.[1]?.trim();

        if (!lobName && paramName) {
          if (!parameterGroups[paramName]) {
            parameterGroups[paramName] = {
              title: paramName,
              value: 0,
              parameterName: paramName,
              parameterId: parameterId, // <-- Store ID
            };
          }
          parameterGroups[paramName].value += item.value;
        } else if (lobName) {
          if (!lobGroups[lobName]) {
            lobGroups[lobName] = {
              title: lobName,
              value: 0,
              parameterName: paramName,
              lobId: lobId,             // <-- Store ID
              parameterId: parameterId, // optional, if needed
            };
          }
          lobGroups[lobName].value += item.value;
        }
      });

      const groupedData = [
        ...Object.values(parameterGroups).map(item => ({ ...item, source: "parameter" })),
        ...Object.values(lobGroups).map(item => ({ ...item, source: "lob" })),
      ];

      setKpiData(groupedData);
    } else if (!apiData?.length && !kpiLoading) {
      setKpiData([]);
      setKpiOldData([]);
    }
  }, [apiData, activeTab, selectedRange, defaultWeeks, kpiLoading]);

  useEffect(() => {
    if (
      activeTab === "House"
      && overallData && overallData?.length
      && selectedRange
    ) {

      const { from, to } = selectedRange;

      const newData = [...overallData];

      // Calculate previous period range
      const previousFrom = from
        ? startOfWeek(subWeeks(from, defaultWeeks - 1), { weekStartsOn: weekStartsOn as import('date-fns').Day })
        : null;
      const previousTo = previousFrom
        ? endOfWeek(subWeeks(from, 1), { weekStartsOn: weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6 })
        : null;

      // Filter for previous period data
      const filteredOldData =
        previousFrom && previousTo
          ? newData.filter((item) => {
            const itemDate = new Date(item.date);
            return (
              (isAfter(itemDate, previousFrom) || isEqual(itemDate, previousFrom)) &&
              (isBefore(itemDate, previousTo) || isEqual(itemDate, previousTo))
            );
          })
          : [];


      setOverallOldData(filteredOldData);

      // Filter for current period data
      const filteredData =
        from && to
          ? newData.filter((item) => {
            const itemDate = new Date(item.date);
            return (
              (isAfter(itemDate, from) || isEqual(itemDate, from)) &&
              (isBefore(itemDate, to) || isEqual(itemDate, to))
            );
          })
          : [];

      setOverallCurrentData(filteredData);
    } else if (!overallData?.length && !overallLoading) {
      setOverallCurrentData([]);
      setOverallOldData([]);
    }
  }, [overallData, activeTab, selectedRange, defaultWeeks, overallLoading]);

  const [pos, setPos] = useState({ x: window.innerWidth - 380, y: 40 });
  const [dragging, setDragging] = useState(false);
  const [rel, setRel] = useState({ x: 0, y: 0 });

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!dragging) return;
      setPos({
        x: e.clientX - rel.x,
        y: e.clientY - rel.y,
      });
    };

    const handleMouseUp = () => {
      setDragging(false);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [dragging, rel]);

  // Inside the renderTabContent function
  const renderTabContent = () => {
    switch (activeTab) {
      case "House":
        return (
          <Home
            overallData={overallCurrentData}
            overallOldData={overallOldData}
            overallLoading={overallLoading}
            options={businessDataList}
            parametersList={parametersList}
            setActiveMainTab={setActiveTab}
            setBusinessData={setBusinessData}
            defaultDecimalPrecisions={defaultDecimalPrecisions}
            defaultConfig={defaultConfig}
            configLoading={configLoading}
            businessData={businessData}
            userData={userData}
          >
            <DateRangePicker
              date={selectedRange}
              onDateChange={setSelectedRange}
              onDateDefaultChange={onDateDefaultChange}
              className=""
              weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6}
              defaultPreset="This Quarter"
            />
          </Home>
        );
      case "actualData":
        if (kpiLoading) {
          return (
            <div className="space-y-4">
              <LoadingSkeleton />
            </div>
          ); // or use a Spinner component
        }
        return (
          <ActualDataTab
            data={{ dz_df: apiData }}
            aggregationType={aggregationType}
            setAggregationType={setAggregationType}
            plotAggregationType={aggregationType}
            metricKeys={kpiData.map(item => item.title)}
            metricData={kpiData}
            defaultDecimalPrecisions={defaultDecimalPrecisions}
            setPlotAggregationType={setAggregationType}
          >
            <DateRangePicker
              date={selectedRange}
              onDateChange={setSelectedRange}
              onDateDefaultChange={onDateDefaultChange}
              className=""
              weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6}
            />
          </ActualDataTab>
        );
      case "forecast":
        if (kpiLoading || kpiForeCastLoading) {
          return (
            <div className="space-y-4">
              <LoadingSkeleton />
            </div>
          ); // or use a Spinner component
        }

        return (
          <ForecastTab
            aggregationType={aggregationType}
            modelType={modelType}
            businessData={businessData}
            defaultDecimalPrecisions={defaultDecimalPrecisions}
            forecastPeriod={defaultForecastWeeks}
            selectedRange={selectedRange}
            setSelectedMetrics={setSelectedMetrics}
            selectedParentMetrics={selectedMetrics}
            metricKeys={kpiData.map(item => item.title)}
            metricData={kpiData}
            data={{
              dz_df: Array.isArray(apiData) ? apiData : [],
              future_df: Array.isArray(apiForecastData) ? apiForecastData : [],
            }}
            algorithmModels={algorithmList}
            lobData={lobData}
            insights={insights}
            setIsDrawerOpen={setIsDrawerOpen}
            setTriger={setTriger}
            loading={kpiLoading}
            weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6}
          >
            <DateRangePicker
              date={selectedRange}
              onDateChange={setSelectedRange}
              onDateDefaultChange={onDateDefaultChange}
              className=""
              weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6}
            />
          </ForecastTab>
        );
      case "modelValidation":
        if (kpiLoading || kpiForeCastLoading) {
          return (
            <div className="space-y-4">
              <LoadingSkeleton />
            </div>
          ); // or use a Spinner component
        }

        return (
          <ModelValidationTab
            aggregationType={aggregationType}
            metricData={kpiData}
            setParentSelectedModels={setParentSelectedModels}
            setParentSelectedMetric={setParentSelectedMetric}
            parentSelectedModels={selectedParentModels}
            parentSelectedMetric={selectedParentMetric}
            data={Array.isArray(apiData) ? apiData : []}
            defaultDecimalPrecisions={defaultDecimalPrecisions}
            businessData={businessData}
            algorithmModelsData={algorithmModelsData}
            foreCastdata={Array.isArray(forecastModeldata) ? forecastModeldata : []}
            modelType={modelType}
            setTriger={setTriger}
            trigger={trigger}
            algorithmModels={algorithmList}
            lobData={lobData}
            userData={userData}
            loading={kpiLoading || algDataLoading}
            versionsLoading={versionsLoading}
            forecastVersions={forecastVersions}
            foreCastLoading={algDataLoading}
            setPlotAggregationType={setAggregationType}
            plotAggregationType={aggregationType}
            selectedRange={selectedRange}
          >
            <DateRangePicker
              date={selectedRange}
              onDateChange={setSelectedRange}
              onDateDefaultChange={onDateDefaultChange}
              disableFuture
              className=""
              weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6}
            />
          </ModelValidationTab>
        );
      case "insights":
        if (kpiLoading) {
          return (
            <div className="space-y-4">
              <LoadingSkeleton />
            </div>
          ); // or use a Spinner component
        }
        return (
          <InsightsTab defaultDecimalPrecisions={defaultDecimalPrecisions} loading={kpiLoading} metricKeys={kpiData.map(item => item.title)} metricData={kpiData} data={Array.isArray(apiData) ? apiData : []} insights={insights}>
            <DateRangePicker
              date={selectedRange}
              onDateChange={setSelectedRange}
              onDateDefaultChange={onDateDefaultChange}
              disableFuture
              className=""
              weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6}
            />
          </InsightsTab>
        );
      // case "planning":
      //   return <PlanningTab />;
      // default:
      //   return apiResponse ? (
      //     <ActualDataTab data={apiResponse} aggregationType={aggregationType} />
      //   ) : (
      //     <div>No data available</div>
      //   );
    }
  };

  // Current date for KPI reporting
  const currentDate = new Date();
  const kpiTimePeriod = `${currentDate.toLocaleString("default", {
    month: "long",
  })} ${currentDate.getFullYear()}`;

  // New state variables for forecast settings
  const [forecastType, setForecastType] = useState("single");
  const [selectedModel, setSelectedModel] = useState("Prophet");
  const [selectedModels, setSelectedModels] = useState([]);
  const handleModelChange = (model) => {
    setSelectedModels((prevSelectedModels) => {
      if (prevSelectedModels.includes(model)) {
        return prevSelectedModels.filter((m) => m !== model);
      } else {
        return [...prevSelectedModels, model];
      }
    });
  };
  const [selectedHybridModels, setSelectedHybridModels] = useState([]);
  const [externalFactors, setExternalFactors] = useState({
    majorEvents: false,
    dynamicTarget: false,
    dynamicTargetStartDate: "",
    dynamicTargetEndDate: "",
    dynamicTargetDecreasePercentage: "",
  });

  const navigate = useNavigate();

  const handleNavigateToPlanningPage = () => {
    navigate("/planning");
  };

  const onWhatIfMove = (id) => {
    setActiveTab('What-If')
  };

  return (
    <SidebarProvider>
      <div className="flex h-screen w-full overflow-hidden">
        <CustomSidebar
          activeTab={activeTab}
          headerText={headerText}
          defaultConfig={defaultConfig}
          setActiveTab={setActiveTab}
          setOpenModal={setIsDrawerOpen}
          handleLogout={handleLogout}
        />


        <div className="flex flex-col flex-1 h-screen overflow-hidden bg-background text-foreground transition-all duration-300"
          style={{ zIndex: 0, overflowY: 'auto' }} // Ensure content is behind the floating sidebar and enable vertical scrolling
        >
          {/* Dashboard Header (Sticky) */}
          <div className="sticky top-0 z-1200 flex items-center justify-between py-1 px-1 bg-background  dark:border-gray-700 ">
            <div className="flex items-center gap-0 w-full">
              <DashboardHeader
                title={headerText} // Pass the dynamic title
                description={businessData?.description || "Walmart Fulfillment Services Dashboard"}
                businessDataList={businessDataList}
                setBusinessUnit={setBusinessData}
                lastUpdated={new Date().toLocaleDateString("en-GB")}
                defaultConfigId={defaultConfigId}
                userData={userData}
                activeTab={activeTab}
                businessData={businessData}
                handleLogout={handleLogout}
                options={businessDataList}
                setActiveMainTab={setActiveTab}
              />
            </div>
          </div>

          {activeTab === "businessPerformance" && (
            <div className="mx-2  ">
              <KPIMetrics businessId={businessData?.id || 0} businessName={businessData?.display_name || ''} weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6} selectedRange={selectedRange} defaultDecimalPrecisions={defaultDecimalPrecisions} parametersList={parametersList} kpiData={kpiData} kpiOldData={kpiOldData} loading={kpiLoading}>
                <DateRangePicker
                  date={selectedRange}
                  onDateChange={setSelectedRange}
                  onDateDefaultChange={onDateDefaultChange}
                  className=""
                  weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6}
                />
              </KPIMetrics>
            </div>
          )}

          {activeTab === "Occupany" && (
            <div className="mx-2">
              <ContactCenterApp weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6} />
            </div>
          )}

          {activeTab === "planning" && (
            <div className="mb-4">
              <TacticalPlanSelection navigateSimulator={onWhatIfMove} businessId={businessData?.id || 0} />
            </div>
          )}

          {activeTab === "Strategic" && (
            <div className="mb-4">
              <PlanSelection navigateSimulator={onWhatIfMove} businessId={businessData?.id || 0} />
            </div>
          )}
          {activeTab === "What-If" && (
            <div className="mb-4">
              {simulatorsLoading ? (
                <div className="space-y-4">
                  <LoadingSkeleton />
                </div>
              ) : (
                <Simulators simulatorsList={simulatorsList} />
              )}
            </div>
          )}
          {activeTab === "dowIntraday" && (
            <div className="mb-4">
              {intraDaysLoading ? (
                <div className="space-y-4">
                  <LoadingSkeleton />
                </div>
              ) : (
                <IntraDays intraTrigger={intraTrigger} setIntraTrigger={setIntraTrigger} businessId={businessData?.id || 0} weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6} metricData={kpiData} intradaysList={intraDaysList} />
              )}
            </div>
          )}
          {activeTab === "businessPerformance2" &&
            <div>
              <CapacityLook hideSummary options={[businessData?.display_name]} metricKeys={lobData.map(item => item.name)}>
                <DateRangePicker
                  date={selectedRange}
                  onDateChange={setSelectedRange}
                  onDateDefaultChange={onDateDefaultChange}
                  className=""
                  weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6}
                />
              </CapacityLook>
            </div>
          }

          {activeTab !== "businessPerformance" && (
            <div className="mb-4">
              {activeTab === "forecast" && (
                <Sheet open={isDrawerOpen} onOpenChange={setIsDrawerOpen}>
                  <SheetContent
                    side="right"
                    className="w-[1000px] h-screen bg-card text-card-foreground shadow-lg border border-border overflow-y-auto fixed top-0 right-0 z-[1000]" // MODIFIED: Use theme card/border
                  >
                    <SheetHeader>
                      <SheetTitle>Forecast Settings</SheetTitle>
                    </SheetHeader>
                    {loadingIcons.forecast ? (
                      <div className="space-y-4 p-4">
                        <LoadingSkeleton />
                        <div className="flex justify-center">
                          <CircularProgress />
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-4 =-2 p-2 mt-2">
                        {/* Forecast Period */}
                        <div>
                          <label className="block text-sm font-medium mb-2">Forecast Period</label>
                          <div className="flex items-center gap-2">
                            <input
                              type="number"
                              min="1"
                              value={tempForecastPeriod}
                              onChange={(e) => setTempForecastPeriod(Number(e.target.value))}
                              className="w-full border  px-2 py-1 text-sm dark:bg-gray-700 dark:text-white"
                            />
                            <select
                              value={tempAggregationType}
                              onChange={(e) => setTempAggregationType(e.target.value)}
                              className="border  px-2 py-1 text-sm dark:bg-gray-700 dark:text-white"
                            >
                              <option value="Weekly">Weeks</option>
                            </select>
                          </div>
                        </div>
                        <div className="space-y-4 border-t border-gray-200 dark:border-gray-700 pt-4">
                          <h3 className="text-sm font-medium">External Factors</h3>
                          <label className="flex items-center space-x-2">
                            <input
                              type="checkbox"
                              checked={externalFactors.majorEvents}
                              onChange={(e) =>
                                setExternalFactors({ ...externalFactors, majorEvents: e.target.checked })
                              }
                              className="dark:bg-gray-700 dark:text-white"
                            />
                            <span>Major Events</span>
                            <TooltipProvider>
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <div className="ml-2 flex items-center justify-center w-4 h-4 bg-gray-200 dark:bg-gray-700 rounded-full hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors">
                                    <Info className="h-3 w-3 text-gray-600 dark:text-gray-300" />
                                  </div>
                                </TooltipTrigger>
                                <TooltipContent
                                  side="right"
                                  className="bg-gray-800 dark:bg-gray-200 text-white dark:text-gray-800 text-xs p-1 rounded-md"
                                >
                                  <p>Includes holidays and major events like Christmas, New Year, etc.</p>
                                </TooltipContent>
                              </Tooltip>
                            </TooltipProvider>
                          </label>
                          <div>
                            <label className="flex items-center space-x-2">
                              <input
                                type="checkbox"
                                checked={externalFactors.dynamicTarget}
                                onChange={(e) =>
                                  setExternalFactors({ ...externalFactors, dynamicTarget: e.target.checked })
                                }
                                className="dark:bg-gray-700 dark:text-white"
                              />
                              <span>Dynamic Target</span>
                              <TooltipProvider>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <div className="ml-2 flex items-center justify-center w-4 h-4 bg-gray-200 dark:bg-gray-700 rounded-full hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors">
                                      <Info className="h-3 w-3 text-gray-600 dark:text-gray-300" />
                                    </div>
                                  </TooltipTrigger>
                                  <TooltipContent
                                    side="right"
                                    className="bg-gray-800 dark:bg-gray-200 text-white dark:text-gray-800 text-xs p-1 rounded-md"
                                  >
                                    <p>
                                      Specify a percentage amount for a specific date range to see its impact
                                      on actual and predicted data.
                                    </p>
                                  </TooltipContent>
                                </Tooltip>
                              </TooltipProvider>
                            </label>
                            {externalFactors.dynamicTarget && (
                              <div className="ml-6 space-y-2 mt-2">
                                <input
                                  type="date"
                                  value={externalFactors.dynamicTargetStartDate}
                                  onChange={(e) =>
                                    setExternalFactors({
                                      ...externalFactors,
                                      dynamicTargetStartDate: e.target.value,
                                    })
                                  }
                                  className="w-full border rounded px-2 py-1 text-sm dark:bg-gray-700 dark:text-white"
                                  placeholder="Start Date"
                                />
                                <input
                                  type="date"
                                  value={externalFactors.dynamicTargetEndDate}
                                  onChange={(e) =>
                                    setExternalFactors({
                                      ...externalFactors,
                                      dynamicTargetEndDate: e.target.value,
                                    })
                                  }
                                  className="w-full border rounded px-2 py-1 text-sm dark:bg-gray-700 dark:text-white"
                                  placeholder="End Date"
                                />
                                <input
                                  type="number"
                                  value={externalFactors.dynamicTargetDecreasePercentage}
                                  onChange={(e) =>
                                    setExternalFactors({
                                      ...externalFactors,
                                      dynamicTargetDecreasePercentage: e.target.value,
                                    })
                                  }
                                  className="w-full border rounded px-2 py-1 text-sm dark:bg-gray-700 dark:text-white"
                                  placeholder="Decrease Percentage"
                                />
                              </div>
                            )}
                          </div>
                        </div>
                        {/* Apply Changes Button */}
                        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                          <Button
                            size="sm"
                            className="w-full"
                            onClick={async () => {
                              setIsDrawerOpen(false);
                              await handleApplyChanges(tempForecastPeriod, modelType);
                              toast({
                                title: "Settings Applied",
                                description: "Forecast settings have been updated successfully.",
                              });
                            }}
                          >
                            Apply Changes
                          </Button>
                        </div>
                      </div>
                    )}
                  </SheetContent>
                </Sheet>
              )}

              <div className="p-2">
                {renderTabContent()}
              </div>
            </div>
          )}

          {activeTab === "businessPerformance2" &&
            <div>
              <CapacityLook hideSummary options={[businessData?.display_name]} metricKeys={lobData.map(item => item.name)}>
                <DateRangePicker
                  date={selectedRange}
                  onDateChange={setSelectedRange}
                  onDateDefaultChange={onDateDefaultChange}
                  className=""
                  weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6}
                />
              </CapacityLook>
            </div>
          }

        </div>
      </div>
      <footer className="fixed bottom-0 left-0 w-full py-4 border-t border-gray-200 dark:border-gray-700 text-center text-sm text-gray-600 dark:text-gray-300 bg-gray-100 dark:bg-[rgb(20,21,24)]">
        <p>© 2025 Aptino. All rights reserved.</p>
      </footer>
    </SidebarProvider>
  );
};
