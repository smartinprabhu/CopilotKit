import React, { useState, useMemo, useEffect, useDeferredValue } from 'react';
import { Calendar, BarChart3, Table, Download, Edit, Pencil } from 'lucide-react';
import { toZonedTime, format } from 'date-fns-tz';
import axios from "axios";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import ExcelJS from 'exceljs';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui2/button";
import { useToast } from "@/hooks/use-toast";
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui2/select";
import { eachDayOfInterval, startOfWeek, addDays, parseISO, addMinutes, getDay, isWithinInterval, startOfDay, getISOWeek, getISOWeekYear, startOfToday, isAfter, isBefore } from 'date-fns';
import { LineChart, Brush, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar, Label } from 'recharts';
import { Input } from "@/components/ui2/input";
import { Textarea } from "@/components/ui2/textarea";
import DateRangePicker from "./DateRangePicker";
import AuthService from "@/auth/utils/authService";

import { useChunkedChartData } from './UseChunkData'
import LoadingSkeleton from "./LoadingSkeleton";
import AppConfig from '../auth/config.js';

const WEBAPPAPIURL = `${AppConfig.API_URL}/api/v2/`;

interface IntradayData {
  lob: string;
  dow: string;
  date: string;
  halfHourData: number[];
}

function formatDateToUTCString(date: Date, formats: string): string {
  const utcDate = toZonedTime(date, 'UTC');
  return format(utcDate, formats);
}

const timeZone = 'Asia/Kolkata';

function getLocalDateString(utcDateStr: string, formatStr = 'yyyy-MM-dd', tz = timeZone): string {
  const fixedUtcString = utcDateStr.replace(' ', 'T') + 'Z';
  const zonedDate = toZonedTime(new Date(fixedUtcString), tz);
  return format(zonedDate, formatStr, { timeZone: tz });
}

const getDatesToShow = (start, end, filter) => {
  const adjustedStart = addDays(start, 1);
  const allDates = eachDayOfInterval({ start: adjustedStart, end }).map(date => ({
    date: format(date, 'yyyy-MM-dd'),
    dayOfWeek: format(date, 'EEEE'), // Full day name like "Monday"
    shortDay: format(date, 'EEE'),   // Short like "Mon"
    displayDate: format(date, 'dd MMM'),
  }));

  return filter
    ? allDates.filter(d => d.shortDay === filter)
    : allDates;
};

interface Filters {
  businessUnit: string;
  lineOfBusiness: string;
  dateRange: {
    start: string;
    end: string;
  };
  selectedDOW: string;
  aggregationType: 'daily' | 'monthly' | 'yearly';
  dailyFilters: string[];
  monthlyFilters: string[];
  yearlyFilters: number[];
  intradayFilters: { start: number; end: number };
}

interface DOWIntradayTabProps {
  detailsDefault: any;
  metricData: any[];
  navigateList: (id: any) => void;
  setIntraTrigger?: (id: any) => void;
  weekStartsOn?: 0 | 1 | 2 | 3 | 4 | 5 | 6;
}

const getWeekdays = (weekStarts = 0) => {
  const start = startOfWeek(new Date(), { weekStartsOn: weekStarts as 0 | 1 | 2 | 3 | 4 | 5 | 6 });
  return Array.from({ length: 7 }).map((_, i) => format(addDays(start, i), 'eee')); // 'eee' gives 'Mon', 'Tue', etc.
};

const TIME_INTERVALS = Array.from({ length: 48 }, (_, i) =>
  format(addMinutes(startOfDay(new Date()), i * 30), 'HH:mm')
);



const DOWIntradayTab: React.FC<DOWIntradayTabProps> = ({ setIntraTrigger, metricData, detailsDefault, weekStartsOn = 1 as 0 | 1 | 2 | 3 | 4 | 5 | 6, navigateList }) => {

  const { toast } = useToast();
  const [activeView, setActiveView] = useState<'table' | 'chart'>('chart');
  const [details, setDetails] = useState(detailsDefault);
  const [analysisType, setAnalysisType] = useState<'dow' | 'intraday'>('dow');
  const [chartType, setChartType] = useState<'line' | 'bar' | 'area'>('line');
  const [aggregationType, setAggregationType] = useState<'halfhour' | 'hourly' | 'daily' | 'weekly' | 'monthly'>('halfhour');
  const [filters, setFilters] = useState<Filters>({
    businessUnit: 'POS',
    lineOfBusiness: 'Phone',
    dateRange: {
      start: '2024-01-28',
      end: '2024-03-18',
    },
    selectedDOW: '',
    aggregationType: 'daily',
    dailyFilters: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
    monthlyFilters: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    yearlyFilters: [new Date().getFullYear()],
    intradayFilters: { start: 0, end: 24 },
  });

  const WEEKDAYS = useMemo(() => getWeekdays(weekStartsOn), [weekStartsOn]);

  useMemo(() => (
    setDetails(detailsDefault)
  ), [detailsDefault]);



  const [dowForecastLoading, setDowForecastLoading] = useState(false);
  const [dowForeCastData, setDowForeCastData] = useState([]);
  const [dowReferenceData, setDowReferenceData] = useState([]);
  const [dowReferenceLoading, setDowReferenceLoading] = useState(false);
  const [addOpen, setAddOpen] = useState(false);
  const [formLoading, setFormLoading] = useState(false);

  const [formData, setFormData] = useState<Record<string, any>>({ name: "", lob_id: "", remarks: "", reference_start_week: "", reference_end_week: "", forecast_start_week: "", forecast_end_week: "", referenceData: [] });

  useEffect(() => {
    if (details && details.id) {
      setFormData({
        name: details.name || "",
        lob_id: details.lob_id?.[1] || details.parameter_id?.[1], // Use ID (1900)
        remarks: details.remarks || "",
        reference_start_week: new Date(details.reference_start_week),
        reference_end_week: new Date(details.reference_end_week),
        forecast_start_week: new Date(details.forecast_start_week),
        forecast_end_week: new Date(details.forecast_end_week),
        referenceData: (details.data || []).map((item) => ({
          ...item,
          reference_start_week: new Date(item.reference_start_week),
          reference_end_week: new Date(item.reference_end_week),
        }))
      });
    }
  }, [details]);

  const handleFormInputChange = (name: string, value: any) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const onDateChange = (obj: { from: string; to: string }, field: any, index?: number) => {
    if (field.isMultiple) {
      const updated = [...(formData.referenceData || [])];
      updated[index!] = {
        ...updated[index!],
        [field.name]: obj.from,
        [field.endField]: obj.to,
      };
      setFormData((prev) => ({ ...prev, [field.name]: obj.from, [field.endField]: obj.to, referenceData: updated }));
    } else {
      setFormData((prev) => ({ ...prev, [field.name]: obj.from, [field.endField]: obj.to }));
    }
  };

  const onWeightageChange = (value: number, index: number) => {
    const updated = [...(formData.referenceData || [])];
    updated[index] = { ...updated[index], weightage: value };
    setFormData((prev) => ({ ...prev, referenceData: updated }));
  };

  const addNewForecastBlock = () => {
    setFormData((prev) => ({
      ...prev,
      referenceData: [
        ...(prev.referenceData || []),
        { reference_start_week: "", reference_end_week: "", weightage: 0 },
      ],
    }));
  };

  const distributeWeightage = () => {
    const count = formData.referenceData?.length || 0;
    if (count === 0) return;

    const base = Math.floor(100 / count);
    const remainder = 100 % count;

    const updated = formData.referenceData.map((item, index) => ({
      ...item,
      weightage: index === 0 ? base + remainder : base,
    }));

    setFormData((prev) => ({ ...prev, referenceData: updated }));
  };

  const removeForecastBlock = (index: number) => {
    const updated = [...formData.referenceData];
    updated.splice(index, 1);
    setFormData((prev) => ({ ...prev, referenceData: updated }));
  };


  const onDateDefaultChange = (field, index?: number) => {
    if (field.isMultiple && typeof index === "number") {
      const updated = [...(formData.referenceData || [])];
      updated[index] = {
        ...updated[index],
        [field.name]: null,
        [field.endField]: null,
      };
      setFormData((prev) => ({ ...prev, referenceData: updated }));
    } else {
      setFormData((prev) => ({
        ...prev,
        [field.name]: null,
        [field.endField]: null,
      }));
    }
  };



  const validateDate = (value: string, field: any): string | null => {
    const date = parseISO(value);
    const today = startOfToday();

    if (field.restrictPast && field.type === 'date' && isBefore(date, today)) {
      return "Date cannot be in the past.";
    }

    if (field.restrictFuture && field.type === 'date' && isAfter(date, today)) {
      return "Date cannot be in the future.";
    }

    return null;
  };

  const fields = useMemo(() => {
    const metricOptions = metricData.map((item) => ({
      id: item.lobId ?? item.parameterId,
      name: item.title,
    }));

    return [
      { name: "name", label: "Name", placeholder: "Name your DOW & Intraday..", type: "text", required: true },
      {
        name: "lob_id",
        label: "LOB",
        placeholder: "Select LOB",
        type: "select",
        options: metricOptions,
        required: true,
      },
      {
        name: "forecast_start_week",
        label: "Forecast Range",
        placeholder: "Select Forecast Range",
        type: "range",
        endField: "forecast_end_week",
        restrictFuture: false,
        required: true,
      },
      {
        name: "reference_start_week",
        isMultiple: true,
        label: "Reference Range",
        placeholder: "Select Reference Range",
        type: "range",
        endField: "reference_end_week",
        restrictFuture: true,
        required: true,
      },
      {
        name: "remarks",
        label: "Remarks",
        placeholder: "Enter the Remarks",
        type: "textarea",
        required: false,
      },
    ];
  }, [metricData]);

  async function updateForm(id: number, model: string, values: Record<string, unknown>) {
    const formData = new FormData();

    formData.append("ids", `[${id}]`);
    formData.append("model", model);
    formData.append("values", JSON.stringify(values)); // nested object as JSON string

    const url = `${WEBAPPAPIURL}write`;
    setFormLoading(true);
    try {
      const response = await axios.put(url, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      setFormLoading(false);
      setAddOpen(false);
      toast({
        title: "DOW Updated",
        description: "DOW filters have been updated successfully.",
      });
      setIntraTrigger(Math.random());
      return response.data;
    } catch (e) {
      setFormLoading(false);
      setAddOpen(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await updateForm(Number(details?.id), "dow_intraday", {
        name: formData.name,
        remarks: formData.remarks,
        parameter_id: metricData.find(item => item.parameterName === formData.lob_id && item.source === 'parameter')?.parameterId || false,
        lob_id: metricData.find(item => item.title === formData.lob_id && item.source === 'lob')?.lobId || false,
        forecast_start_week: formData.forecast_start_week,
        forecast_end_week: formData.forecast_end_week,
        data: formData.referenceData
      });
      const paramData = metricData.find(item => item.parameterName === formData.lob_id && item.source === 'parameter');
      const lobData = metricData.find(item => item.title === formData.lob_id && item.source === 'lob')
      setDetails((prev) => ({
        ...prev,
        name: formData.name,
        remarks: formData.remarks,
        parameter_id: paramData ? [paramData?.id, paramData?.title] : false,
        lob_id: lobData ? [lobData?.id, lobData?.title] : false,
        forecast_start_week: format(formData.forecast_start_week, 'yyyy-MM-dd'),
        forecast_end_week: format(formData.forecast_end_week, 'yyyy-MM-dd'),
        data: formData.referenceData.map((item) => ({
          ...item,
          reference_start_week: format(new Date(item.reference_start_week), 'yyyy-MM-dd'),
          reference_end_week: format(new Date(item.reference_end_week), 'yyyy-MM-dd'),
        })),
      }));

    } catch (err) {
      setFormLoading(false);
      console.error("API Error:", err);
    }

  };


  const fetchDowForecastData = async (id: number, lobId: number, parameterId: number, formattedFrom: string, formattedTo: string) => {

    let domain = `[["parameter_id.id", "=", ${parameterId}], ["lob_id.id", "=", ${lobId ? lobId : 'false'}],["business_unit_id","=",${id}],["date",">=", "${formatDateToUTCString(formattedFrom, 'yyyy-MM-dd HH:mm:ss')}"],["date","<=", "${formatDateToUTCString(formattedTo, 'yyyy-MM-dd HH:mm:ss')}"]]`

    if (lobId && !parameterId) {
      domain = `[["lob_id.id", "=", ${lobId ? lobId : 'false'}],["business_unit_id","=",${id}],["date",">=", "${formatDateToUTCString(formattedFrom, 'yyyy-MM-dd HH:mm:ss')}"],["date","<=", "${formatDateToUTCString(formattedTo, 'yyyy-MM-dd HH:mm:ss')}"]]`
    } else if (!lobId && parameterId) {
      domain = `[["parameter_id.id", "=", ${parameterId}],["lob_id.id", "=", false],["business_unit_id","=",${id}],["date",">=", "${formatDateToUTCString(formattedFrom, 'yyyy-MM-dd HH:mm:ss')}"],["date","<=", "${formatDateToUTCString(formattedTo, 'yyyy-MM-dd HH:mm:ss')}"]]`
    }

    const params = new URLSearchParams({
      offset: '0',
      domain: domain,
      fields: '["date","calendar_week_id","forecast_model_version","lower_bound","upper_bound","business_unit_id", "lob_id", "bpo_id", "week_no", "parameter_id", "value"]',
      model: 'forecast_feeds',
    });

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const fetchDowReferenceData = async (id: number, lobId: number, parameterId: number, formattedFrom: string, formattedTo: string) => {

    let domain = `[["entity_id.parameter_id.id", "=", ${parameterId}], ["entity_id.lob_id.id", "=", ${lobId ? lobId : 'false'}],["entity_id.business_unit_id","=",${id}],["date",">=", "${formatDateToUTCString(formattedFrom, 'yyyy-MM-dd HH:mm:ss')}"],["date","<=", "${formatDateToUTCString(formattedTo, 'yyyy-MM-dd HH:mm:ss')}"]]`

    if (lobId && !parameterId) {
      domain = `[["entity_id.lob_id.id", "=", ${lobId ? lobId : 'false'}],["entity_id.business_unit_id","=",${id}],["date",">=", "${formatDateToUTCString(formattedFrom, 'yyyy-MM-dd HH:mm:ss')}"],["date","<=", "${formatDateToUTCString(formattedTo, 'yyyy-MM-dd HH:mm:ss')}"]]`
    } else if (!lobId && parameterId) {
      domain = `[["entity_id.parameter_id.id", "=", ${parameterId}],["entity_id.lob_id.id", "=", false],["entity_id.business_unit_id","=",${id}],["date",">=", "${formatDateToUTCString(formattedFrom, 'yyyy-MM-dd HH:mm:ss')}"],["date","<=", "${formatDateToUTCString(formattedTo, 'yyyy-MM-dd HH:mm:ss')}"]]`
    }

    const params = new URLSearchParams({
      offset: '0',
      domain: domain,
      fields: '["value","entity_id","date"]',
      model: 'data_feeds_entity',
    });

    setDowReferenceLoading(true)

    try {
      const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
        headers: {
          Authorization: `Bearer ${AuthService.getAccessToken()}`
        },
      });
      setDowReferenceLoading(false);
      return response.data;
    } catch (error) {
      setDowReferenceLoading(false);
      console.error('Error fetching data:', error);
    }
  };


  useEffect(() => {
    const runFetch = async () => {
      setDowForeCastData([])
      if (details && details?.id) {
        try {
          const formattedFrom = details.forecast_start_week ? format(details.forecast_start_week, 'yyyy-MM-dd') : '';
          const formattedTo = details.forecast_end_week ? format(details.forecast_end_week, 'yyyy-MM-dd') : '';
          setDowForecastLoading(true)
          const result = await fetchDowForecastData(details?.business_unit?.[0], details?.lob_id?.[0] || false, details?.parameter_id?.[0] || false, formattedFrom, formattedTo); // call the function
          setDowForecastLoading(false);
          setDowForeCastData(result);

        } catch (err) {
          console.error("Failed to fetch:", err);
          setDowForecastLoading(false);
          setDowForeCastData([])
        }
      }
    };
    const runFetch2 = async () => {
      setDowReferenceData([])
      if (details && details?.id) {
        try {
          const startRef = details.data && details.data.length ? details.data[0]?.reference_start_week : details.reference_start_week;
          const endRef = details.data && details.data.length ? details.data[details.data.length - 1]?.reference_end_week : details.reference_end_week;

          const formattedFrom = startRef ? startRef : '';
          const formattedTo = endRef ? endRef : '';
          const result = await fetchDowReferenceData(details?.business_unit?.[0], details?.lob_id?.[0] || false, details?.parameter_id?.[0] || false, formattedFrom, formattedTo); // call the function
          setDowReferenceData(result);

        } catch (err) {
          console.error("Failed to fetch:", err);
          setDowReferenceData([])
        }
      }
    };

    runFetch();
    runFetch2();
  }, [details]);


  const [summary, setSummary] = useState({});
  const [dailyTotals, setDailyTotals] = useState({});
  const [dailyAverages, setDailyAverages] = useState({});

  useEffect(() => {
    if (!dowReferenceData?.length || !details?.data?.length) return;

    // Step 1: Initialize summary object with all time intervals and weekdays
    const summaryObj = {};
    TIME_INTERVALS.forEach((interval) => {
      summaryObj[interval] = WEEKDAYS.reduce((acc, day) => {
        acc[day] = 0;
        return acc;
      }, {});
    });

    // Step 2: Process each reference block separately like (AVERAGEIF(...)*weight)
    details?.data.forEach((block) => {
      const { reference_start_week, reference_end_week, weightage } = block;

      const blockWeight = weightage / 100; // Convert 50 to 0.5 if needed
      const entriesInRange = dowReferenceData.filter((item) => {
        const d = parseISO(item.date);
        return isWithinInterval(d, {
          start: parseISO(reference_start_week),
          end: parseISO(reference_end_week),
        });
      });

      // Group values by timeKey and weekday
      const grouped = {};
      entriesInRange.forEach((item) => {
        const date = parseISO(item.date);
        const timeKey = format(date, 'HH:mm');
        const weekday = WEEKDAYS[getDay(date)];

        if (!grouped[timeKey]) grouped[timeKey] = {};
        if (!grouped[timeKey][weekday]) grouped[timeKey][weekday] = [];
        grouped[timeKey][weekday].push(item.value || 0);
      });

      // Add weighted average to summary
      Object.entries(grouped).forEach(([timeKey, weekdayMap]) => {
        Object.entries(weekdayMap).forEach(([weekday, values]) => {
          const avg = values.reduce((sum, val) => sum + val, 0) / values.length;
          const weightedAvg = avg * blockWeight;

          if (!summaryObj[timeKey]) {
            summaryObj[timeKey] = WEEKDAYS.reduce((acc, day) => {
              acc[day] = 0;
              return acc;
            }, {});
          }

          summaryObj[timeKey][weekday] += weightedAvg;
        });
      });
    });

    // Step 3: Compute totals per weekday
    const totals = WEEKDAYS.reduce((acc, day) => {
      acc[day] = TIME_INTERVALS.reduce((sum, interval) => {
        return sum + (summaryObj[interval]?.[day] || 0);
      }, 0);
      return acc;
    }, {});

    // Step 4: Compute averages per weekday
    const averages = WEEKDAYS.reduce((acc, day) => {
      const total = totals[day];
      const count = TIME_INTERVALS.length;
      acc[day] = parseFloat((total / count).toFixed(2));
      return acc;
    }, {});

    // Step 5: Update state
    setSummary(summaryObj);
    setDailyTotals(totals);
    setDailyAverages(averages);
  }, [dowReferenceData, details, WEEKDAYS]);

  const intradayForecast = useMemo(() => {
    if (!dowForeCastData?.length || !Object.keys(summary).length || !Object.keys(dailyTotals).length) {
      return [];
    }

    const forecastStart = new Date(getLocalDateString(details?.forecast_start_week));
    const forecastEnd = new Date(getLocalDateString(details?.forecast_end_week));

    return dowForeCastData.flatMap((forecast) => {
      const totalWeeklyValue = forecast.value;
      const weekStartDate = startOfWeek(new Date(getLocalDateString(forecast.date)), { weekStartsOn }); // Monday

      const dailyTotalSum = WEEKDAYS.reduce(
        (sum, day) => sum + (dailyTotals[day] || 0),
        0
      ) || 1; // avoid divide-by-zero

      return WEEKDAYS.flatMap((day, dayIndex) => {
        const dayTotalPercent = (dailyTotals[day] || 0) / dailyTotalSum;
        const dayTotal = totalWeeklyValue * dayTotalPercent;
        const dateForDay = addDays(weekStartDate, dayIndex);

        if (dateForDay < forecastStart || dateForDay > forecastEnd) {
          return [];
        }

        const formattedDate = format(dateForDay, 'yyyy-MM-dd');

        const hourlyTotalSum = TIME_INTERVALS.reduce(
          (sum, interval) => sum + (summary[interval]?.[day] || 0),
          0
        ) || 1;

        return TIME_INTERVALS.map((interval) => {
          const hourPercent = (summary[interval]?.[day] || 0) / hourlyTotalSum;
          const value = parseFloat((dayTotal * hourPercent).toFixed(2));

          return {
            date: formattedDate,
            interval,
            value,
          };
        });
      });
    });
  }, [dowForeCastData, dailyTotals, details, WEEKDAYS]);

  const filteredForecast = useMemo(() => {
    if (!filters.selectedDOW) return intradayForecast;

    return intradayForecast.filter((item) => {
      const day = format(parseISO(item.date), 'EEE'); // e.g., "Mon", "Tue"
      return day === filters.selectedDOW;
    });
  }, [intradayForecast, filters.selectedDOW]);


  const { chartData, xAxisLabel, overallAverage, loading } = useChunkedChartData(
    filteredForecast,
    aggregationType,
    analysisType
  );




  const businessUnits = ['POS'];
  const linesOfBusiness = [
    'Phone',
    'Chat',
    'Case Type 1',
    'Case Type 2',
    'Case Type 3',
    'Case Type 4',
    'Case Type 5',
    'Case Type 6',
  ];
  const daysOfWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];


 const formatDateToWeekLabel = (dateStr: string): string => {
  try {
    if (!dateStr) {
      return "";
    }
    const date = parseISO(dateStr);
    const week = getISOWeek(date);
    const formattedDate = format(date, "dd/MM/yyyy");
    return `WK${week} (${formattedDate})`;
  } catch {
    return "";
  }
};



  // Handle filter changes
  const handleFilterChange = (key: keyof Filters, value: any) => {
    setFilters((prev) => {
      const newFilters = { ...prev, [key]: value };
      if (key === 'dateRange') {
        newFilters.selectedDOW = '';
      }
      return newFilters;
    });
  };


  const exportCombinedSideBySideXLSX = () => {
    if (!intradayForecast.length || !summary || !dailyTotals) return;

    const weekdays = WEEKDAYS;
    const uniqueDates = [...new Set(intradayForecast.map(d => d.date))].sort();
    const dowCols = 1 + weekdays.length + 1; // Time Interval + 7 days + Average
    const forecastCols = 1 + uniqueDates.length;

    const workbook = new ExcelJS.Workbook();
    const ws = workbook.addWorksheet('Forecast Export');

    const forecastRange = formatDateToWeekLabel(details.forecast_start_week) + ' to ' + formatDateToWeekLabel(details.forecast_end_week);
    const referenceRanges = details?.data?.map(ref =>
      `${formatDateToWeekLabel(ref.reference_start_week)} to ${formatDateToWeekLabel(ref.reference_end_week)} (${ref.weightage}%)`
    ) || [];

    const forecastStartCol = dowCols + 4;
    const forecastSpan = forecastCols;
    const forecastLabelCol = dowCols + 3;

    // === 1. Title Row ===
    ws.addRow([]);
    ws.mergeCells(1, 1, 1, dowCols);
    ws.getCell(1, 1).value = 'DOW and Intraday Distribution';
    ws.getCell(1, 1).alignment = { horizontal: 'left' };
    ws.getCell(1, 1).font = { bold: true };

    ws.mergeCells(1, forecastLabelCol, 1, forecastLabelCol + forecastSpan - 1);
    ws.getCell(1, forecastLabelCol).value = 'Daily and Intraday Forecast';
    ws.getCell(1, forecastLabelCol).alignment = { horizontal: 'left' };
    ws.getCell(1, forecastLabelCol).font = { bold: true };

    // === 2. Reference & Forecast Range Row(s) ===
    const baseRowIdx = 2;
    ws.getCell(baseRowIdx, 1).value = 'Reference Range:';
    ws.getCell(baseRowIdx, 1).font = { bold: true };
    ws.getCell(baseRowIdx, 2).value = referenceRanges[0] || '';
    ws.getCell(baseRowIdx, 2).alignment = { wrapText: false };

    for (let i = 1; i < referenceRanges.length; i++) {
      const refRow = ws.addRow([]);
      refRow.getCell(2).value = referenceRanges[i];
      refRow.getCell(2).alignment = { wrapText: false };
    }

    ws.getCell(baseRowIdx, forecastLabelCol).value = 'Forecast Range:';
    ws.getCell(baseRowIdx, forecastLabelCol).font = { bold: true };
    ws.mergeCells(baseRowIdx, forecastStartCol, baseRowIdx, forecastStartCol + forecastSpan - 1);
    ws.getCell(baseRowIdx, forecastStartCol).value = forecastRange;
    ws.getCell(baseRowIdx, forecastStartCol).alignment = { wrapText: false };

    ws.addRow([]); // Blank line

    // === 3. DOW % Table ===
    const dowTotal = weekdays.reduce((sum, day) => sum + (dailyTotals[day] || 0), 0);
    const headerRow = ws.addRow(['', ...weekdays]);
    headerRow.font = { bold: true };

    const dowPercentValues = weekdays.map(day =>
      dowTotal ? (dailyTotals[day] / dowTotal) : 0
    );
    const dowPercentRow = ws.addRow(['DOW %', ...dowPercentValues]);

    for (let i = 2; i <= weekdays.length + 1; i++) {
      dowPercentRow.getCell(i).numFmt = '0.00%';
    }

    ws.addRow([]); // Blank line

    // === 4. Headers (Side-by-Side) ===
    const dowHeader = ['Time Interval', ...weekdays, 'Average'];
    const forecastHeader = ['Time Interval', ...uniqueDates];
    const combinedHeaderRow = ws.addRow([...dowHeader, '', '', ...forecastHeader]);
    combinedHeaderRow.font = { bold: true };

    // === 5. DOW Intraday Rows (only for side-by-side) ===
    const dowIntradayRows = TIME_INTERVALS.map(interval => {
      const row = [interval];
      let rowTotal = 0;
      weekdays.forEach(day => {
        const value = summary?.[interval]?.[day] || 0;
        const dayTotal = dailyTotals?.[day] || 1;
        const percent = (value / dayTotal);
        row.push(percent); // numeric 0.xx
        rowTotal += percent;
      });
      row.push(rowTotal / weekdays.length); // average
      return row;
    });

    const grandDayTotal = weekdays.reduce((total, day) => total + (dailyTotals[day] || 0), 0);
    const dowTotalRow = ['Average'];
    weekdays.forEach(day => {
      const total = dailyTotals[day] || 0;
      const pct = grandDayTotal ? (total / grandDayTotal) : 0;
      dowTotalRow.push(pct); // numeric 0.xx
    });
    dowTotalRow.push(1); // 100%
    dowIntradayRows.push(dowTotalRow);

    // === 6. Forecast Rows ===
    const forecastRows = TIME_INTERVALS.map(interval => {
      const row = [interval];
      uniqueDates.forEach(date => {
        const val = intradayForecast.find(d => d.date === date && d.interval === interval)?.value || 0;
        row.push(parseFloat(val.toFixed(2)));
      });
      return row;
    });

    const forecastDailyTotalRow = ['Daily Total'];
    uniqueDates.forEach(date => {
      const total = TIME_INTERVALS.reduce((sum, interval) => {
        const val = intradayForecast.find(d => d.date === date && d.interval === interval)?.value || 0;
        return sum + val;
      }, 0);
      forecastDailyTotalRow.push(parseFloat(total.toFixed(2)));
    });
    forecastRows.push(forecastDailyTotalRow);

    // === 7. Add Rows Side-by-Side ===
    const maxRows = Math.max(dowIntradayRows.length, forecastRows.length);
    for (let i = 0; i < maxRows; i++) {
      const left = i < dowIntradayRows.length ? dowIntradayRows[i] : Array(dowCols).fill('');
      const right = i < forecastRows.length ? forecastRows[i] : Array(forecastCols).fill('');
      const row = ws.addRow([...left, '', '', ...right]);

      // Format DOW percentage columns
      for (let j = 2; j <= weekdays.length + 2; j++) {
        if (typeof row.getCell(j).value === 'number') {
          row.getCell(j).numFmt = '0.00%';
        }
      }
    }

    // === 8. Column Styling ===
    ws.columns.forEach(col => {
      col.width = 16;
    });

    // === 9. Download XLSX ===
    workbook.xlsx.writeBuffer().then(buffer => {
      const blob = new Blob([buffer], {
        type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'DOW_Intraday_Forecast.xlsx';
      a.click();
      URL.revokeObjectURL(url);
    });
  };

  const exportCombinedSideBySideCSV = () => {
    if (!intradayForecast.length || !summary || !dailyTotals) return;

    const weekdays = WEEKDAYS;
    const uniqueDates = [...new Set(intradayForecast.map(d => d.date))].sort();
    const dowCols = 1 + weekdays.length + 1; // Time Interval + 7 days + Average
    const forecastCols = 1 + uniqueDates.length;

    const forecastRange = formatDateToWeekLabel(details.forecast_start_week) + ' to ' + formatDateToWeekLabel(details.forecast_end_week);
    const referenceRanges = details?.data?.map(ref =>
      `${formatDateToWeekLabel(ref.reference_start_week)} to ${formatDateToWeekLabel(ref.reference_end_week)} (${ref.weightage}%)`
    ) || [];

    const csvLines = [];

    // === 1. Title Row ===
    csvLines.push([
      'DOW and Intraday Distribution',
      ...Array(dowCols - 1).fill(''),
      '',
      '',
      'Daily and Intraday Forecast',
      ...Array(forecastCols - 1).fill('')
    ].join(','));

    // === 2. Reference & Forecast Range Rows ===
    csvLines.push(['Reference Range:', referenceRanges[0] || '', ...Array(dowCols - 2).fill(''), '', '', 'Forecast Range:', forecastRange]);
    for (let i = 1; i < referenceRanges.length; i++) {
      csvLines.push(['', referenceRanges[i]]);
    }

    csvLines.push(''); // Blank line

    // === 3. DOW % Table ===
    const dowTotal = weekdays.reduce((sum, day) => sum + (dailyTotals[day] || 0), 0);
    const dowPercentValues = weekdays.map(day =>
      dowTotal ? (dailyTotals[day] / dowTotal) : '0.0000'
    );

    csvLines.push(['', ...weekdays].join(','));
    csvLines.push(['DOW %', ...dowPercentValues.map(p => `${(parseFloat(p) * 100)} %`)].join(','));

    csvLines.push(''); // Blank line

    // === 4. Headers (Side-by-Side) ===
    const dowHeader = ['Time Interval', ...weekdays, 'Average'];
    const forecastHeader = ['Time Interval', ...uniqueDates];
    csvLines.push([...dowHeader, '', '', ...forecastHeader].join(','));

    // === 5. DOW Intraday Rows ===
    const dowIntradayRows = TIME_INTERVALS.map(interval => {
      const row = [interval];
      let rowTotal = 0;
      weekdays.forEach(day => {
        const value = summary?.[interval]?.[day] || 0;
        const dayTotal = dailyTotals?.[day] || 1;
        const percent = (value / dayTotal);
        row.push(`${(percent * 100)} %`);
        rowTotal += percent;
      });
      row.push(`${((rowTotal / weekdays.length) * 100)} %`);
      return row;
    });

    const grandDayTotal = weekdays.reduce((total, day) => total + (dailyTotals[day] || 0), 0);
    const dowTotalRow = ['Average'];
    weekdays.forEach(day => {
      const total = dailyTotals[day] || 0;
      const pct = grandDayTotal ? (total / grandDayTotal) : 0;
      dowTotalRow.push(`${(pct * 100)} %`);
    });
    dowTotalRow.push('100.00 %');
    dowIntradayRows.push(dowTotalRow);

    // === 6. Forecast Rows ===
    const forecastRows = TIME_INTERVALS.map(interval => {
      const row = [interval];
      uniqueDates.forEach(date => {
        const val = intradayForecast.find(d => d.date === date && d.interval === interval)?.value || 0;
        row.push(parseFloat(val));
      });
      return row;
    });

    const forecastDailyTotalRow = ['Daily Total'];
    uniqueDates.forEach(date => {
      const total = TIME_INTERVALS.reduce((sum, interval) => {
        const val = intradayForecast.find(d => d.date === date && d.interval === interval)?.value || 0;
        return sum + val;
      }, 0);
      forecastDailyTotalRow.push(parseFloat(total));
    });
    forecastRows.push(forecastDailyTotalRow);

    // === 7. Combine Side-by-Side Rows ===
    const maxRows = Math.max(dowIntradayRows.length, forecastRows.length);
    for (let i = 0; i < maxRows; i++) {
      const left = i < dowIntradayRows.length ? dowIntradayRows[i] : Array(dowCols).fill('');
      const right = i < forecastRows.length ? forecastRows[i] : Array(forecastCols).fill('');
      csvLines.push([...left, '', '', ...right].join(','));
    }

    // === 8. Export CSV ===
    const csvContent = csvLines.join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'DOW_Intraday_Forecast.csv';
    a.click();
    URL.revokeObjectURL(url);
  };




  // Render DOW Table View
  const renderDOWTableView = () => {
    const filteredWeekdays = filters.selectedDOW
      ? [filters.selectedDOW]  // e.g., 'Sun'
      : WEEKDAYS;
    const grandTotal = TIME_INTERVALS.reduce((total, interval) => {
      const row = summary[interval];
      return total + filteredWeekdays.reduce((sum, day) => sum + (row && row[day] ? row[day] : 0), 0);
    }, 0);
    const grandDayTotal = filteredWeekdays.reduce((total, day) => total + (dailyTotals[day] || 0), 0);


    return (
      <div className="rounded-lg overflow-hidden">
        <div className="overflow-auto h-full" style={{ maxHeight: 'calc(100vh - 200px)' }}>
          <table className="w-full">
            <thead>
              <tr className="bg-muted/60 sticky top-0 z-20">
                <th className="px-4 py-3 text-left text-sm font-medium text-foreground border-r border-border sticky left-0 bg-muted/60 z-30">
                  Time Interval
                </th>
                {filteredWeekdays.map((day, index) => (
                  <th
                    key={day}
                    className="px-3 py-3 bg-muted/60 text-center text-sm font-medium text-foreground border-r border-border min-w-[120px]"
                  >
                    <div className="flex flex-col">
                      <span>{day}</span>
                    </div>
                  </th>
                ))}
                <th className="px-3 py-3 text-center text-sm font-medium text-foreground border-r border-border min-w-[120px] sticky right-0 bg-muted/60 z-30">
                  <div className="flex flex-col">
                    <span>Hourly Total</span>
                    <span className="text-xs text-muted-foreground">Avg & %</span>
                  </div>
                </th>
              </tr>
            </thead>
            <tbody>
              {TIME_INTERVALS.map(interval => {
                const row = summary[interval];
                const rowTotal = filteredWeekdays.reduce((sum, day) => sum + (row?.[day] ?? 0), 0);
                const avgPercent = filteredWeekdays.reduce((sum, day) => {
                  const total = dailyTotals[day] || 1;
                  return sum + ((row?.[day] ?? 0) / total * 100);
                }, 0) / filteredWeekdays.length;

                return (
                  <tr key={interval} className="bg-muted/50">
                    <td className="px-4 py-2 text-sm font-medium text-foreground border-r border-border sticky left-0 bg-muted z-10">
                      {interval}
                    </td>
                    {filteredWeekdays.map(day => {
                      const value = row?.[day] ?? 0;
                      const total = dailyTotals[day] || 1;
                      const percent = (value / total) * 100;

                      return (
                        <td
                          className="px-3 py-2 text-center text-sm border-r border-border"
                          key={day}
                        >
                          <div className="flex flex-col">
                            <span className="text-foreground">{value.toFixed(0)}</span>
                            <span className="text-xs text-primary">{percent.toFixed(2)}%</span>
                          </div>
                        </td>
                      );
                    })}
                    <td className="px-3 py-2 text-center text-sm border-r border-border sticky right-0 bg-muted z-10">
                      <div className="flex flex-col">
                        <span className="text-muted-foreground font-semibold">{rowTotal.toFixed(0)}</span>
                        <span className="text-xs text-primary">{avgPercent.toFixed(2)}%</span>
                      </div>
                    </td>
                  </tr>
                );
              })}
              <tr className="bg-muted/60 sticky bottom-0 z-20 border-t-2 border-border">
                <td className="px-4 py-3 text-sm font-bold text-foreground border-r border-border sticky left-0 bg-muted/60 z-30">
                  Total
                </td>
                {filteredWeekdays.map((dowData, dowIndex) => {
                  const total = dailyTotals[dowData] || 0;
                  const percentOfGrand = grandDayTotal > 0 ? (total / grandDayTotal) * 100 : 0;

                  return (
                    <td key={dowIndex} className="px-3 bg-muted/60  py-3 text-center text-sm border-r border-border">
                      <div className="flex flex-col">
                        <span className="text-foreground font-semibold">{total.toFixed(0)}</span>
                        <span className="text-xs text-primary">{percentOfGrand.toFixed(2)}%</span>
                      </div>
                    </td>
                  );
                })}
                <td className="px-3 py-3 text-center bg-muted/60 text-sm border-r border-border">
                  <div className="flex flex-col">
                    <span className="text-foreground font-semibold">{grandTotal.toFixed(0)}</span>
                    <span className="text-xs text-primary">100.0%</span>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

    );
  };

  // Render Intraday Table View
  const renderIntradayTableView = () => {
    const datesToShow = getDatesToShow(new Date(getLocalDateString(details?.forecast_start_week)), new Date(getLocalDateString(details?.forecast_end_week)), filters.selectedDOW);
    return (
      <div className="rounded-lg overflow-hidden">
        <div className="overflow-auto h-full" style={{ maxHeight: 'calc(100vh - 200px)' }}>
          <table className="w-full">
            <thead>
              <tr className="bg-muted/60 sticky top-0 z-20">
                <th className="px-4 py-3 text-left text-sm font-medium text-foreground border-r border-border sticky left-0 bg-muted/60 z-30">
                  Time Interval
                </th>
                {datesToShow.map((dateInfo, index) => (
                  <th
                    key={index}
                    className="px-3 py-3 text-center text-sm font-medium text-foreground border-r border-border min-w-[80px]"
                  >
                    <div className="flex flex-col">
                      <span className="text-xs text-muted-foreground">{dateInfo.dayOfWeek.slice(0, 3)}</span>
                      <span>{dateInfo.displayDate}</span>
                    </div>
                  </th>
                ))}
                <th className="px-3 py-3 text-center text-sm font-medium text-foreground border-r border-border min-w-[120px] sticky right-0 bg-muted/60 z-30">
                  <div className="flex flex-col">
                    <span>Hourly Total</span>
                    <span className="text-xs text-muted-foreground">Sum & %</span>
                  </div>
                </th>
              </tr>
            </thead>
            <tbody>
              {TIME_INTERVALS.map((timeSlot, timeIndex) => (
                <tr key={timeIndex} className={timeIndex % 2 === 0 ? 'bg-muted' : 'bg-muted/70'}>
                  <td className="px-4 py-2 text-sm font-medium text-foreground border-r border-border sticky left-0 bg-muted z-10">
                    {timeSlot}
                  </td>
                  {datesToShow.map((dateInfo, dateIndex) => {
                    const match = intradayForecast.find(
                      (d) => d.date === dateInfo.date && d.interval === timeSlot
                    );
                    const value = match ? match.value : 0;
                    return (
                      <td
                        key={dateIndex}
                        className="px-3 py-2 text-center text-sm text-muted-foreground border-r border-border"
                      >
                        {value.toFixed(0)}
                      </td>
                    );
                  })}
                  <td className="px-3 py-2 text-center text-sm border-r border-border sticky right-0 bg-muted z-10">
                    <span className="text-muted-foreground font-semibold">
                      {datesToShow.reduce((sum, dateInfo) => {
                        const match = intradayForecast.find(
                          (d) => d.date === dateInfo.date && d.interval === timeSlot
                        );
                        return parseFloat(Number(sum)) + parseFloat(Number(match ? match.value : 0));
                      }, 0).toFixed(0)}
                    </span>
                  </td>
                </tr>
              ))}
              <tr className="bg-muted/60 sticky border-t-2 border-border">
                <td className="px-4 py-3 sticky text-sm font-bold text-foreground border-r border-border sticky left-0 bg-muted/60 z-30">
                  Daily Total
                </td>
                {datesToShow.map((dateInfo, colIndex) => {
                  let sum = 0;
                  TIME_INTERVALS.forEach((interval) => {
                    const item = intradayForecast.find(
                      (d) => d.date === dateInfo.date && d.interval === interval
                    );
                    sum += item?.value || 0;
                  });
                  return (
                    <td
                      key={colIndex}
                      className="px-3 py-3 text-center text-sm text-foreground font-semibold border-r border-border"
                    >
                      {sum.toFixed(0)}
                    </td>
                  );
                })}
                <td className="px-3 py-3 text-center text-sm border-r border-border sticky right-0 bg-muted/60 z-30">
                  <span className="text-foreground font-bold">
                    {datesToShow.reduce((grandSum, dateInfo) => {
                      const dailySum = TIME_INTERVALS.reduce((sum, interval) => {
                        const item = intradayForecast.find(
                          (d) => d.date === dateInfo.date && d.interval === interval
                        );
                        return sum + (item?.value || 0);
                      }, 0);
                      return grandSum + dailySum;
                    }, 0).toFixed(0)}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

    );
  };


  // Render DOW Chart View
  const renderDOWChartView = () => {
    if (!dailyTotals || Object.keys(dailyTotals).length === 0) {
      return <div className="flex items-center justify-center h-96 text-gray-400">No data available</div>;
    }

    // Convert object to chart array
    const chartData = Object.entries(dailyTotals).map(([dow, value]) => ({
      dow,
      avgTotalVolume: Number(value) / TIME_INTERVALS.length, // if value is total not avg
    }));

    // Filter if a specific DOW is selected
    const dataToShow = filters.selectedDOW
      ? chartData.filter((item) => item.dow === filters.selectedDOW)
      : chartData;

    return (
      <div className="bg-background rounded-lg p-6">
        <h3 className="text-lg font-semibold text-foreground mb-2">
          Average Cases Distribution by Day of Week
        </h3>
        <p className="text-sm text-muted-foreground">
          Selected LOB: {details.parameter_id?.[1] || details.lob_id?.[1] || ''} | Date Range:{" "}
          {formatDateToWeekLabel(
            details.data?.length > 0
              ? details.data[0].reference_start_week
              : details.reference_start_week
          )}{" "}
          to{" "}
          {formatDateToWeekLabel(
            details.data?.length > 0
              ? details.data[details.data.length - 1].reference_end_week
              : details.reference_end_week
          )}
        </p>

        <ResponsiveContainer width="100%" height={400}>
          <BarChart
            data={dataToShow}
            margin={{ top: 20, right: 10, left: 10, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" /> {/* slate-300 */}

            <XAxis
              dataKey="dow"
              stroke="#64748b" // slate-500
              tick={{ fill: '#64748b', fontSize: 12 }}
              label={{
                value: 'Day of the Week',
                position: 'insideBottom',
                offset: -40,
                style: { fill: '#64748b' },
              }}
            />

            <YAxis
              stroke="#64748b"
              tick={{ fill: '#64748b', fontSize: 12 }}
              label={{
                value: 'Cases',
                angle: -90,
                position: 'insideLeft',
                offset: 0,
                style: { fill: '#64748b' },
              }}
            />

            <Tooltip
              contentStyle={{
                backgroundColor: '#f1f5f9', // slate-100
                border: '1px solid #cbd5e1', // slate-300
              }}
              labelStyle={{
                color: '#3b82f6', // slate-900
              }}
              itemStyle={{
                color: '#3b82f6', // slate-800
              }}
              formatter={(value) => [Number(value).toFixed(0), 'Average Cases']}
            />


            <Legend
              wrapperStyle={{
                color: '#475569', // slate-600
                paddingBottom: 10,
              }}
            />

            <Bar dataKey="avgTotalVolume" fill="#3b82f6" name="Average Cases" barSize={40} /> {/* blue-500 */}
          </BarChart>
        </ResponsiveContainer>

      </div>
    );
  };

  // Render Intraday Chart View
  const renderIntradayChartView = () => {
    // generateIntradayData.filter((item) => !filters.selectedDOW || item.dow === filters.selectedDOW);
    let yAxisLabel = 'Volume';

    // Add overall average to each data point for the average line
    const chartDataWithAverage = chartData.map(item => ({
      ...item,
      average: overallAverage
    }));

    const minVolume = chartData.length > 0 ? Math.min(...chartData.map((d) => d.value)) : 0;
    const maxVolume = chartData.length > 0 ? Math.max(...chartData.map((d) => d.value)) : 0;

    // Calculate dynamic width based on data points and aggregation type
    const getChartWidth = () => {
      const baseWidth = 800;
      const minWidth = 1200;
      const maxWidth = 3000;

      let pointWidth = 50;
      if (aggregationType === 'halfhour') pointWidth = 30;
      else if (aggregationType === 'hourly') pointWidth = 40;
      else if (aggregationType === 'daily') pointWidth = 60;
      else if (aggregationType === 'weekly') pointWidth = 100;
      else if (aggregationType === 'monthly') pointWidth = 150;

      const calculatedWidth = Math.max(minWidth, chartData.length * pointWidth);
      return Math.min(maxWidth, calculatedWidth);
    };
    const renderChart = () => {
      if (!chartData.length || !intradayForecast.length) {
        return <div className="flex items-center justify-center h-96 text-gray-400">No data available for the selected filters</div>;
      }
      if (loading) {
        return <div className="flex items-center justify-center h-96 text-gray-400">Loading..</div>;
      }

      const chartWidth = getChartWidth();
      const showXAxisLabels = true; // chartData.length <= 50; // Only show labels if not too many points
      const labelInterval = Math.max(0, Math.floor(chartData.length / 15)); // Show max 15 labels

      const renderSharedElements = () => (
        <>
          <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
          <XAxis
            dataKey="label"
            stroke="#94a3b8"
            // interval={labelInterval}
            angle={-45}
            textAnchor="end"
            height={80}
            fontSize={12}
            tick={showXAxisLabels ? { fontSize: 10, fill: '#94a3b8' } : false}
            tickFormatter={(label) => {
              // If label is ISO string like "2025-07-29T20:30:00Z"
              try {
                const date = parseISO(label); // or new Date(label)
                return `${format(date, 'MMM d')}\n${format(date, 'HH:mm')}`;
                // Alternative: `${format(date, 'EEE')} ${format(date, 'HH:mm')}` for "Tue 20:30"
              } catch {
                return label;
              }
            }}
          />
          <YAxis
            stroke="#94a3b8"
            yAxisId="left"
            orientation="left"
            fontSize={12}
            label={{ value: yAxisLabel, angle: -90, position: 'insideLeft', textAnchor: 'middle', fill: '#94a3b8' }}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
            labelStyle={{ color: '#cbd5e1' }}
            formatter={(value) => Number(value).toFixed(0)}
          />
          <Legend wrapperStyle={{ color: '#cbd5e1' }} />
          <Brush
            dataKey="label"
            height={30}
            stroke="#64748b"
            startIndex={0}
            endIndex={Math.min(48, chartDataWithAverage.length - 1)}
            travellerWidth={10}
            tickFormatter={(label) => {
              try {
                const date = parseISO(label);
                return format(date, 'MMM d');
              } catch {
                return label;
              }
            }}
          />
        </>
      );

      // Chart Factory
      const renderChartByType = () => {
        const baseProps = {
          data: chartDataWithAverage,
          margin: { top: 20, right: 30, left: 20, bottom: 60 },
        };

        switch (chartType) {
          case 'line':
            return (
              <LineChart {...baseProps}>
                {renderSharedElements()}
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="value"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={{ r: chartData.length > 100 ? 0 : 3 }}
                  name="Volume"
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="average"
                  stroke="#ef4444"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                  name={`Overall Average (${Math.round(overallAverage).toLocaleString()})`}
                />
              </LineChart>
            );

          case 'area':
            return (
              <AreaChart {...baseProps}>
                <defs>
                  <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                {renderSharedElements()}
                <Area
                  yAxisId="left"
                  type="monotone"
                  dataKey="value"
                  stroke="#3b82f6"
                  fillOpacity={1}
                  fill="url(#areaGradient)"
                  name="Volume"
                />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="average"
                  stroke="#ef4444"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                  name={`Overall Average (${Math.round(overallAverage).toLocaleString()})`}
                />
              </AreaChart>
            );

          case 'bar':
          default:
            return (
              <BarChart {...baseProps}>
                {renderSharedElements()}
                <Bar yAxisId="left" dataKey="value" fill="#3b82f6" name="Volume" />
                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="average"
                  stroke="#ef4444"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                  name={`Overall Average (${Math.round(overallAverage).toLocaleString()})`}
                />
              </BarChart>
            );
        }
      };


      return (
        <div style={{ width: '100%', overflowX: 'auto', overflowY: 'hidden' }}>
          <ResponsiveContainer width={chartWidth} height={450}>
            {renderChartByType()}
          </ResponsiveContainer>
        </div>
      );
    };

    return (
      <div className="bg-background rounded-lg p-6">
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold text-foreground mb-2">
                {aggregationType === 'halfhour'
                  ? 'Intraday Volume Distribution'
                  : aggregationType === 'hourly'
                    ? 'Hourly Volume Distribution'
                    : aggregationType === 'daily'
                      ? 'Daily Volume Distribution'
                      : aggregationType === 'weekly'
                        ? 'Weekly Volume Distribution'
                        : 'Monthly Volume Distribution'}
              </h3>
              <p className="text-sm text-muted-foreground">
                Selected LOB: {details.parameter_id?.[1] || details.lob_id?.[1] || ''} | Date Range:{" "}
                {formatDateToWeekLabel(details.forecast_start_week)} to{" "}
                {formatDateToWeekLabel(details.forecast_end_week)}
                {filters.selectedDOW && ` | Filtered: ${filters.selectedDOW}`}
                {overallAverage > 0 && ` | Avg: ${Math.round(overallAverage).toLocaleString()}`}
              </p>
            </div>

            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <label className="text-sm text-muted-foreground">Chart:</label>
                <select
                  value={chartType}
                  onChange={(e) => setChartType(e.target.value as 'line' | 'bar' | 'area')}
                  className="bg-muted border border-border rounded px-3 py-1 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                >
                  <option value="line">Line</option>
                  <option value="bar">Bar</option>
                  <option value="area">Area</option>
                </select>
              </div>
              <div className="flex items-center space-x-2">
                <label className="text-sm text-muted-foreground">Aggregation:</label>
                <select
                  value={aggregationType}
                  onChange={(e) =>
                    setAggregationType(
                      e.target.value as 'halfhour' | 'hourly' | 'daily' | 'weekly' | 'monthly'
                    )
                  }
                  className="bg-muted border border-border rounded px-3 py-1 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                >
                  <option value="halfhour">Intraday</option>
                  <option value="hourly">Hourly</option>
                  <option value="daily">Daily</option>
                  <option value="weekly">Weekly</option>
                  <option value="monthly">Monthly</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-card p-4 rounded-lg">{renderChart()}</div>

        {chartData.length > 0 && (
          <div className="mt-4 flex items-center justify-between text-sm text-muted-foreground">
            <p>
              {xAxisLabel} | Total {yAxisLabel}:{' '}
              {chartData.reduce((sum, item) => sum + item.value, 0).toLocaleString()}
            </p>
            <p>
              Data Points: {chartData.length} | Range: {minVolume.toLocaleString()} -{' '}
              {maxVolume.toLocaleString()} | Avg: {Math.round(overallAverage).toLocaleString()}
            </p>
          </div>
        )}
      </div>

    );
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="border-b border-border">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between mb-4">
            <div>
              {details && details?.name && (
                <h1 className="text-xl font-semibold text-foreground">
                  <span
                    className="ml-1 cursor-pointer"
                    onClick={() => navigateList(details?.id)}
                  >
                    DOW & Intradays
                  </span>
                  <span className="ml-2 opacity-7" style={{ opacity: 0.7 }}>
                    /
                    {' '}
                    {`${details?.name} (${details.create_uid && details.create_uid?.[1] || ''})`}
                  </span>
                </h1>
              )}
            </div>
            <div className="flex items-center space-x-4">
              {/* Toggle: DOW / Intraday */}
              <div className="flex items-center bg-muted rounded-lg p-1">
                <button
                  onClick={() => setAnalysisType('dow')}
                  className={`px-3 py-1 text-sm rounded transition-colors ${analysisType === 'dow'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                    }`}
                >
                  DOW
                </button>
                <button
                  onClick={() => {
                    setAnalysisType('intraday');
                    setAggregationType('halfhour');
                  }}
                  className={`px-3 py-1 text-sm rounded transition-colors ${analysisType === 'intraday'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                    }`}
                >
                  Intraday
                </button>
              </div>

              {/* Toggle: Table / Chart */}
              <div className="flex items-center bg-muted rounded-lg p-1">
                <button
                  onClick={() => setActiveView('table')}
                  className={`px-3 py-1 text-sm rounded transition-colors ${activeView === 'table'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                    }`}
                >
                  <Table className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setActiveView('chart')}
                  className={`px-3 py-1 text-sm rounded transition-colors ${activeView === 'chart'
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                    }`}
                >
                  <BarChart3 className="w-4 h-4" />
                </button>
              </div>

              <TooltipProvider>
                <UITooltip>
                  <TooltipTrigger asChild>
                    <div>
                      <Popover>
                        <PopoverTrigger asChild>
                          <Button variant="outline" size="icon">
                            <Download className="h-4 w-4" />
                          </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-[200px] p-0" align="end">
                          <div className="p-2">
                            <p className="text-sm font-medium">Select Export Type</p>
                          </div>
                          <Separator />
                          <div className="p-1">
                            <button
                              onClick={exportCombinedSideBySideCSV}
                              className="w-full text-left p-2 text-sm rounded hover:bg-accent disabled:opacity-50 disabled:pointer-events-none"
                            >
                              CSV
                            </button>
                            <button
                              onClick={exportCombinedSideBySideXLSX}
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

          </div>
          <div className="flex items-center space-x-4 flex-wrap">
            <div className="flex items-center space-x-2">
              <label className="text-sm text-muted-foreground">LOB:</label>
              <span className="text-sm">{details.parameter_id?.[1] || details.lob_id?.[1] || ''}</span>
            </div>

            <div className="flex items-center space-x-2">
              <label className="text-sm text-muted-foreground">Forecast Range:</label>
              <span className="text-sm">
                {formatDateToWeekLabel(details.forecast_start_week)} to {formatDateToWeekLabel(details.forecast_end_week)}
              </span>
            </div>

            <TooltipProvider>
              <UITooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center space-x-2 cursor-pointer">
                    <label className="text-sm text-muted-foreground">Reference Range:</label>
                    <span className="text-sm">
                      {formatDateToWeekLabel(details.data?.[0]?.reference_start_week ?? details.reference_start_week)} to{' '}
                      {formatDateToWeekLabel(details.data?.[details.data.length - 1]?.reference_end_week ?? details.reference_end_week)}
                    </span>
                  </div>
                </TooltipTrigger>

                <TooltipContent
                  side="bottom"
                  className="bg-popover text-popover-foreground text-sm rounded-md shadow-lg p-4 w-64 text-center"
                >
                  <div className="space-y-2">
                    <p className="font-semibold underline">Reference Ranges</p>
                    {details.data && Array.isArray(details.data) && details.data?.map((item, idx) => (
                      <div key={idx} className="border border-border rounded p-2 text-sm">
                        <p className="font-medium">Weightage: {item.weightage}%</p>
                        <p>
                          {item.reference_start_week ? format(new Date(item.reference_start_week), 'dd MMM yyyy') : ''} →{' '}
                          {item.reference_end_week ? format(new Date(item.reference_end_week), 'dd MMM yyyy') : ''}
                        </p>
                      </div>
                    ))}
                  </div>
                </TooltipContent>
              </UITooltip>
            </TooltipProvider>

            <div className="flex items-center space-x-2">
              <Pencil className="w-4 h-4 ml-1 cursor-pointer" onClick={() => setAddOpen(true)} />
            </div>
          </div>

        </div>
      </div>
      <Sheet open={addOpen} onOpenChange={setAddOpen}>
        <SheetContent
          side="right"
          className="!w-[700px] !max-w-none h-screen bg-card text-card-foreground shadow-lg border border-border overflow-y-auto fixed top-0 right-0 z-[1000]"
        >
          <SheetHeader>
            <SheetTitle>{details?.name || ''}</SheetTitle>
          </SheetHeader>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-4 mt-3">
              {/* Forecast Period */}
              <div className="p-2">

                <div className="">
                  {fields.map((field) => {
                    if (field.type === 'group' && Array.isArray(field.fields)) {
                      return (
                        <div key={field.name} className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-3">
                          {field.fields.map((subField) => (
                            <div key={subField.name}>
                              <label htmlFor={subField.name} className="text-gray-800 dark:text-slate-300">
                                {subField.label}
                              </label>
                              <Input
                                id={subField.name}
                                type={subField.type}
                                value={formData[subField.name] || ''}
                                required={subField.required}
                                placeholder={subField.placeholder}
                                onChange={(e) => {
                                  const value = e.target.value;
                                  const error = validateDate(value, subField);

                                  if (error) {
                                    alert(error); // Or replace with toast or inline error display
                                    return;
                                  }

                                  handleFormInputChange(subField.name, value);
                                }}
                                className="w-full bg-white border-gray-300 text-black placeholder:text-gray-500
                                                                                              dark:bg-slate-700 dark:border-slate-600 dark:text-white dark:placeholder:text-slate-400"
                              />
                            </div>
                          ))}
                        </div>
                      );
                    }

                    return (
                      <div key={field.name} className={field.type === 'textarea' ? 'md:col-span-2 mb-2' : 'mb-3'}>
                        <label htmlFor={field.name} className="text-gray-800 dark:text-slate-300">
                          {field.label} {field.required && <span className="text-red-500">*</span>}
                        </label>

                        {field.type === 'select' ? (
                          <Select
                            value={formData[field.name] || ''}
                            onValueChange={(value) => handleFormInputChange(field.name, value)}
                          >
                            <SelectTrigger className="bg-white border-gray-300 text-black placeholder:text-gray-500
                                                                                                          dark:bg-slate-700 dark:border-slate-600 dark:text-white dark:placeholder:text-slate-400">
                              <SelectValue placeholder={`Select ${field.label}`} />
                            </SelectTrigger>
                            <SelectContent className="z-[9999] bg-white border-gray-300 text-black dark:bg-slate-700 dark:border-slate-600 dark:text-white">
                              {field.options?.map((option) => (
                                <SelectItem
                                  key={option.id}
                                  value={option.name}
                                  className="text-black hover:bg-gray-100
                                                                                              dark:text-white dark:hover:bg-slate-600"
                                >
                                  {option.name}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        ) : field.type === 'textarea' ? (
                          <Textarea
                            id={field.name}
                            value={formData[field.name] || ''}
                            onChange={(e) => handleFormInputChange(field.name, e.target.value)}
                            className="w-full bg-white border-gray-300 text-black placeholder:text-gray-500
                                                                                          dark:bg-slate-700 dark:border-slate-600 dark:text-white dark:placeholder:text-slate-400"
                            rows={3}
                            placeholder={field.placeholder}
                          />
                        ) : field.type === 'range' ? (
                          <>
                            {field.isMultiple ? (
                              <>
                                {(formData.referenceData || []).map((item: any, index: number) => (
                                  <div key={index} className="mb-2 p-0 mt-2">
                                    <div className="flex items-end gap-4 flex-wrap">

                                      {/* Index + Date Picker */}
                                      <div className="flex items-end gap-2">
                                        <span className="ml-[-1.3rem]">{`${index + 1})`}</span>
                                        <DateRangePicker
                                          date={{
                                            from: item[field.name] || '',
                                            to: item[field.endField] || '',
                                          }}
                                          onDateChange={(obj) => onDateChange(obj, field, index)}
                                          onDateDefaultChange={() => onDateDefaultChange(field, index)}
                                          className=""
                                          isForm
                                          disableFuture={field.restrictFuture}
                                          weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6}
                                        />
                                      </div>

                                      {/* Weightage Field */}
                                      <div className="w-[200px]">
                                        <label className="text-sm text-gray-800 dark:text-slate-300 mb-1 block">Weightage (%)</label>
                                        <Input
                                          type="text"
                                          inputMode="numeric"
                                          pattern="[0-9]*"
                                          value={item.weightage}
                                          onChange={(e) => {
                                            const val = e.target.value;
                                            if (/^\d*$/.test(val)) {
                                              onWeightageChange(Number(val), index);
                                            }
                                          }}
                                          placeholder="Weightage (%)"
                                          className="w-full bg-white border border-gray-300 text-black dark:bg-slate-700 dark:border-slate-600 dark:text-white"
                                        />
                                      </div>

                                      {/* Remove Button */}
                                      <div className="pt-6">
                                        <button
                                          type="button"
                                          onClick={() => removeForecastBlock(index)}
                                          className="text-red-500 text-sm hover:underline"
                                        >
                                          Remove
                                        </button>
                                      </div>

                                    </div>
                                  </div>

                                ))}
                                <button
                                  type="button"
                                  onClick={addNewForecastBlock}
                                  className="text-blue-600 dark:text-blue-400 text-sm hover:underline"
                                >
                                  + Add Range
                                </button>
                                {(formData.referenceData || []).length > 1 && (
                                  <button
                                    type="button"
                                    onClick={distributeWeightage}
                                    className="text-green-600 dark:text-green-400 text-sm hover:underline ml-3 mt-2"
                                  >
                                    Distribute to 100%
                                  </button>
                                )}
                              </>
                            ) : (
                              <DateRangePicker
                                date={{
                                  from: formData[field.name] || '',
                                  to: formData[field.endField] || '',
                                }}
                                onDateChange={(obj) => onDateChange(obj, field)}
                                onDateDefaultChange={() => onDateDefaultChange(field)}
                                className=""
                                isForm
                                disableFuture={field.restrictFuture}
                                weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6}
                              />
                            )}
                          </>

                        ) : (
                          <Input
                            id={field.name}
                            type={field.type}
                            value={formData[field.name] || ''}
                            required={field.required}
                            placeholder={field.placeholder}
                            onChange={(e) => handleFormInputChange(field.name, e.target.value)}
                            className="w-full bg-white border-gray-300 text-black placeholder:text-gray-500
                                                                                          dark:bg-slate-700 dark:border-slate-600 dark:text-white dark:placeholder:text-slate-400"
                          />
                        )}
                      </div>
                    );
                  })}

                </div>
              </div>
              {/* Apply Changes Button */}
              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                <Button
                  size="sm"
                  className="w-full mb-2"
                  type="submit"
                  disabled={formLoading}
                >
                  {formLoading ? 'Updating..' : 'Update'}
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  className="w-full"
                  type="submit"
                  onClick={() => setAddOpen(false)}
                  disabled={formLoading}
                >
                  Cancel
                </Button>
              </div>
            </div>
          </form>
        </SheetContent>
      </Sheet>
      <div className="px-6 py-4 bg-slate-850 border-b border-slate-700">
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-2">
            <label className="text-sm text-muted-foreground font-medium">{analysisType === 'dow' ? 'Filter DOW:' : 'Filter Days:'}</label>
            <div className="flex space-x-1">
              <button
                onClick={() => handleFilterChange('selectedDOW', '')}
                className={`px-3 py-1 text-xs rounded transition-colors ${filters.selectedDOW === '' ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                  }`}
              >
                All
              </button>
              {WEEKDAYS.map((day) => (
                <button
                  key={day}
                  onClick={() => handleFilterChange('selectedDOW', day)}
                  className={`px-3 py-1 text-xs rounded transition-colors ${filters.selectedDOW === day ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                    }`}
                >
                  {day}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
      <div className="p-6">
        {(dowForecastLoading || dowReferenceLoading) && (
          <div className="space-y-4">
            <LoadingSkeleton />
          </div>
        )}
        {(!dowForecastLoading && !dowReferenceLoading) && (
          <div>
            {analysisType === 'dow' ? activeView === 'table' ? renderDOWTableView() : renderDOWChartView() : activeView === 'table' ? renderIntradayTableView() : renderIntradayChartView()}
          </div>
        )}
      </div>
    </div>
  );
};

export default DOWIntradayTab; 