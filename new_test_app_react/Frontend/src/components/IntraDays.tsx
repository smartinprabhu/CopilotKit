import React, { useState, useMemo, useEffect } from 'react';
import { Calendar, Plus } from 'lucide-react';
import { format, parseISO, getISOWeek, getISOWeekYear, startOfToday, isAfter, isBefore } from 'date-fns';
import { Chip } from "@mui/material"
import { Card, CardContent } from "@/components/ui/card";
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Button } from "@/components/ui2/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import axios from 'axios';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui2/select";
import { Input } from "@/components/ui2/input";
import { Textarea } from "@/components/ui2/textarea";
import { Label } from "@/components/ui2/label";
import DateRangePicker from "./DateRangePicker";
import { useToast } from "@/hooks/use-toast";

import DOWIntradayTab from "./DOWIntradayTab";

import NoDataFullPage from "@/components/noDataPage";
import AuthService from "@/auth/utils/authService";

import AppConfig from '../auth/config.js';

type CardItem = {
  id: string;
  name: string;
  description: string;
  start_date: string;
  end_date: string;
};

const forecastData = [
  {
    name: "Tech Support Q3",
    reference: "Weekly DOW + Intraday Trends",
    state: "active",
    create_uid: [1, "hjaja"],
    create_date: "2025-07-01 10:30:00",
    forecast_range_start: "2025-07-15",
    forecast_range_end: "2025-08-15",
    reference_range_start: "2025-06-01",
    reference_range_end: "2025-07-01",
  },
  {
    name: "Billing Summer Surge",
    reference: "High Volume Escalation Analysis",
    state: "draft",
    create_uid: [2, "sanjay"],
    create_date: "2025-06-28 15:45:00",
    forecast_range_start: "2025-07-10",
    forecast_range_end: "2025-08-10",
    reference_range_start: "2025-06-01",
    reference_range_end: "2025-07-01",
  },
  {
    name: "Returns & Refunds H1 Review",
    reference: "Historical Refund Load",
    state: "archived",
    create_uid: [3, "maria"],
    create_date: "2025-05-10 09:00:00",
    forecast_range_start: "2025-05-15",
    forecast_range_end: "2025-06-15",
    reference_range_start: "2025-04-01",
    reference_range_end: "2025-05-01",
  },
  {
    name: "Tech Support Holiday Load",
    reference: "Holiday Surge Simulation",
    state: "active",
    create_uid: [4, "lee"],
    create_date: "2025-07-10 13:20:00",
    forecast_range_start: "2025-08-01",
    forecast_range_end: "2025-09-01",
    reference_range_start: "2025-06-15",
    reference_range_end: "2025-07-15",
  },
  {
    name: "International Billing Trends",
    reference: "Global Billing DOW Baseline",
    state: "draft",
    create_uid: [5, "amina"],
    create_date: "2025-07-12 08:10:00",
    forecast_range_start: "2025-07-20",
    forecast_range_end: "2025-08-20",
    reference_range_start: "2025-06-10",
    reference_range_end: "2025-07-10",
  },
];




const formatUtcToLocal = (utcString: string): string => {
  const date = new Date(utcString + 'Z'); // Parses UTC ISO string
  return format(date, 'dd MMM yyyy hh:mm a') // Formats in local time
}

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

const CardList = ({ metricData, intraTrigger, intradaysList, setIntraTrigger, businessId, weekStartsOn = 1 as 0 | 1 | 2 | 3 | 4 | 5 | 6, }) => {


  const [activeTab, setActiveTab] = useState(false);
  const [details, setDetails] = useState({});
  const [addOpen, setAddOpen] = useState(false);

  type ReferenceItem = {
    reference_start_week: string;
    reference_end_week: string;
    weightage: number;
  };

  const validateReferenceData = (referenceData: ReferenceItem[]): boolean => {
    return referenceData.every(
      (item) =>
        item.reference_start_week !== "" &&
        item.reference_end_week !== ""
    );
  };

  const [formLoading, setFormLoading] = useState(false);

  const { toast } = useToast();

  const WEBAPPAPIURL = `${AppConfig.API_URL}/api/v2/`;

  const [formData, setFormData] = useState<Record<string, any>>({ name: "", lob_id: "", remarks: "", reference_start_week: "", reference_end_week: "", forecast_start_week: "", forecast_end_week: "", referenceData: [] });

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


  const onViewPlan = (obj) => {
    setDetails(obj);
    setActiveTab(true);
  };

  const onClosePlan = (id) => {
    setDetails({});
    setActiveTab(false);
    setIntraTrigger(false)
  };

  useEffect(() => {
    if (intraTrigger && intradaysList.length > 0) {
      onViewPlan(intradaysList[0])
    }
  }, [intradaysList]);

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


  type ForecastFormData = {
    name: string;
    remarks: string;
    lob_id: number | false;
    parameter_id: number | false;
    forecast_start_week: string;
    forecast_end_week: string;
    reference_start_week: string;
    reference_end_week: string;
    business_unit: number;
    client_id: number;
    data: string;
  };


  async function submitForecastForm(model: string, values: ForecastFormData) {
    const formData = new FormData();

    formData.append("model", model);
    formData.append("values", JSON.stringify(values)); // nested object as JSON string

    const url = `${WEBAPPAPIURL}create`;

    const response = await axios.post(url, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
        Authorization: `Bearer ${AuthService.getAccessToken()}`
      },
    });

    return response.data;
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormLoading(true);
    try {
      const response = await submitForecastForm("dow_intraday", {
        name: formData.name,
        remarks: formData.remarks,
        parameter_id: metricData.find(item => item.parameterName === formData.lob_id && item.source === 'parameter')?.parameterId || false,
        lob_id: metricData.find(item => item.title === formData.lob_id && item.source === 'lob')?.lobId || false,
        forecast_start_week: formData.forecast_start_week,
        forecast_end_week: formData.forecast_end_week,
        reference_start_week: formData.reference_start_week,
        reference_end_week: formData.reference_end_week,
        business_unit: businessId,
        client_id: AuthService.getCompanyId(),
        data: formData.referenceData
      });
      toast({
        title: "Record Created",
        description: `DOW & Intraday has been created successfully.`,
      });
      setFormLoading(false);
      setAddOpen(false);
      setIntraTrigger(Math.random());
    } catch (err) {
      setFormLoading(false);
      console.error("API Error:", err);
    }

  };

  return (
    <div className={activeTab ? '' : 'space-y-3 p-4'}>
      {!activeTab && (
        <h5 className="text-xl mt-0 flex items-center">
          DOW & Intradays
          <div className='ml-auto flex items-center gap-2'>
            <Button variant="outline" size="sm" onClick={() => setAddOpen(true)}>
              <Plus className="mr-1" /> Create DOW & Intraday
            </Button>
          </div>
          <Sheet open={addOpen} onOpenChange={setAddOpen}>
            <SheetContent
              side="right"
              className="!w-[700px] !max-w-none h-screen bg-card text-card-foreground shadow-lg border border-border overflow-y-auto fixed top-0 right-0 z-[1000]"
            >
              <SheetHeader>
                <SheetTitle>Create DOW & Intraday</SheetTitle>
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
                                  <Label htmlFor={subField.name} className="text-gray-800 dark:text-slate-300">
                                    {subField.label}
                                  </Label>
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
                            <Label htmlFor={field.name} className="text-gray-800 dark:text-slate-300">
                              {field.label} {field.required && <span className="text-red-500">*</span>}
                            </Label>

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
                      disabled={formLoading || !formData.referenceData.length || !validateReferenceData(formData.referenceData) || !formData.forecast_start_week || !formData.forecast_end_week}
                    >
                      {formLoading ? 'Saving' : 'Show DOW & Intraday'}
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
        </h5>
      )}
      {!activeTab && (
        <div className="flex align-items-center gap-4">
          <div className="w-full sm:w-1/2 h-auto overflow-y-auto" style={{ maxHeight: '100vh' }}>
            {intradaysList && intradaysList?.length > 0 && intradaysList.map((card) => (
              <div
                key={card.id}
                onClick={() => onViewPlan(card)}
                className="cursor-pointer relative shadow-md p-3 mb-3 h-auto executive-card transition duration-200 ease-in-out"
              >
                {/* Top-right: Play Button */}
                <button
                  className="absolute right-4 top-1/2 -translate-y-1/2 z-10 p-2 rounded-full bg-primary shadow"
                  title="View"
                >
                  <Calendar className="w-6 h-6 bg-primary" />
                </button>

                {/* Title & Description */}
                <div className="space-y-2 pr-16">
                  <p className="text-sm m-0 text-gray-600 dark:text-gray-200 flex items-center" style={{ opacity: 0.5 }}>{card.reference} -  {card.parameter_id && card.parameter_id?.[1] || card.lob_id && card.lob_id?.[1] || ''}</p>
                  <h3
                    className="text-lg mt-0 font-semibold text-gray-900 dark:text-white"
                    style={{ lineHeight: '15px' }}>
                    {card.name} | <span className="text-sm">{card.create_uid && card.create_uid?.[1] || ''}</span> | <span className="text-sm">{card.create_date ? formatUtcToLocal(card.create_date) : ''}</span>
                  </h3>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-2 gap-2 mt-2">
                  <div className="">
                    <p className="text-xs text-gray-600 dark:text-gray-200" style={{ opacity: 0.5 }}>Forecast Range</p>
                    <p className="text-sm mb-0 text-gray-600 dark:text-gray-200">{formatDateToWeekLabel(card.forecast_start_week)} to {formatDateToWeekLabel(card.forecast_end_week)}</p>
                  </div>
                  <TooltipProvider>
                    <UITooltip>
                      <TooltipTrigger asChild>
                        <div className="cursor-pointer">
                          <p className="text-xs text-gray-600 dark:text-gray-200" style={{ opacity: 0.5 }}>Reference Range</p>
                          <p className="text-sm mb-0 text-gray-600 dark:text-gray-200">{formatDateToWeekLabel(card.data && card.data.length > 0 ? card.data[0].reference_start_week : card.reference_start_week)} {formatDateToWeekLabel(card.data && card.data.length > 0 ? card.data[0].reference_start_week : card.reference_start_week) ? 'to' : ''} {formatDateToWeekLabel(card.data && card.data.length > 0 ? card.data[card.data.length - 1].reference_end_week : card.reference_end_week)}</p>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent
                        side="bottom"
                        className="bg-white dark:bg-gray-800 text-sm text-gray-800 dark:text-gray-200 rounded-md shadow-lg p-4 w-64 text-center"
                      >
                        <div className="space-y-2">
                          <p className="font-semibold underline">Reference Ranges</p>
                          {Array.isArray(card.data) && card.data.length > 0 && card.data.map((item, idx) => (
                            <div key={idx} className="border border-gray-200 dark:border-gray-700 rounded p-2">
                              <p className="font-medium">Weightage: {item.weightage}%</p>
                              <p>
                                {item.reference_start_week ? format(new Date(item.reference_start_week), 'dd MMM yyyy') : ''} →{" "}
                                {item.reference_end_week ? format(new Date(item.reference_end_week), 'dd MMM yyyy') : ''}
                              </p>
                            </div>
                          ))}
                        </div>
                      </TooltipContent>
                    </UITooltip>
                  </TooltipProvider>
                </div>
              </div>
            ))}
            {!(intradaysList && intradaysList?.length > 0) && (
              <NoDataFullPage message="No DOW & Intradays are available." />
            )}
          </div>
          <div className="w-full sm:w-1/2 h-auto">
            <Card className="p-5 w-full h-auto transition duration-200 ease-in-out" style={{ borderLeft: '3px solid grey' }}>
              <h4 className="text-xl">DOW & Intraday Scenarios</h4>
              <p className="text-lg mt-2 mb-2">
                To improve support operations forecasting, analyze weekly data by <strong>Day-of-Week (DOW)</strong> and <strong>Intraday</strong> patterns.
              </p>
              <p className="text-lg mt-2 mb-2">
                DOW trends help identify high- and low-volume days, enabling better resource planning. Intraday analysis breaks down daily volume into time blocks, revealing peak hours for smarter shift and queue management.
              </p>
              <p className="text-lg mt-2 mb-2">
                This dual-layered approach ensures accurate forecasts, optimized staffing, and improved SLA performance—turning raw data into actionable operational insights.
              </p>

              <p className="text-lg font-semibold mt-4 mb-2">Steps:</p>
              <div className="p-2">
                <ul className="list-disc list-inside space-y-1 text-base">
                  <li>Select Line of Business (LOB): Choose the specific support function (e.g., Tech Support, Billing) to ensure focused and relevant analysis.</li>
                  <li>Define Forecast Range: Set the future time period for which support volumes need to be forecasted.</li>
                  <li>Define Reference Range: Select the past date range that will serve as the baseline to identify historical trends.</li>
                  <li>Perform DOW Split: Analyze support volume by day of the week to identify high- and low-volume days.</li>
                  <li>Perform Intraday Split: Break down each day’s data into time intervals (e.g., hourly) to find peak and quiet periods.</li>
                  <li>Extract Insights and Apply to Forecast: Use DOW and Intraday patterns to model and distribute expected ticket volumes across the forecast range.</li>
                </ul>
              </div>
            </Card>
          </div>
        </div>
      )}
      {activeTab && (
        <DOWIntradayTab setIntraTrigger={setIntraTrigger} metricData={metricData} weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6} detailsDefault={details} navigateList={onClosePlan} />
      )}

    </div>
  );
};

export default CardList;

