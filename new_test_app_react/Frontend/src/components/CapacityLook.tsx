// HeadcountTrend.tsx
import React, { useMemo, useState } from 'react';
import {
  LineChart, ReferenceLine, Cell, Line, Bar, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ComposedChart, ResponsiveContainer,
} from 'recharts';
import { Card, CardContent } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import AskAIButton from './AskAIButton'

const options = ["Option A", "Option B", "Option C"];



type SummaryCardProps = {
  title: string;
  color: string;
  value: number;
  trendData?: { value: number }[];
};

const SummaryCard = ({ title, value, color, trendData = [] }: SummaryCardProps) => (
  <Card className="bg-background text-black dark:text-white p-4 w-full shadow-lg">
    <CardContent className="flex flex-col gap-2">
      <div>
        <span className="text-sm text-gray-400">{title}</span>
        <div className="text-2xl font-semibold">{value}</div>
      </div>

      {/* Tiny Line Chart (Sparkline) */}
      <div className="h-6">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={trendData}>
            <Line
              type="monotone"
              dataKey="value"
              stroke={color} // Tailwind's green-500
              strokeWidth={2}
              dot={false}
            />
            <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none' }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </CardContent>
  </Card>
);


const HeadcountTrend = ({ hideSummary, options, children, metricKeys }) => {

    const [selectedTraces, setSelectedTraces] = useState<string[]>(["inventory"]);
    const [traces, setTraces] = useState<any[]>([]);

         useMemo(() => {
              setTraces(metricKeys)
              setSelectedTraces(hideSummary && metricKeys && metricKeys.length > 0 ? metricKeys : []);
          }, [hideSummary, metricKeys]);
    
      const handleTraceChange = (value: string[]) => {
        setSelectedTraces(value);
      };

const generateRandomData = () => {
  const weeks = Array.from({ length: 10 }, (_, i) => `WK ${i + 1}`);
  const optionData: Record<string, any[]> = {};
  const summary = {
    required: Array(10).fill(0),
    actual: Array(10).fill(0),
    ou: Array(10).fill(0),
  };

  options.forEach((option) => {
    const data = weeks.map((week, i) => {
      const required = Math.floor(Math.random() * 10) + 20;
      const actual = Math.floor(Math.random() * 10) + 18;
      const ou = actual - required;
      summary.required[i] += required;
      summary.actual[i] += actual;
      summary.ou[i] += ou;
      return {
        week,
        Required: required,
        Actual: actual,
        ['O/U']: ou,
      };
    });
    optionData[option] = data;
  });

  const totalSummary = {
    required: summary.required.reduce((a, b) => a + b, 0),
    actual: summary.actual.reduce((a, b) => a + b, 0),
    ou: summary.ou.reduce((a, b) => a + b, 0),
  };

  const summaryData = weeks.map((week, i) => ({
    week,
    Required: summary.required[i],
    Actual: summary.actual[i],
    'O/U': summary.ou[i],
  }));

  return { optionData, summaryData, totalSummary };
};

const generateRandomDataMetrics = () => {
  const weeks = Array.from({ length: 10 }, (_, i) => `WK ${i + 1}`);
  const optionData1: Record<string, any[]> = {};
  const summary = {
    required: Array(10).fill(0),
    actual: Array(10).fill(0),
    ou: Array(10).fill(0),
  };

  selectedTraces.forEach((option) => {
    const data = weeks.map((week, i) => {
      const required = Math.floor(Math.random() * 10) + 20;
      const actual = Math.floor(Math.random() * 10) + 18;
      const ou = actual - required;
      summary.required[i] += required;
      summary.actual[i] += actual;
      summary.ou[i] += ou;
      return {
        week,
        Required: required,
        Actual: actual,
        ['O/U']: ou,
      };
    });
    optionData1[option] = data;
  });

  const totalSummary1 = {
    required: summary.required.reduce((a, b) => a + b, 0),
    actual: summary.actual.reduce((a, b) => a + b, 0),
    ou: summary.ou.reduce((a, b) => a + b, 0),
  };

  const summaryData1 = weeks.map((week, i) => ({
    week,
    Required: summary.required[i],
    Actual: summary.actual[i],
    'O/U': summary.ou[i],
  }));

  return { optionData1, summaryData1, totalSummary1 };
};

  const { optionData, summaryData, totalSummary } = useMemo(() => generateRandomData(), [options]);
  const { optionData1, summaryData1, totalSummary1 } = useMemo(() => generateRandomDataMetrics(), [selectedTraces]);


  return (
    <div className="p-1 mt-0 space-y-3">
        {!hideSummary && (
            <div className="executive-card shadow-sm p-3">
                    <h2 className="text-xl font-bold flex items-center">
                      Summary
                      <AskAIButton mainData={options} aiParams={{ cards: totalSummary, graph: summaryData }} parametersList={[]} />
                    </h2>
                    <div className="flex flex-col lg:flex-row gap-4 items-start">
                      {/* Cards - 30% width on large screens */}
                      <div className="w-full lg:w-1/5 space-y-4 mt-2">
                          <SummaryCard title="Total Required HC" color="#f97316" value={totalSummary.required} trendData={summaryData.map((d) => ({ value: d.Required }))} />
                          <SummaryCard title="Total Actual HC" color="#3b82f6" value={totalSummary.actual} trendData={summaryData.map((d) => ({ value: d.Actual }))} />
                          <SummaryCard title="Total Over/Under HC" color="#999" value={totalSummary.ou} trendData={summaryData.map((d) => ({ value: d['O/U'] }))} />
                      </div>

                {/* Chart - 70% width on large screens */}
                <div className="w-full lg:w-4/5 space-y-4">
                    <h2 className="text-lg font-bold pl-10">
                      Required Vs Actual Trend
                    </h2>
                    <div className="w-full h-96">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart
                        data={summaryData}
                        margin={{ top: 20, right: 20, bottom: 20, left: 10 }}
                        >
                        <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                        <XAxis dataKey="week" stroke="#ccc" />
                        <YAxis yAxisId="left" stroke="#ccc" />
                        <YAxis yAxisId="right" orientation="right" stroke="#ccc" />
                        <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none' }} />
                        <Legend />
                             <ReferenceLine
                              y={0}
                              yAxisId="right"
                              stroke="red"
                              strokeDasharray="3 3"
                              strokeWidth={1}
                            />
                        <Line yAxisId="left" type="monotone" dataKey="Required" stroke="#f97316" strokeWidth={2} />
                        <Line yAxisId="left" type="monotone" dataKey="Actual" stroke="#3b82f6" strokeWidth={2} />
                        <Bar yAxisId="right" dataKey="O/U" barSize={20} fill="#999">
                          {summaryData.map((entry, index) => (
                            <Cell
                              key={`cell-${index}`}
                              fill={entry["O/U"] >= 0 ? '#008001CA' : '#ff0200CA'}
                            />
                          ))}
                        </Bar>
                        </ComposedChart>
                    </ResponsiveContainer>
                    </div>
                </div>
                    </div>
                </div>
        )}

         {hideSummary && (
            <div className="executive-card shadow-sm p-3">
                 <h2 className="text-xl font-bold flex items-center">
                      {options?.[0] || 'Summary'}
                            <div className="ml-auto flex gap-2 items-center">
                              {children}
                               <AskAIButton mainData={selectedTraces} aiParams={{ cards: totalSummary1, graph: summaryData1 }} parametersList={[]} />
                                    <DropdownMenu>
                                      <DropdownMenuTrigger asChild>
                                        <Button variant="outline" className="w-[200px] justify-between">
                                          {selectedTraces.length === 1
                                            ? traces.find((t) => t === selectedTraces[0])
                                            : selectedTraces.length === traces.length
                                              ? "All Metrics"
                                              : `${selectedTraces.length} traces selected`}
                                          <ChevronDown className="ml-2 h-4 w-4 opacity-50" />
                                        </Button>
                                      </DropdownMenuTrigger>
                                      <DropdownMenuContent className="w-[200px]" style={{ maxHeight: "400px", overflowY: "auto" }} onCloseAutoFocus={(e) => e.preventDefault()}>
                                        <DropdownMenuLabel>Select Metrics</DropdownMenuLabel>
                                        <DropdownMenuSeparator />
                                          <DropdownMenuCheckboxItem
                                            key="select-all"
                                            checked={selectedTraces.length === traces.length}
                                            onCheckedChange={(checked) => {
                                              if (checked) {
                                                handleTraceChange(traces.map((t) => t));
                                              } else {
                                                handleTraceChange([]);
                                              }
                                            }}
                                            onSelect={(e) => e.preventDefault()}
                                          >
                                            Select All
                                          </DropdownMenuCheckboxItem>
                                        <DropdownMenuSeparator />
                                        {traces.map((trace) => (
                                          <DropdownMenuCheckboxItem
                                            key={trace}
                                            checked={selectedTraces.includes(trace)}
                                            onCheckedChange={(checked) => {
                                              if (checked) {
                                                handleTraceChange([...selectedTraces, trace]);
                                              } else {
                                                handleTraceChange(selectedTraces.filter((t) => t !== trace));
                                              }
                                            }}
                                            onSelect={(e) => e.preventDefault()}
                                          >
                                            {trace}
                                          </DropdownMenuCheckboxItem>
                                        ))}
                                      </DropdownMenuContent>
                                    </DropdownMenu>
                             </div>
                 </h2>
                    <div className="flex flex-col lg:flex-row gap-4 items-start">
                {/* Cards - 30% width on large screens */}
                <div className="w-full lg:w-1/5 space-y-4 mt-2">
                    <SummaryCard title="Total Required HC" color="#f97316" value={totalSummary1.required} trendData={summaryData1.map((d) => ({ value: d.Required }))} />
                    <SummaryCard title="Total Actual HC" color="#3b82f6" value={totalSummary1.actual} trendData={summaryData1.map((d) => ({ value: d.Actual }))} />
                    <SummaryCard title="Total Over/Under HC" color="#999" value={totalSummary1.ou} trendData={summaryData1.map((d) => ({ value: d['O/U'] }))} />
                </div>

                {/* Chart - 70% width on large screens */}
                <div className="w-full lg:w-4/5 space-y-4">
                    <h2 className="text-lg font-bold pl-10">
                      Required Vs Actual Trend
                    </h2>
                    <div className="w-full h-96">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart
                        data={summaryData1}
                        margin={{ top: 20, right: 20, bottom: 20, left: 10 }}
                        >
                        <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                        <XAxis dataKey="week" stroke="#ccc" />
                        <YAxis yAxisId="left" stroke="#ccc" />
                        <YAxis yAxisId="right" orientation="right" stroke="#ccc" />
                        <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none' }} />
                        <Legend />
                             <ReferenceLine
                              y={0}
                              yAxisId="right"
                              stroke="red"
                              strokeDasharray="3 3"
                              strokeWidth={1}
                            />
                        <Line yAxisId="left" type="monotone" dataKey="Required" stroke="#f97316" strokeWidth={2} />
                        <Line yAxisId="left" type="monotone" dataKey="Actual" stroke="#3b82f6" strokeWidth={2} />
                        <Bar yAxisId="right" dataKey="O/U" barSize={20} fill="#999">
                          {summaryData1.map((entry, index) => (
                            <Cell
                              key={`cell-${index}`}
                              fill={entry["O/U"] >= 0 ? '#008001CA' : '#ff0200CA'}
                            />
                          ))}
                        </Bar>
                        </ComposedChart>
                    </ResponsiveContainer>
                    </div>
                </div>
                    </div>
                </div>
        )}

      {!hideSummary && options.map((option) => (
        <div key={option} className="space-y-4 executive-card shadow-sm p-3">
          <h2 className="text-xl font-bold flex items-center">
            {option}
            <AskAIButton mainData={options} aiParams={{ cards: optionData[option], graph: optionData[option] }} parametersList={[]} />
          </h2>
          <div className="flex flex-col lg:flex-row gap-4 items-start">
             <div className="w-full lg:w-1/5 space-y-4 mt-2">
                <SummaryCard
                    title="Required HC"
                    color="#f97316"
                    value={optionData[option].reduce((acc, cur) => acc + cur.Required, 0)}
                    trendData={optionData[option].map((d) => ({ value: d.Required }))}
                />
                <SummaryCard
                    title="Actual HC"
                    color="#3b82f6"
                    value={optionData[option].reduce((acc, cur) => acc + cur.Actual, 0)}
                     trendData={optionData[option].map((d) => ({ value: d.Actual }))}
                />
                <SummaryCard
                    title="Over/Under HC"
                    color="#999"
                    value={optionData[option].reduce((acc, cur) => acc + cur['O/U'], 0)}
                    trendData={optionData[option].map((d) => ({ value: d['O/U'] }))}
                />
            </div>
            <div className="w-full lg:w-4/5 space-y-4">
                <h2 className="text-lg font-bold pl-10">
                  Required Vs Actual Trend
                </h2>
                <div className="w-full h-96">
                    <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={optionData[option]} margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                        <XAxis dataKey="week" stroke="#ccc" />
                        <YAxis yAxisId="left" stroke="#ccc" />
                        <YAxis yAxisId="right" orientation="right" stroke="#ccc" />
                        <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none' }} />
                        <Legend />
                          <ReferenceLine
                              y={0}
                              yAxisId="right"
                              stroke="red"
                              strokeDasharray="3 3"
                              strokeWidth={1}
                            />
                        <Line yAxisId="left" type="monotone" dataKey="Required" stroke="#f97316" strokeWidth={2} />
                        <Line yAxisId="left" type="monotone" dataKey="Actual" stroke="#3b82f6" strokeWidth={2} />
                        <Bar yAxisId="right" dataKey="O/U" barSize={20} fill="#999">
                          {optionData[option].map((entry, index) => (
                            <Cell
                              key={`cell-${index}`}
                              fill={entry["O/U"] >= 0 ? '#008001CA' : '#ff0200CA'}
                            />
                          ))}
                        </Bar>
                    </ComposedChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
        </div>
      ))}
       {hideSummary && selectedTraces.map((option) => (
        <div key={option} className="space-y-4 executive-card shadow-sm p-3">
          <h2 className="text-xl font-bold flex items-center">
            {option}
            <AskAIButton mainData={selectedTraces} aiParams={{ cards: optionData1[option], graph: optionData1[option] }} parametersList={[]} />
          </h2>
          <div className="flex flex-col lg:flex-row gap-4 items-start">
             <div className="w-full lg:w-1/5 space-y-4 mt-2">
                <SummaryCard
                    title="Required HC"
                    color="#f97316"
                    value={optionData1[option].reduce((acc, cur) => acc + cur.Required, 0)}
                    trendData={optionData1[option].map((d) => ({ value: d.Required }))}
                />
                <SummaryCard
                    title="Actual HC"
                    color="#3b82f6"
                    value={optionData1[option].reduce((acc, cur) => acc + cur.Actual, 0)}
                     trendData={optionData1[option].map((d) => ({ value: d.Actual }))}
                />
                <SummaryCard
                    title="Over/Under HC"
                    color="#999"
                    value={optionData1[option].reduce((acc, cur) => acc + cur['O/U'], 0)}
                    trendData={optionData1[option].map((d) => ({ value: d['O/U'] }))}
                />
            </div>
            <div className="w-full lg:w-4/5 space-y-4">
                <h2 className="text-lg font-bold pl-10">Required Vs Actual Trend</h2>
                <div className="w-full h-96">
                    <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={optionData1[option]} margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                        <XAxis dataKey="week" stroke="#ccc" />
                        <YAxis yAxisId="left" stroke="#ccc" />
                        <YAxis yAxisId="right" orientation="right" stroke="#ccc" />
                        <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none' }} />
                        <Legend />
                          <ReferenceLine
                              y={0}
                              yAxisId="right"
                              stroke="red"
                              strokeDasharray="3 3"
                              strokeWidth={1}
                            />
                        <Line yAxisId="left" type="monotone" dataKey="Required" stroke="#f97316" strokeWidth={2} />
                        <Line yAxisId="left" type="monotone" dataKey="Actual" stroke="#3b82f6" strokeWidth={2} />
                        <Bar yAxisId="right" dataKey="O/U" barSize={20} fill="#999">
                          {optionData1[option].map((entry, index) => (
                            <Cell
                              key={`cell-${index}`}
                              fill={entry["O/U"] >= 0 ? '#008001CA' : '#ff0200CA'}
                            />
                          ))}
                        </Bar>
                    </ComposedChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
        </div>
      ))}
    </div>
  );
};

export default HeadcountTrend;