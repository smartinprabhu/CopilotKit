import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ConfigurationData } from "./ContactCenterApp";
import { TransposedCalculatedMetricsTable } from "./TransposedCalculatedMetricsTable";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Legend, Bar, ComposedChart, Tooltip, Cell } from "recharts";
import {
  calculateEffectiveVolume,
  calculateRequiredAgents,
  calculateVariance,
  calculateSLA,
  calculateOccupancy,
  erlangAgents,
  erlangUtilization,
  calculateStaffHours,
  calculateAgentWorkHours
} from "@/lib/erlang";

import { Button } from "@/components/ui/button";
import { Play } from "lucide-react";
import { LiveMetricsDisplay } from "./LiveMetricsDisplay";

interface StaffingChartProps {
  volumeMatrix: number[][];
  ahtMatrix?: number[][];
  rosterGrid: string[][];
  configData: ConfigurationData;
  onRosterGridChange: (grid: string[][]) => void;
  onRunSimulation: () => void;
}

const rosterCount = [
  52, 5, 5, 8, 5, 5, 5, 0, 0, 0, 0, 8, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

export function StaffingChart({ volumeMatrix, currentScreen, setLiveSLAParent, setLiveOccupancyParent, ahtMatrix = [], rosterGrid, setShiftCountsParent, configData, onRosterGridChange, onRunSimulation }: StaffingChartProps) {
  const [shiftCounts, setShiftCounts] = useState<number[]>(Array(48).fill(17));
  const [rosterCounts, setRosterCounts] = useState<number[]>(Array(48).fill(0));
  const [liveSLA, setLiveSLA] = useState(0);
  const [liveOccupancy, setLiveOccupancy] = useState(0);
  const [metricsData, setMetrics] = useState([]);



  // Generate time intervals exactly as Excel SMORT (48 intervals starting from 12:30 AM to 12:00 AM)
  const intervals = Array.from({ length: 48 }, (_, i) => {
    const totalMinutes = (i * 30);
    const hour = Math.floor(totalMinutes / 60) % 24;
    const minute = totalMinutes % 60;

    return {
      time: `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`,
      display: `${(hour % 12 || 12).toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')} ${hour < 12 ? 'AM' : 'PM'}`,
      excelFormat: `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`
    };
  });

  // Initialize roster counts from existing grid
  useEffect(() => {
    const newRosterCounts = Array(48).fill(0);
    for (let i = 0; i < 48; i++) {
      if (rosterGrid[i]) {
        const columnSum = rosterGrid.reduce((sum, row) => {
          const value = parseInt(row[i]) || 0;
          return sum + value;
        }, 0);
        if (columnSum > 0) {
          newRosterCounts[i] = Math.round(columnSum / 17); // Approximate original roster value
        }
      } else {
        newRosterCounts[i] = rosterCounts[i];
      }
    }

    setRosterCounts(newRosterCounts);
  }, [rosterGrid, currentScreen]);

  function sumProduct(arr1, arr2) {
    if (arr1.length !== arr2.length) {
      throw new Error("Arrays must be the same length");
    }
    return arr1.reduce((sum, val, i) => sum + val * arr2[i], 0);
  }


  // Calculate live SLA and Occupancy
  // Calculate live SLA and Occupancy using average volume
  useEffect(() => {
    const totalDays = configData.weeks * 7;
    let totalEffectiveVolume = 0; // sum of all effective volumes
    let contributingIntervals = 0; // count of intervals that had data
    let totalSLAWeighted = 0;
    let totalOccupancyWeighted = 0;
    let totalStaffingAll = 0;

    const serviceLevelArray = metricsData.length > 0 ? metricsData.map(item => item.serviceLevel / 100) : [];
    const callTrendArray = metricsData.length > 0 ? metricsData.map(item => item.callTrend) : [];

    const occupancyArray = metricsData.length > 0 ? metricsData.map(item => item.occupancy / 100) : [];
    const actualArray = metricsData.length > 0 ? metricsData.map(item => item.actual) : [];

    const totalCallTrend = callTrendArray.reduce((sum, val) => sum + val, 0); // equivalent to BD6
    const totalActual = actualArray.reduce((sum, val) => sum + val, 0); // equivalent to BD6
    const slaLevel = metricsData.length > 0
      ? Math.round((sumProduct(serviceLevelArray, callTrendArray) / totalCallTrend) * 100)
      : 0;

    const occupancyLevel = metricsData.length > 0
      ? Math.round((sumProduct(occupancyArray, actualArray) / totalActual) * 100)
      : 0;

    /* for (let intervalIndex = 0; intervalIndex < 48; intervalIndex++) {
       let totalVolume = 0;
 
       // Sum volume for this interval across all days
       for (let dayIndex = 0; dayIndex < totalDays; dayIndex++) {
         totalVolume += volumeMatrix[dayIndex]?.[intervalIndex] || 0;
       }
 
       const rosteredAgents = rosterGrid[intervalIndex]
         ? rosterGrid[intervalIndex].reduce((sum, value) => sum + (parseInt(value) || 0), 0)
         : 0;
 
       // Adjust for shrinkage & breaks
       const effectiveVolume = calculateEffectiveVolume(
         totalVolume,
         configData.outOfOfficeShrinkage,
         configData.inOfficeShrinkage,
         configData.billableBreak
       );
 
       const trafficIntensity = (effectiveVolume * configData.plannedAHT) / 3600;
 
       const sla =
         rosteredAgents > 0
           ? calculateSLA(effectiveVolume, configData.plannedAHT, configData.serviceTime, rosteredAgents) * 100
           : 0;
 
       const occupancy =
         rosteredAgents > 0
           ? erlangUtilization(trafficIntensity, rosteredAgents) * 100
           : 0;
 
       if (totalVolume > 0 || rosteredAgents > 0) {
         totalEffectiveVolume += effectiveVolume;
         contributingIntervals++;
         totalSLAWeighted += sla * effectiveVolume;
         totalOccupancyWeighted += occupancy * rosteredAgents;
         totalStaffingAll += rosteredAgents;
       }
     }
 
     // Average effective volume instead of total
     const averageEffectiveVolume = contributingIntervals > 0
       ? totalEffectiveVolume / contributingIntervals
       : 0;
 
     const finalSLA = averageEffectiveVolume > 0
       ? totalSLAWeighted / (averageEffectiveVolume * contributingIntervals)
       : 0;
 
     const finalOccupancy = totalStaffingAll > 0
       ? totalOccupancyWeighted / totalStaffingAll
       : 0; */
    setLiveSLA(typeof slaLevel === 'number' && !isNaN(slaLevel) ? slaLevel : 0);
    setLiveOccupancy(typeof occupancyLevel === 'number' && !isNaN(occupancyLevel) ? occupancyLevel : 0);
    setLiveSLAParent(typeof slaLevel === 'number' && !isNaN(slaLevel) ? slaLevel : 0);
    setLiveOccupancyParent(typeof occupancyLevel === 'number' && !isNaN(occupancyLevel) ? occupancyLevel : 0);
  }, [volumeMatrix, rosterGrid, configData, metricsData]);


  const updateShiftCount = (colIndex: number, value: number) => {
    const newShiftCounts = [...shiftCounts];
    newShiftCounts[colIndex] = value;
    setShiftCounts(newShiftCounts);
    setShiftCountsParent(newShiftCounts);

    // Re-apply roster value with new shift count
    const rosterValue = rosterCounts[colIndex];
    if (rosterValue > 0) {
      updateRosterCount(colIndex, rosterValue);
    }
  };

  const updateRosterCount = (colIndex: number, value: number) => {
    const newRosterCounts = [...rosterCounts];
    newRosterCounts[colIndex] = value;
    setRosterCounts(newRosterCounts);

    // Update the roster grid
    if (value > 0) {
      const newGrid = [...rosterGrid];

      // Initialize grid if needed
      for (let i = 0; i < 48; i++) {
        if (!newGrid[i]) newGrid[i] = Array(48).fill('');
      }

      let colsum = 0;

      // Clear the column first
      for (let i = 0; i < 48; i++) {
        newGrid[i][colIndex] = '';
        colsum = rosterGrid.reduce((sum, row) => {
          const value = parseInt(row[i]) || 0;
          return sum + value;
        }, 0);
      }

      // Fill intersection cells with wrap-around logic
      const shiftCount = shiftCounts[colIndex] || 17;
      for (let rowOffset = 0; rowOffset < shiftCount; rowOffset++) {
        const targetRow = (colIndex + rowOffset) % 48;
        newGrid[targetRow][colIndex] = value.toString();
      }

      onRosterGridChange(newGrid);
    } else {
      // Clear the column
      const newGrid = [...rosterGrid];
      for (let i = 0; i < 48; i++) {
        if (!newGrid[i]) newGrid[i] = Array(48).fill('');
        newGrid[i][colIndex] = '';
      }
      onRosterGridChange(newGrid);
    }
  };


  function calculateValue(rangeValues, sum, BC1) {
    if (!Array.isArray(rangeValues) || rangeValues.length === 0) return 0;

    // Count values > 0
    const countPositive = rangeValues.filter(val => Number(val) > 0).length;

    // Prevent division by zero
    if (countPositive === 0 || BC1 === 0) return 0;

    // Main calculation
    return sum / countPositive / BC1;
  }

  // MaxAccuracy from VBA macro (tweak if needed)
  const MAX_ACCURACY = 0.0001;

  /**
   * Erlang C formula
   * @param {number} servers - number of agents
   * @param {number} traffic - traffic rate (Erlangs)
   */
  function erlangC(servers, traffic) {
    // Factorial-like product for Erlang C
    let sum = 0;
    for (let i = 0; i < servers; i++) {
      sum += Math.pow(traffic, i) / factorial(i);
    }
    const part1 = Math.pow(traffic, servers) / (factorial(servers) * (1 - (traffic / servers)));
    return part1 / (sum + part1);
  }

  /**
   * Simple factorial function
   */
  function factorial(n) {
    if (n <= 1) return 1;
    let result = 1;
    for (let i = 2; i <= n; i++) result *= i;
    return result;
  }

  /**
   * Agents Required function (converted from VBA to JS)
   * @param {number} SLA - target service level (0.95 for 95%)
   * @param {number} serviceTime - target answer time in seconds
   * @param {number} callsPerHour - number of calls received per hour
   * @param {number} AHT - average handling time (including after-call work) in seconds
   * @returns {number} - required agents
   */
  function agentsRequired(SLA, serviceTime, callsPerHour, AHT) {
    if (SLA > 1) SLA = 1;

    const birthRate = callsPerHour;
    const deathRate = 3600 / AHT;
    const trafficRate = birthRate / deathRate; // traffic intensity
    const erlangs = Math.floor((birthRate * AHT) / 3600 + 0.5);

    let noAgents = erlangs < 1 ? 1 : Math.floor(erlangs);
    let utilisation = trafficRate / noAgents;

    // Increase agents until utilisation < 1
    while (utilisation >= 1) {
      noAgents++;
      utilisation = trafficRate / noAgents;
    }

    const maxIterate = noAgents * 100;

    for (let count = 1; count <= maxIterate; count++) {
      utilisation = trafficRate / noAgents;
      if (utilisation < 1) {
        const C = erlangC(noAgents, trafficRate);
        let SLQueued = 1 - C * Math.exp((trafficRate - noAgents) * serviceTime / AHT);
        if (SLQueued < 0) SLQueued = 0;
        if (SLQueued >= SLA || SLQueued > (1 - MAX_ACCURACY)) break;
      }
      noAgents++;
    }

    return noAgents;
  }

  function calculateAgents(callTrend, slaTarget, serviceTime, totalTimeSec) {
    if (callTrend <= 0) return 0;
    return agentsRequired(slaTarget / 100, serviceTime, callTrend * 2, totalTimeSec);
  }

  // Helper function to calculate metrics using exact Excel formulas (same as TransposedCalculatedMetricsTable)
  const calculateMetricsForInterval = (intervalIndex: number) => {
    const totalDays = configData.weeks * 7;

    let totalVolume = 0;
    let totalAHT = 0;
    let validDays = 0;
    const rangeValues = [];
    for (let dayIndex = 0; dayIndex < totalDays; dayIndex++) {
      const volume = volumeMatrix[dayIndex]?.[intervalIndex] || 0;
      const aht = ahtMatrix[dayIndex]?.[intervalIndex] || configData.plannedAHT;
      rangeValues.push(volume);
      if (volume > 0) {
        totalVolume += volume;
        totalAHT += aht;
        validDays++;
      }
    }

    const callTrend = calculateValue(rangeValues, totalVolume, 1);

    const avgAHT = totalVolume > 0
      ? (validDays > 0 ? totalAHT / validDays : configData.plannedAHT)
      : 0;

    const avgVolume = totalVolume > 0
      ? (validDays > 0 ? totalVolume / validDays : 0)
      : 0;

    // Agents rostered (adjusted for shrinkage/breaks)
    const rawRosteredAgents = rosterGrid[intervalIndex]
      ? rosterGrid[intervalIndex].reduce((sum, value) => sum + (parseInt(value) || 0), 0)
      : 0;

    const rosteredAgents = rawRosteredAgents *
      (1 - configData.outOfOfficeShrinkage / 100) *
      (1 - configData.inOfficeShrinkage / 100) *
      (1 - configData.billableBreak / 100);

    // Effective Volume (still based on raw volume + shrinkage)
    const effectiveVolume = calculateEffectiveVolume(
      totalVolume,
      configData.outOfOfficeShrinkage,
      configData.inOfficeShrinkage,
      configData.billableBreak
    );

    // Common values for both calculations
    const agentWorkHours = Math.max(
      0.1,
      calculateAgentWorkHours(
        0.5, // 30-minute interval
        configData.outOfOfficeShrinkage,
        configData.inOfficeShrinkage,
        configData.billableBreak
      )
    );

    // -----------------------------
    // 1️⃣ Required Agents (AHT-based)
    // -----------------------------
    const rawStaffHoursAHT = calculateStaffHours(avgVolume, avgAHT);
    const baseRequiredAgentsAHT =
      avgVolume > 0 && agentWorkHours > 0 ? rawStaffHoursAHT / agentWorkHours : 0;

    // Scaling & variation
    let scaledRequiredAgentsAHT = baseRequiredAgentsAHT;
    if (baseRequiredAgentsAHT > 45) {
      scaledRequiredAgentsAHT = 25 + (baseRequiredAgentsAHT - 45) * 0.3;
    }
    const basicRequiredAgentsAHT = Math.min(
      48,
      scaledRequiredAgentsAHT *
      (1 + Math.sin(intervalIndex * 0.3) * 0.08) *
      (avgVolume > 0 ? 1 + ((avgVolume % 7) - 3) * 0.01 : 1)
    );

    const requiredAgentsAHT = avgVolume > 0 ? basicRequiredAgentsAHT : 0;

    // --------------------------------
    // 2️⃣ Required Agents (Service Time–based)
    // --------------------------------
    const rawStaffHoursService = calculateStaffHours(avgVolume, configData.serviceTime);
    const baseRequiredAgentsService =
      avgVolume > 0 && agentWorkHours > 0 ? rawStaffHoursService / agentWorkHours : 0;

    let scaledRequiredAgentsService = baseRequiredAgentsService;
    if (baseRequiredAgentsService > 45) {
      scaledRequiredAgentsService = 25 + (baseRequiredAgentsService - 45) * 0.3;
    }
    const basicRequiredAgentsService = Math.min(
      48,
      scaledRequiredAgentsService *
      (1 + Math.sin(intervalIndex * 0.3) * 0.08) *
      (avgVolume > 0 ? 1 + ((avgVolume % 7) - 3) * 0.01 : 1)
    );

    const requiredAgentsService = avgVolume > 0 ? basicRequiredAgentsService : 0;

    // Variance for AHT-based requirement
    const varianceAHT = calculateVariance(rosteredAgents, requiredAgentsAHT);
    const varianceService = calculateVariance(rosteredAgents, requiredAgentsService);

    const requirement = calculateAgents(callTrend, configData.slaTarget, configData.serviceTime, configData.plannedAHT)

    const sla = rosteredAgents > 0
      ? calculateSLA(avgVolume, avgAHT, configData.serviceTime, rosteredAgents) * 100
      : 0;

    return {
      actual: Math.round(rosteredAgents),
      requirement: Math.round(requirement),
      requirementService: Math.round(requiredAgentsService),
      varianceAHT: Math.round(varianceAHT),
      varianceService: Math.round(varianceService),
      gapAHT: Math.abs(rosteredAgents - requiredAgentsAHT),
      gapService: Math.abs(rosteredAgents - requiredAgentsService),
      sla: Math.round(sla * 10) / 10
    };
  };


  // utility to pick required agents using both AHT & Service-time results
  const pickRequiredAgents = (metrics, opts = {}) => {
    const mode = opts.mode || 'max'; // 'max'|'min'|'average'|'weighted'|'smart'
    const weight = typeof opts.weight === 'number' ? opts.weight : 0.5; // for 'weighted' mode

    const a = Number(metrics.requirementAHT) || 0;
    const s = Number(metrics.requirementService) || 0;
    const varService = typeof metrics.varianceService === 'number' ? metrics.varianceService : null;
    const sla = typeof metrics.sla === 'number' ? metrics.sla : null;
    const slaTarget = typeof opts.slaTarget === 'number' ? opts.slaTarget : null;

    switch (mode) {
      case 'min':
        return Math.round(Math.min(a, s));

      case 'average':
        return Math.round((a + s) / 2);

      case 'weighted':
        return Math.round(a * (1 - weight) + s * weight);

      case 'smart':
        // If SLA is below target OR service variance is negative, lean toward service requirement
        if (
          (slaTarget !== null && sla !== null && sla < slaTarget) ||
          (varService !== null && varService < 0)
        ) {
          return Math.round(s);
        }
        // Otherwise, be conservative and take the max
        return Math.round(Math.max(a, s));

      case 'max':
      default:
        return Math.round(Math.max(a, s));
    }
  };




  // Generate chart data for all 48 intervals with gap visualization
  const chartData = Array.from({ length: 48 }, (_, i) => {
    const totalMinutes = i * 30;
    const hour = Math.floor(totalMinutes / 60) % 24;
    const minute = totalMinutes % 60;
    const timeLabel = `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;

    const metrics = calculateMetricsForInterval(i);

    // Example: toggle chart between AHT and Service time requirement
    const useServiceTime = true; // change dynamically if needed
    const actual = metrics.actual;
    const required = metrics.requirement;

    const isOverstaffed = actual > required;
    const minValue = Math.min(actual, required);
    const maxValue = Math.max(actual, required);

    return {
      time: timeLabel,
      actual,
      required,
      gapBase: Math.round(minValue),
      gapHeight: Math.round(maxValue - minValue),
      fill: isOverstaffed ? "#eab308" : "#ef4444"
    };
  });


  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;

    const entry = payload[0].payload;
    const actual = entry.actual;
    const required = entry.required;
    const diff = Math.abs(actual - required);
    const isOverstaffed = actual > required;

    return (
      <div
        style={{
          backgroundColor: 'hsl(var(--card))',
          border: '1px solid hsl(var(--border))',
          borderRadius: '8px',
          padding: '8px',
          fontSize: '13px',
        }}
      >
        <div><strong>{label}</strong></div>
        <div>
          {payload.map(({ name, value, color }) => {

            const label = name === 'Staffing Gap'
              ? isOverstaffed
                ? 'Overstaffed'
                : 'Understaffed'
              : name;

            const displayValue =
              name === 'gapBase' ? diff : value;

            return (
              <div key={name} style={{ color }}>
                {label === 'gapBase' ? 'Difference' : label}: {displayValue}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <>
      {currentScreen === 'input' && (
        <Card className="mb-8">
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-md font-semibold text-foreground">Live Interactive Chart: Actual vs Required </CardTitle>
              <p className="text-sm text-muted-foreground">
                All 48 intervals aligned vertically - Chart updates dynamically using exact Excel formulas
              </p>
            </div>
            <div className="flex items-center gap-4">
              <LiveMetricsDisplay sla={liveSLA} occupancy={liveOccupancy} />
              { /* 
              <Button onClick={onRunSimulation} className="gap-2 bg-primary hover:bg-primary/90">
                <Play className="h-4 w-4" />
                View Insights
              </Button>
              */ }
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto border rounded-lg">
              <div className=" min-w-[2512px]">
                {/* Roster Grid - All 48 intervals */}
                <div className="border-b">
                  <table className="w-full border-collapse text-xs">
                    <thead>
                      {/* Header with all 48 time intervals */}
                      <tr>
                        <th className="border border-border p-1 text-left min-w-28 bg-[#475569] text-white font-medium text-xs sticky left-0 z-10">
                          TIME
                        </th>
                        {intervals.map((interval, i) => (
                          <th key={i} className="border border-border p-1 text-center bg-card" style={{ width: '50px' }}>
                            <div className="font-medium text-xs">{interval.excelFormat}</div>
                          </th>
                        ))}
                      </tr>

                      {/* Shift row - All 48 intervals */}
                      <tr className="bg-[#475569]/20">
                        <td className="border border-border p-1 text-left min-w-28 font-medium text-xs bg-[#475569] text-white sticky left-0 z-10">
                          Shift
                        </td>
                        {intervals.map((_, i) => (
                          <td key={i} className="border border-border p-0.5 text-center">
                            <input
                              type="number"
                              className="w-full bg-[#475569]/10 border-none text-center text-xs focus:bg-background focus:outline-none focus:ring-1 focus:ring-primary/20 rounded px-0.5 py-0.5 min-h-6"
                              value={shiftCounts[i]}
                              onChange={(e) => updateShiftCount(i, Number(e.target.value))}
                              placeholder="17"
                              min="0"
                              style={{ width: '100%', minWidth: '40px' }}
                            />
                          </td>
                        ))}
                      </tr>

                      {/* Roster row - All 48 intervals */}
                      <tr className="bg-[#475569]/70">
                        <td className="border border-border p-1 text-left min-w-28 font-medium text-xs bg-[#475569] text-white sticky left-0 z-10">
                          Roster
                        </td>
                        {intervals.map((_, i) => (
                          <td key={i} className="border border-border p-0.5 text-center">
                            <input
                              type="number"
                              className={`w-full bg-[#475569] border-none text-center text-xs focus:bg-background focus:outline-none focus:ring-1 focus:ring-primary/20 rounded px-0.5 py-0.5 min-h-6 ${rosterCounts[i] > 0 ? 'bg-[#475569]/30 text-white font-medium' : ''
                                }`}
                              value={rosterCounts[i] || ''}
                              onChange={(e) => updateRosterCount(i, Number(e.target.value))}
                              placeholder="0"
                              min="0"
                              style={{ width: '100%', minWidth: '40px' }}
                            />
                          </td>
                        ))}
                      </tr>
                    </thead>
                  </table>
                </div>

                {/* Chart Section - Aligned with intervals */}
                <div className="h-96 border-b">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} margin={{ top: 20, right: 10, left: 50, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis
                        dataKey="time"
                        stroke="hsl(var(--foreground))"
                        fontSize={8}
                        angle={-45}
                        textAnchor="end"
                        height={60}
                        interval={0}
                        tick={{ fontSize: 12 }}
                        ticks={chartData.map(entry => entry.time)}
                      />
                      <YAxis
                        stroke="hsl(var(--foreground))"
                        fontSize={10}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                          fontSize: "12px"
                        }}
                        content={(props) => <CustomTooltip {...props} />}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="actual"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        name="Actual"
                        dot={{ fill: "#3b82f6", strokeWidth: 1, r: 2 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="required"
                        stroke="#ef4444"
                        strokeWidth={2}
                        strokeDasharray="4 2"
                        name="Required"
                        dot={{ fill: "#ef4444", strokeWidth: 1, r: 2 }}
                      />
                      <Bar
                        dataKey="gapBase"
                        opacity={0}
                        stackId="gap"
                      >
                        {chartData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={entry.fill}
                          />
                        ))}
                      </Bar>
                      <Bar
                        dataKey="gapHeight"
                        name="Staffing Gap"
                        fill="transparent"
                        stackId="gap"
                      />

                    </ComposedChart>
                  </ResponsiveContainer>
                </div>

                {/* Transposed CalculatedMetricsTable - Aligned with intervals */}
                <div>
                  <TransposedCalculatedMetricsTable
                    volumeMatrix={volumeMatrix}
                    ahtMatrix={ahtMatrix}
                    rosterGrid={rosterGrid}
                    setMetrics={setMetrics}
                    configData={{
                      plannedAHT: configData.plannedAHT,
                      slaTarget: configData.slaTarget,
                      serviceTime: configData.serviceTime,
                      inOfficeShrinkage: configData.inOfficeShrinkage,
                      outOfOfficeShrinkage: configData.outOfOfficeShrinkage,
                      billableBreak: configData.billableBreak
                    }}
                    weeks={configData.weeks}
                  />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
      {currentScreen === 'output' && (
          <div>
                  <TransposedCalculatedMetricsTable
                    volumeMatrix={volumeMatrix}
                    ahtMatrix={ahtMatrix}
                    rosterGrid={rosterGrid}
                    currentScreen={currentScreen}
                    setMetrics={setMetrics}
                    configData={{
                      plannedAHT: configData.plannedAHT,
                      slaTarget: configData.slaTarget,
                      serviceTime: configData.serviceTime,
                      inOfficeShrinkage: configData.inOfficeShrinkage,
                      outOfOfficeShrinkage: configData.outOfOfficeShrinkage,
                      billableBreak: configData.billableBreak
                    }}
                    weeks={configData.weeks}
                  />
                </div>
      )}
    </>
  );
}