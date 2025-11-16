import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import React, { useEffect } from 'react';
import { Calculator, TrendingUp, TrendingDown } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  calculateEffectiveVolume,
  calculateRequiredAgents,
  calculateVariance,
  calculateSLA,
  calculateSLAWithTraffic,
  calculateOccupancy,
  calculateInflux,
  calculateAgentDistributionRatio,
  erlangAgents,
  erlangUtilization,
  calculateCallTrend,
  calculateCallTrendShrinkage,
  calculateStaffHours,
  calculateAgentWorkHours
} from "@/lib/erlang";

interface TransposedCalculatedMetricsTableProps {
  volumeMatrix: number[][];
  ahtMatrix: number[][];
  rosterGrid: string[][];
  configData: {
    plannedAHT: number;
    slaTarget: number;
    serviceTime: number;
    inOfficeShrinkage: number;
    outOfOfficeShrinkage: number;
    billableBreak: number;
  };
  weeks: 4 | 8 | 12;
}

export function TransposedCalculatedMetricsTable({
  volumeMatrix,
  ahtMatrix,
  rosterGrid,
  configData,
  setMetrics,
  currentScreen,
  weeks
}: TransposedCalculatedMetricsTableProps) {


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

  function erlangB(servers, intensity) {
    if (servers < 0 || intensity < 0) {
      return 0;
    }

    const maxIterate = Math.floor(servers);
    let val = intensity;
    let last = 1; // for server = 0
    let B = 0;

    for (let count = 1; count <= maxIterate; count++) {
      B = (val * last) / (count + (val * last));
      last = B;
    }

    return minMax(B, 0, 1);
  }

  // Erlang C formula
  function erlangCservers(servers, intensity) {
    if (servers < 0 || intensity < 0) {
      return 0;
    }

    const B = erlangB(servers, intensity);
    const C = B / (((intensity / servers) * B) + (1 - (intensity / servers)));

    return minMax(C, 0, 1);
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

  function minMax(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function SLA(agents, serviceTime, callsPerHour, AHT) {
    try {
      const birthRate = callsPerHour;
      const deathRate = 3600 / AHT;

      const trafficRate = birthRate / deathRate;
      let utilisation = trafficRate / agents;
      if (utilisation >= 1) utilisation = 0.99;

      const C = erlangCservers(agents, trafficRate);
      const SLQueued = 1 - C * Math.exp((trafficRate - agents) * serviceTime / AHT);

      return minMax(SLQueued, 0, 1);
    } catch {
      return 0;
    }
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

  function utilisation(agents, callsPerHour, ahtSeconds) {
    if (agents <= 0 || callsPerHour < 0 || ahtSeconds <= 0) return 0;

    const birthRate = callsPerHour; // calls/hour
    const deathRate = 3600 / ahtSeconds; // completions/hour per agent
    const trafficRate = birthRate / deathRate; // Erlangs (offered load)

    const util = trafficRate / agents; // occupancy ratio
    return minMax(util, 0, 1);
  }


  function calculateAgents(callTrend, slaTarget, serviceTime, totalTimeSec) {
    if (callTrend <= 0) return 0;
    return agentsRequired(slaTarget / 100, serviceTime, callTrend * 2, totalTimeSec);
  }


  const calculateMetrics = () => {
    const metrics = [];
    const totalDays = weeks * 7;

    const totalAgents = rosterGrid.reduce((total, row) => {
      return total + row.reduce((rowSum, val) => rowSum + (parseInt(val) || 0), 0);
    }, 0) || 1;

    for (let intervalIndex = 0; intervalIndex < 48; intervalIndex++) {
      const totalMinutes = intervalIndex * 30; // Start from 00:00
      const hour = Math.floor(totalMinutes / 60) % 24;
      const minute = totalMinutes % 60;

      const timeDisplay = `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;

      let totalVolume = 0;
      let totalAHT = 0;
      let validDays = 0;
      let totalHandlingTime = 0; // volume * aht
      let hasTraffic = false;

      for (let dayIndex = 0; dayIndex < totalDays; dayIndex++) {
        const volume = volumeMatrix[dayIndex]?.[intervalIndex] || 0;
        const aht = ahtMatrix[dayIndex]?.[intervalIndex] || configData.plannedAHT;

        if (volume > 0) {
          totalVolume += volume;
          totalAHT += aht;
          totalHandlingTime += volume * aht;
          validDays++;
          hasTraffic = true;
        }
      }

      // Excel SMORT AHT: =IF(BD7=0,0,$BE$6)
      // If traffic intensity is 0, return 0, otherwise return planned AHT (BE6)

      const avgAHT = hasTraffic
        ? (configData.plannedAHT / 60)
        : 0;

      const rawRosteredAgents = rosterGrid[intervalIndex] ?
        rosterGrid[intervalIndex].reduce((sum, value) => sum + (parseInt(value) || 0), 0) : 0;



      const rosteredAgents = rawRosteredAgents *
        (1 - configData.outOfOfficeShrinkage / 100) *
        (1 - configData.inOfficeShrinkage / 100) *
        (1 - configData.billableBreak / 100);


      const effectiveVolume = calculateEffectiveVolume(
        totalVolume,
        configData.outOfOfficeShrinkage,
        configData.inOfficeShrinkage,
        configData.billableBreak
      );

      // Fixed required agents calculation
      // Basic staffing calculation using raw volume (no double shrinkage)
      const rawStaffHours = calculateStaffHours(totalVolume, avgAHT);
      const agentWorkHours = Math.max(0.1, calculateAgentWorkHours(
        0.5, // 30-minute interval
        configData.outOfOfficeShrinkage,
        configData.inOfficeShrinkage,
        configData.billableBreak
      )); // Ensure minimum 0.1 hours to prevent division by very small numbers

      // Basic requirement: Raw staff hours / adjusted agent work hours (only if we have actual volume)
      const baseRequiredAgents = (totalVolume > 0 && agentWorkHours > 0) ?
        rawStaffHours / agentWorkHours : 0;

      // Smart scaling to keep values under 50 but add variation
      let scaledRequiredAgents = baseRequiredAgents;
      if (baseRequiredAgents > 45) {
        // Scale down large values to fit under 50
        scaledRequiredAgents = 25 + (baseRequiredAgents - 45) * 0.3;
      }

      const nonZeroVolumes = [];
      const rangeValues = [];
      for (let dayIndex = 0; dayIndex < totalDays; dayIndex++) {
        const volume = volumeMatrix[dayIndex]?.[intervalIndex] || 0;
        rangeValues.push(volume);
        if (volume > 0) nonZeroVolumes.push(volume);
      }

      const callTrend = calculateValue(rangeValues, totalVolume, 1); //avgVolume > 0 && plannedVolume > 0 ? (avgVolume / plannedVolume) * 100 : 0;

      // Add natural variation (smaller now to stay within bounds)
      const intervalVariationFactor = 1 + (Math.sin(intervalIndex * 0.3) * 0.08); // ±8% variation
      const volumeVariationFactor = totalVolume > 0 ?
        1 + ((totalVolume % 7) - 3) * 0.01 : 1; // Small variation based on volume patterns

      const basicRequiredAgents = calculateAgents(callTrend, configData.slaTarget, configData.serviceTime, configData.plannedAHT) // Math.min(48, scaledRequiredAgents * intervalVariationFactor * volumeVariationFactor);

      // Excel SMORT BD7*2 pattern: Traffic intensity calculation
      // BD7 = traffic for 30-min interval, BD7*2 = doubled for Erlang functions
      const trafficIntensityBase = (effectiveVolume * avgAHT) / 3600; // BD7 in Erlangs
      const trafficIntensity = trafficIntensityBase; // For our calculations
      const trafficIntensityDoubled = trafficIntensityBase * 2; // BD7*2 for Excel functions
      // Excel SMORT Agents($A$1,$B$1,BD7*2,BE7) - uses doubled traffic intensity
      const erlangRequiredAgents = (effectiveVolume > 0 && trafficIntensityDoubled > 0) ?
        erlangAgents(configData.slaTarget / 100, configData.serviceTime, trafficIntensityDoubled, avgAHT) : 0;

      // Use basic calculation, but if no volume, requirement should be 0
      const requiredAgents = totalVolume > 0 ? basicRequiredAgents : 0;

      const variance = calculateVariance(rosteredAgents, requiredAgents);
      // Excel SMORT Call Trend: =IFERROR((SUM(BP7:CQ7)/COUNTIF(BP7:CQ7,">0")/$BC$1),0)
      // This calculates average of a range divided by a baseline ($BC$1)
      // For our implementation: average volume across days divided by planned volume

      const avgVolume = nonZeroVolumes.length > 0 ? nonZeroVolumes.reduce((a, b) => a + b, 0) / nonZeroVolumes.length : 0;
      const plannedVolume = Math.max(10, avgVolume * 0.9); // BC1 equivalent - planned baseline


      // Excel SMORT SLA(BA7,$B$1,BD7*2,BE7) - Service level calculation
      // Use traffic intensity approach instead of volume
      const serviceLevel = SLA(rosteredAgents, configData.serviceTime, callTrend * 2, configData.plannedAHT);

      const occLevel = utilisation(rosteredAgents, callTrend * 2, configData.plannedAHT);

      // Excel SMORT Utilisation(BA7,BD7*2,BE7) - Occupancy calculation
      const occupancy = rosteredAgents > 0 ?
        Math.min(100, (trafficIntensityDoubled / rosteredAgents) * 100) : 0;
      // Excel SMORT Influx: =IFERROR((BA7/BB7),0) = Effective Volume / Required Agents
      const influx = requiredAgents > 0 ? effectiveVolume / requiredAgents : 0;
      const agentDistributionRatio = calculateAgentDistributionRatio(rosteredAgents, totalAgents);

      if (totalVolume > 0 || rosteredAgents > 0) {
        metrics.push({
          time: timeDisplay,
          actual: Math.round(rosteredAgents),
          requirement: Math.round(requiredAgents),
          variance: Math.round(variance * 10) / 10,
          callTrend: Math.round(callTrend),
          aht: Math.round(avgAHT),
          serviceLevel: serviceLevel && !isNaN(serviceLevel) ? Math.round(serviceLevel * 100) : 0,
          occupancy: Math.round(occLevel * 100),
          influx: Math.round(influx),
          agentDistributionRatio: Math.round(agentDistributionRatio * 10) / 10,
          raw: {
            totalVolume,
            effectiveVolume,
            avgAHT,
            trafficIntensity,
            trafficIntensityDoubled,
            requiredAgents,
            callTrend,
            basicRequiredAgents,
            erlangRequiredAgents,
            plannedVolume,
            avgVolume,
            nonZeroVolumes,
            rosteredAgents,
            rawRosteredAgents,
            totalAgents,
            serviceLevel,
            occupancy,
            influx,
            agentDistributionRatio,
            variance,
            rawStaffHours,
            agentWorkHours,
            outOfOfficeShrinkage: configData.outOfOfficeShrinkage,
            inOfficeShrinkage: configData.inOfficeShrinkage,
            billableBreak: configData.billableBreak
          }
        });
      }
    }

    return metrics;
  };

  const metrics = calculateMetrics();

  useEffect(() => {
    setMetrics(metrics);
  }, [
    volumeMatrix,
    ahtMatrix,
    rosterGrid,
    configData,
    currentScreen,
  ]);


  const metricTypes = [
    { key: 'actual', label: 'Actual', format: 'number' },
    { key: 'requirement', label: 'Requirement', format: 'number' },
    { key: 'variance', label: 'Variance', format: 'number' },
    { key: 'callTrend', label: 'Call Trend', format: 'number' },
    { key: 'aht', label: 'AHT (min)', format: 'time' },
    { key: 'serviceLevel', label: 'Service Level', format: 'percentage' },
    { key: 'occupancy', label: 'Occupancy', format: 'percentage' },
    { key: 'influx', label: 'Influx', format: 'number' },
    { key: 'agentDistributionRatio', label: 'Ratio', format: 'percentage' }
  ];

  const formatValue = (value: number, type: string) => {
    const roundedValue = Math.round(value);
    switch (type) {
      case 'percentage':
        return `${roundedValue}%`;
      case 'time':
        return `${roundedValue} min`;
      default:
        return roundedValue.toString();
    }
  };

  const getVarianceColor = (metricKey: string, value: number) => {
    if (metricKey === 'variance') {
      if (value > 0) return "text-chart-2";
      if (value < -2) return "text-chart-1";
      return "text-chart-4";
    }
    return "";
  };

  const getVarianceIcon = (metricKey: string, value: number) => {
    if (metricKey === 'variance') {
      if (value > 0) return <TrendingUp className="h-3 w-3 inline ml-1" />;
      if (value < 0) return <TrendingDown className="h-3 w-3 inline ml-1" />;
    }
    return null;
  };

  function getTimeIndex(timeStr) {
    const [hours, minutes] = timeStr.split(':').map(Number);
    return hours * 2 + (minutes >= 30 ? 1 : 0);
  }

  const renderTooltipContent = (metric: any, metricTypeKey: string) => {
    const { raw, ...displayMetrics } = metric;
    const value = displayMetrics[metricTypeKey];

    const nonZeroVolumes = [];
    const rangeValues = [];
    const totalDays = weeks * 7;
    let totalVolume = 0;
    const intervalIndex = getTimeIndex(metric.time);
    for (let dayIndex = 0; dayIndex < totalDays; dayIndex++) {
      const volume = volumeMatrix[dayIndex]?.[intervalIndex] || 0;
      if (volume > 0) {
        totalVolume += volume;
      }
      rangeValues.push(volume);
      if (volume > 0) nonZeroVolumes.push(volume);
    }

    const callTrend = raw.callTrend;
    switch (metricTypeKey) {
      case 'actual':
        return (
          <div className="text-sm">
            <div className="font-semibold mb-1">Actual Agents Calculation</div>
            <code className="block bg-muted p-1 rounded mb-2 text-xs">
              Actual = Raw Agents × (1 - OOO) × (1 - IO) × (1 - BB)
            </code>
            <div className="text-xs space-y-1 mt-2">
              <div className="font-medium">Step-by-step:</div>
              <div>1. Raw Rostered Agents = {raw.rawRosteredAgents?.toFixed(2) || 'N/A'}</div>
              <div>2. Shrinkage Factors:</div>
              <div>   - Out of Office: {raw.outOfOfficeShrinkage || 'N/A'}% = {((100 - (raw.outOfOfficeShrinkage || 0)) / 100).toFixed(2)} available</div>
              <div>   - In Office: {raw.inOfficeShrinkage || 'N/A'}% = {((100 - (raw.inOfficeShrinkage || 0)) / 100).toFixed(2)} available</div>
              <div>   - Billable Break: {raw.billableBreak || 'N/A'}% = {((100 - (raw.billableBreak || 0)) / 100).toFixed(2)} available</div>
              <div className="font-medium mt-2">Final Calculation:</div>
              <div>Actual = {raw.rawRosteredAgents?.toFixed(2) || 'N/A'} × {((100 - (raw.outOfOfficeShrinkage || 0)) / 100).toFixed(2)} × {((100 - (raw.inOfficeShrinkage || 0)) / 100).toFixed(2)} × {((100 - (raw.billableBreak || 0)) / 100).toFixed(2)} = {value}</div>
            </div>
          </div>
        );
      case 'requirement':
        return (
          <div className="text-sm">
            <div className="font-semibold mb-1">Requirement Calculation</div>
            <code className="block bg-muted p-1 rounded mb-2 text-xs">
              Required = Agents(SLATarget / 100, Service Time, CallTrend * 2, Planned AHT)
            </code>
            <div className="text-xs space-y-1 mt-2">
              <div className="font-medium">Step-by-step:</div>
              <div>1. SLATarget = {(configData.slaTarget / 100).toFixed(2)}</div>
              <div>2. Service Time = {configData.serviceTime} seconds</div>
              <div>3. CallTrend = {callTrend.toFixed(2) || 'N/A'} × 2 = {parseFloat(callTrend.toFixed(2)) * 2}</div>
              <div>4. Planned AHT = {configData.plannedAHT}</div>
              <div className="font-medium mt-2">Final Calculation:</div>
              <div>Required = {calculateAgents(callTrend, configData.slaTarget, configData.serviceTime, configData.plannedAHT)}</div>
            </div>
          </div>
        );
      case 'variance':
        return (
          <div className="text-sm">
            <div className="font-semibold mb-1">Variance Calculation</div>
            <code className="block bg-muted p-1 rounded mb-2 text-xs">
              Variance = Actual Agents - Required Agents
            </code>
            <div className="text-xs space-y-1 mt-2">
              <div>Actual Agents = {raw.rosteredAgents.toFixed(2)}</div>
              <div>Required Agents = {raw.requiredAgents.toFixed(2)}</div>
              <div className="font-medium mt-2">Calculation:</div>
              <div>= {value}</div>
            </div>
          </div>
        );
      case 'callTrend':
        return (
          <div className="text-sm">
            <div className="font-semibold mb-1">Call Trend Calculation</div>
            <code className="block bg-muted p-1 rounded mb-2 text-xs">
              Call Trend = (Total Volume / Count of Non Zero Volumes / 1)
            </code>
            <div className="text-xs space-y-1 mt-2">
              <div className="font-medium">Step-by-step:</div>
              <div>1. Non-zero volumes: {rangeValues.filter(val => Number(val) > 0).length}</div>
              <div>2. Total Volume = {totalVolume}</div>
              <div className="font-medium mt-2">Final Calculation:</div>
              <div>Call Trend = ({totalVolume} ÷ {rangeValues.filter(val => Number(val) > 0).length} ÷ 1) = {Math.round(calculateValue(rangeValues, totalVolume, 1))}</div>
            </div>
          </div>
        );
      case 'aht':
        return (
          <div className="text-sm">
            <div className="font-semibold mb-1">AHT Calculation</div>
            <code className="block bg-muted p-1 rounded mb-2 text-xs">
              AHT = (Total AHT / 60)
            </code>
            <div className="text-xs space-y-1 mt-2">
              <div>Total AHT = {configData.plannedAHT}s</div>
              <div className="font-medium mt-2">Calculation:</div>
              <div>AHT = {value} min</div>
            </div>
          </div>
        );
      case 'serviceLevel':
        return (
          <div className="text-sm">
            <div className="font-semibold mb-1">Service Level Calculation</div>
            <code className="block bg-muted p-1 rounded mb-2 text-xs">
              SLA = SLA(Actual Agents, Service Time, Call Trend * 2, Planned AHT);
            </code>
            <div className="text-xs space-y-1 mt-2">
              <div>Actual Agents = {Math.round(raw.rosteredAgents)}</div>
              <div>Planned AHT = {configData.plannedAHT}s</div>
              <div>Service Time = {configData.serviceTime}s</div>
              <div>Call Trend = {Math.round(callTrend)} * 2 = {Math.round(callTrend) * 2}</div>
              <div className="font-medium mt-2">Calculation:</div>
              <div> Service Level = {value}%</div>
            </div>
          </div>
        );
      case 'occupancy':
        return (
          <div className="text-sm">
            <div className="font-semibold mb-1">Occupancy Calculation</div>
            <code className="block bg-muted p-1 rounded mb-2 text-xs">
              Occupancy = utilisation(Actual Agents, Call Trend * 2, Planned AHT);
            </code>
            <div className="text-xs space-y-1 mt-2">
              <div>Actual Agents = {Math.round(raw.rosteredAgents)}</div>
              <div>Planned AHT = {configData.plannedAHT}s</div>
              <div>Call Trend = {Math.round(callTrend)} * 2 = {Math.round(callTrend) * 2}</div>
              <div className="font-medium mt-2">Final Calculation:</div>
              <div>Occupancy = {value}%</div>
            </div>
          </div>
        );
      case 'influx':
        return (
          <div className="text-sm">
            <div className="font-semibold mb-1">Influx Calculation</div>
            <code className="block bg-muted p-1 rounded mb-2 text-xs">
              Excel: =IFERROR((BA7/BB7),0) = Effective Volume / Required Agents
            </code>
            <div className="text-xs space-y-1 mt-2">
              <div>BA7 (Effective Volume) = {raw.effectiveVolume?.toFixed(2) || 'N/A'}</div>
              <div>BB7 (Required Agents) = {raw.requiredAgents?.toFixed(2) || 'N/A'}</div>
              <div className="font-medium mt-2">Calculation:</div>
              <div>Influx = {raw.effectiveVolume?.toFixed(2) || 'N/A'} ÷ {raw.requiredAgents?.toFixed(2) || 'N/A'} = {value}</div>
              <div className="text-xs mt-1 italic">Volume per required agent ratio</div>
            </div>
          </div>
        );
      case 'agentDistributionRatio':
        return (
          <div className="text-sm">
            <div className="font-semibold mb-1">Agent Ratio Calculation</div>
            <code className="block bg-muted p-1 rounded mb-2 text-xs">
              Ratio = (Actual Agents / Total Agents) * 100
            </code>
            <div className="text-xs space-y-1 mt-2">
              <div>Actual Agents = {raw.rosteredAgents.toFixed(2)}</div>
              <div>Total Agents = {raw.totalAgents.toFixed(2)}</div>
              <div className="font-medium mt-2">Calculation:</div>
              <div>= {value}%</div>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <>
      {currentScreen === 'input' && (
        <table className="w-full border-collapse text-sm">
          <thead className="sticky top-[-1px] bg-card z-10">
            <tr>
              <th className="border border-border p-2 text-left min-w-28 bg-[#475569]  text-white font-medium sticky left-0 z-20" style={{ width: '70px' }}>Metric</th>
              {metrics.map((metric, index) => (
                <th key={index} className="border border-border p-2 text-center bg-[#475569]  text-white font-medium" style={{ width: '50px' }}>
                  {metric.time}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metricTypes.map((metricType, rowIndex) => (
              <tr key={rowIndex} className="hover:bg-muted/50">
                <td className="border border-border p-2 font-medium bg-[#475569]  text-white sticky left-0 z-10" style={{ width: '70px' }}>
                  {metricType.label}
                </td>
                {metrics.map((metric, colIndex) => {
                  const value = metric[metricType.key as keyof typeof metric] as number;
                  return (
                    <td
                      key={colIndex}
                      className={`border border-border p-2 text-center ${getVarianceColor(metricType.key, value)}`}
                      style={{ width: '50px' }}
                    >
                      <Tooltip>
                        <TooltipTrigger>
                          <span>
                            {formatValue(value, metricType.format)}
                            {getVarianceIcon(metricType.key, value)}
                          </span>
                        </TooltipTrigger>
                        <TooltipContent className="max-w-[350px]">
                          {renderTooltipContent(metric, metricType.key)}
                        </TooltipContent>
                      </Tooltip>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}