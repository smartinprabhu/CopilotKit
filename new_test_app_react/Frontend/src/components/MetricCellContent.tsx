import React, { useState, useEffect, useRef } from "react";
import { Input } from "@/components/ui/input";
import { Edit3, ArrowDown, ArrowUp, Minus } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { CapacityDataRow, TeamPeriodicMetrics, AggregatedPeriodicMetrics, MetricDefinition, TimeInterval, TEAM_METRIC_ROW_DEFINITIONS, AGGREGATED_METRIC_ROW_DEFINITIONS, STANDARD_WEEKLY_WORK_MINUTES, STANDARD_MONTHLY_WORK_MINUTES } from "./types";

interface MetricCellContentProps {
  item: CapacityDataRow;
  metricData: TeamPeriodicMetrics | AggregatedPeriodicMetrics | undefined;
  metricDef: MetricDefinition;
  periodName: string;
  onTeamMetricChange: (lobId: string, teamName: string, periodHeader: string, metricKey: string, newValue: string) => void;
  onLobMetricChange: (lobId: string, periodHeader: string, metricKey: string, newValue: string) => void;
  isEditing: boolean;
  onSetEditingCell: (id: string | null, period: string | null, metricKey: string | null) => void;
  selectedTimeInterval: TimeInterval;
}

const MetricCellContent: React.FC<MetricCellContentProps> = React.memo(({
  item,
  metricData,
  metricDef,
  periodName,
  onTeamMetricChange,
  onLobMetricChange,
  isEditing,
  onSetEditingCell,
  selectedTimeInterval,
}) => {
  const [tempValue, setTempValue] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const rawValue = metricData ? (metricData as any)[metricDef.key] : null;

  const canEditCell = (item.itemType === 'Team' && metricDef.isEditableForTeam && !metricDef.isDisplayOnly &&
    (metricDef.category === 'Assumption' || metricDef.category === 'PrimaryHC' || metricDef.category === 'HCAdjustment')) ||
    (item.itemType === 'LOB' && metricDef.key === 'lobTotalBaseRequiredMinutes' && !metricDef.isDisplayOnly);

  useEffect(() => {
    if (isEditing) {
      setTempValue(rawValue === null || rawValue === undefined ? "" : String(rawValue));
      if (inputRef.current) {
        inputRef.current.focus();
        inputRef.current.select();
      }
    } else {
      setTempValue(null);
    }
  }, [isEditing, rawValue]);

  const handleEditClick = () => {
    if (!canEditCell) return;

    let editId: string | null = null;
    if (item.itemType === 'Team' && item.lobId) {
      editId = `${item.lobId}_${item.name.replace(/\s+/g, '-')}`;
    } else if (item.itemType === 'LOB') {
      editId = item.id;
    }

    if (editId) {
      onSetEditingCell(editId, periodName, metricDef.key as string);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTempValue(e.target.value);
  };

  const handleSave = () => {
    const currentVal = tempValue;
    setTempValue(null);
    onSetEditingCell(null, null, null);

    if (currentVal === null || currentVal.trim() === "") {
      if (item.itemType === 'Team' && metricDef.isEditableForTeam && !metricDef.isDisplayOnly && item.lobId &&
        (metricDef.category === 'Assumption' || metricDef.category === 'PrimaryHC' || metricDef.category === 'HCAdjustment')) {
        onTeamMetricChange(item.lobId, item.name, periodName, metricDef.key as string, "");
      } else if (item.itemType === 'LOB' && metricDef.key === 'lobTotalBaseRequiredMinutes' && !metricDef.isDisplayOnly) {
        onLobMetricChange(item.id, periodName, metricDef.key as string, "");
      }
      return;
    }

    const numVal = parseFloat(currentVal);
    if (isNaN(numVal) && metricDef.key !== 'someStringFieldIfAny') {
      return;
    }

    if (item.itemType === 'Team' && metricDef.isEditableForTeam && !metricDef.isDisplayOnly && item.lobId &&
      (metricDef.category === 'Assumption' || metricDef.category === 'PrimaryHC' || metricDef.category === 'HCAdjustment')) {
      onTeamMetricChange(item.lobId, item.name, periodName, metricDef.key as string, currentVal);
    } else if (item.itemType === 'LOB' && metricDef.key === 'lobTotalBaseRequiredMinutes' && !metricDef.isDisplayOnly) {
      onLobMetricChange(item.id, periodName, metricDef.key as string, currentVal);
    }
  };

  const handleCancel = () => {
    setTempValue(null);
    onSetEditingCell(null, null, null);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSave();
    } else if (e.key === 'Escape') {
      handleCancel();
    }
  };

  if (isEditing) {
    return (
      <Input
        type="number"
        value={tempValue === null ? "" : tempValue}
        onChange={handleInputChange}
        onBlur={handleSave}
        onKeyDown={handleKeyDown}
        className="h-8 w-full max-w-[100px] text-right tabular-nums px-1 py-0.5 text-xs bg-background border-input focus:border-primary focus:ring-1 focus:ring-primary group-hover:border-primary"
        step={metricDef.step || "any"}
        ref={inputRef}
      />
    );
  }

  const isRelevantMetricForAggregated = AGGREGATED_METRIC_ROW_DEFINITIONS.some(def => def.key === metricDef.key);
  const isRelevantMetricForTeam = TEAM_METRIC_ROW_DEFINITIONS.some(def => def.key === metricDef.key && def.category !== 'Internal');

  const shouldDisplayMetric =
    (item.itemType === 'Team' && isRelevantMetricForTeam) ||
    ((item.itemType === 'BU' || item.itemType === 'LOB') && isRelevantMetricForAggregated);

  if (item.itemType === 'LOB' && metricDef.key === 'lobTotalBaseRequiredMinutes') {
    // This is handled in renderCapacityItemContent for LOBs
  }
  if (item.itemType === 'BU' && metricDef.key === 'lobTotalBaseRequiredMinutes') {
    // This is handled in renderCapacityItemContent for BUs
  }

  if (!shouldDisplayMetric || rawValue === null || rawValue === undefined) {
    const isEditableEmptyCell = canEditCell;
    return <div onClick={isEditableEmptyCell ? handleEditClick : undefined} className={`${isEditableEmptyCell ? 'cursor-pointer group relative' : 'relative'} w-full h-full flex items-center justify-end pr-1`}>
      <Minus className="h-4 w-4 text-muted-foreground mx-auto" />
      {isEditableEmptyCell && <Edit3 className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity absolute right-1 top-1/2 -translate-y-1/2" />}
    </div>;
  }

  let displayValue: React.ReactNode = "";
  let textColor = "text-foreground";
  let icon: React.ReactNode = null;
  let formulaText = "";

  const numValue = Number(rawValue);
  const teamMetrics = metricData as TeamPeriodicMetrics;
  const aggMetrics = metricData as AggregatedPeriodicMetrics;
  const standardWorkMinutesForPeriod = selectedTimeInterval === "Week" ? STANDARD_WEEKLY_WORK_MINUTES : STANDARD_MONTHLY_WORK_MINUTES;

  if (metricDef.isPercentage) {
    displayValue = `${numValue.toFixed(1)}%`;
  } else if (metricDef.isTime && metricDef.key === 'aht') {
    displayValue = `${numValue.toFixed(1)} min`;
  } else if (metricDef.isTime) {
    displayValue = `${numValue.toFixed(0)} min`;
  } else if (metricDef.isHC || ['requiredHC', 'actualHC', 'overUnderHC', 'moveIn', 'moveOut', 'newHireBatch', 'newHireProduction', 'attritionLossHC', 'hcAfterAttrition', 'endingHC'].includes(metricDef.key as string)) {
    displayValue = isNaN(numValue) ? '-' : Math.round(numValue).toString();
  } else if (metricDef.key === 'lobTotalBaseRequiredMinutes') {
    displayValue = `${numValue.toFixed(0)} min`;
  } else if (typeof numValue === 'number' && !isNaN(numValue)) {
    const fractionDigits = (['overUnderHC', 'requiredHC', 'actualHC'].includes(metricDef.key as string)) ? 2 : 1;
    displayValue = numValue.toLocaleString(undefined, { minimumFractionDigits: fractionDigits, maximumFractionDigits: fractionDigits });
  } else {
    displayValue = String(rawValue);
  }

  let baseTooltipText = `${item.name} - ${periodName}\n${metricDef.label}: ${displayValue}`;
  if (metricDef.description) {
    baseTooltipText += `\n\nDescription: ${metricDef.description}`;
  }

  if (item.itemType === 'Team') {
    switch (metricDef.key) {
      case 'requiredHC':
        if (teamMetrics?._calculatedRequiredAgentMinutes !== null && teamMetrics?._calculatedRequiredAgentMinutes !== undefined &&
          teamMetrics?.shrinkagePercentage !== null && teamMetrics?.occupancyPercentage !== null && teamMetrics?.occupancyPercentage > 0 && standardWorkMinutesForPeriod > 0) {
          const effMinsPerHC = standardWorkMinutesForPeriod *
            (1 - (teamMetrics.shrinkagePercentage / 100)) *
            (teamMetrics.occupancyPercentage / 100);
          if (effMinsPerHC > 0) {
            formulaText = `Formula: Eff. Req. Mins / (Std Mins * (1-Shrink%) * Occupancy%)\n` +
              `Calc: ${teamMetrics._calculatedRequiredAgentMinutes.toFixed(0)} / (${standardWorkMinutesForPeriod.toFixed(0)} * (1 - ${(teamMetrics.shrinkagePercentage / 100).toFixed(2)}) * ${(teamMetrics.occupancyPercentage / 100).toFixed(2)}) = ${numValue.toFixed(2)}\n` +
              `(Effective Mins per HC: ${effMinsPerHC.toFixed(0)})`;
          } else {
            formulaText = `Formula: Eff. Req. Mins / (Std Mins * (1-Shrink%) * Occupancy%)\n(Cannot calculate due to zero denominator component)`;
          }
        } else if (teamMetrics?._calculatedRequiredAgentMinutes === 0) {
          formulaText = `Formula: Eff. Req. Mins / (Std Mins * (1-Shrink%) * Occupancy%)\nCalculation: 0 / (...) = 0`;
        }
        break;
      case 'overUnderHC':
        if (teamMetrics?.actualHC !== null && teamMetrics?.actualHC !== undefined &&
          teamMetrics?.requiredHC !== null && teamMetrics?.requiredHC !== undefined) {
          formulaText = `Formula: Actual HC - Required HC\nCalc: ${teamMetrics.actualHC.toFixed(2)} - ${teamMetrics.requiredHC.toFixed(2)} = ${numValue.toFixed(2)}`;
        }
        break;
      case 'attritionLossHC':
        if (teamMetrics?.actualHC !== null && teamMetrics?.actualHC !== undefined &&
          teamMetrics?.attritionPercentage !== null && teamMetrics?.attritionPercentage !== undefined) {
          formulaText = `Formula: Actual HC * Attrition %\nCalc: ${teamMetrics.actualHC.toFixed(2)} * ${(teamMetrics.attritionPercentage / 100).toFixed(3)} = ${numValue.toFixed(2)}`;
        }
        break;
      case 'hcAfterAttrition':
        if (teamMetrics?.actualHC !== null && teamMetrics?.actualHC !== undefined &&
          teamMetrics?.attritionLossHC !== null && teamMetrics?.attritionLossHC !== undefined) {
          formulaText = `Formula: Actual HC - Attrition Loss HC\nCalc: ${teamMetrics.actualHC.toFixed(2)} - ${teamMetrics.attritionLossHC.toFixed(2)} = ${numValue.toFixed(2)}`;
        }
        break;
      case 'endingHC':
        if (teamMetrics?.hcAfterAttrition !== null && teamMetrics?.hcAfterAttrition !== undefined &&
          teamMetrics?.newHireProduction !== null && teamMetrics?.newHireProduction !== undefined &&
          teamMetrics?.moveIn !== null && teamMetrics?.moveIn !== undefined &&
          teamMetrics?.moveOut !== null && teamMetrics?.moveOut !== undefined) {
          formulaText = `Formula: HC After Attrition + New Hire Prod. + Move In - Move Out\n` +
            `Calc: ${teamMetrics.hcAfterAttrition.toFixed(2)} + ${teamMetrics.newHireProduction.toFixed(0)} + ${teamMetrics.moveIn.toFixed(0)} - ${teamMetrics.moveOut.toFixed(0)} = ${numValue.toFixed(2)}`;
        }
        break;
      case '_calculatedRequiredAgentMinutes':
      // metricData is available here and should be of type TeamPeriodicMetrics
      // It now contains _lobTotalBaseReqMinutesForCalc, volumeMixPercentage, and backlogPercentage
      const teamMetricData = metricData as TeamPeriodicMetrics; // Cast for type safety/clarity

      const lobBaseMins = teamMetricData?._lobTotalBaseReqMinutesForCalc;
      const volMix = teamMetricData?.volumeMixPercentage;
      const backlog = teamMetricData?.backlogPercentage;

      if (typeof lobBaseMins === 'number' &&
          typeof volMix === 'number' &&
          typeof backlog === 'number' &&
          numValue !== null && !isNaN(numValue)) { // numValue is the already calculated _calculatedRequiredAgentMinutes

        formulaText = `Formula: (LOB Total Base Req Mins * Team Vol Mix %) * (1 + Team Backlog %)
` +
                      `Calc: (${lobBaseMins.toFixed(0)} * ${(volMix / 100).toFixed(2)}) * (1 + ${(backlog / 100).toFixed(2)}) = ${numValue.toFixed(0)}
` +
                      `Represents team's share of LOB demand, adjusted for team's backlog.`;
      } else {
        // Fallback if some data is missing, though it ideally shouldn't be
        formulaText = `Formula: (LOB Total Base Req Mins * Team Vol Mix %) * (1 + Team Backlog %)
` +
                      `Represents team's share of LOB demand, adjusted for team's backlog. (Component data missing for full Calc display)`;
      }
      break;
      case '_calculatedActualProductiveAgentMinutes':
        if (teamMetrics?.actualHC !== null && teamMetrics?.actualHC !== undefined &&
          teamMetrics?.shrinkagePercentage !== null && teamMetrics?.occupancyPercentage !== null && standardWorkMinutesForPeriod > 0) {
          const prodMins = teamMetrics.actualHC * standardWorkMinutesForPeriod * (1 - (teamMetrics.shrinkagePercentage / 100)) * (teamMetrics.occupancyPercentage / 100);
          formulaText = `Formula: Actual HC * Std Mins * (1-Shrink%) * Occupancy%\n` +
            `Calc: ${teamMetrics.actualHC.toFixed(2)} * ${standardWorkMinutesForPeriod.toFixed(0)} * (1 - ${(teamMetrics.shrinkagePercentage / 100).toFixed(2)}) * ${(teamMetrics.occupancyPercentage / 100).toFixed(2)}) = ${prodMins.toFixed(0)}`;
        }
        break;
    }
  } else if (item.itemType === 'LOB' || item.itemType === 'BU') {
    switch (metricDef.key) {
      case 'overUnderHC':
        if (aggMetrics?.actualHC !== null && aggMetrics?.actualHC !== undefined &&
          aggMetrics?.requiredHC !== null && aggMetrics?.requiredHC !== undefined) {
          formulaText = `Formula: Aggregated Actual HC - Aggregated Required HC\nCalc: ${aggMetrics.actualHC.toFixed(2)} - ${aggMetrics.requiredHC.toFixed(2)} = ${numValue.toFixed(2)}`;
        }
        break;
      case 'requiredHC':
      case 'actualHC':
        if (item.children && item.children.length > 0) {
          const childType = item.itemType === 'BU' ? 'LOBs' : 'Teams';
          const childNames = item.children.map(child => child.name).join(', ');
          if (childNames) {
            formulaText = `Formula: Sum of ${metricDef.label} from child ${childType} (${childNames})`;
          } else {
            formulaText = `Formula: Sum of ${metricDef.label} from child ${childType}. (No children to sum from).`;
          }
        } else {
          formulaText = `Formula: Sum of ${metricDef.label} from child ${item.itemType === 'BU' ? 'LOBs' : 'Teams'}. (No children found).`;
        }
        break;
    }
  }

  if (metricDef.key === "overUnderHC") {
    if (numValue >= 0) { // Condition for Green
      textColor = "text-green-600"; // Green color
      icon = <ArrowUp className="h-3 w-3 inline-block ml-1" />;
    } else { // Conditions for Red or Amber (numValue < 0)
      icon = <ArrowDown className="h-3 w-3 inline-block ml-1" />;
      // Retrieve requiredHC for the specific conditions
      const requiredHCValue = metricData?.requiredHC; // Accessing requiredHC from the metricData of the current period

      if (typeof requiredHCValue === 'number' && requiredHCValue <= 10 && numValue === -1) {
        textColor = "text-red-600"; // Specific Red color
      } else if (typeof requiredHCValue === 'number' && requiredHCValue >= 100 && numValue === -1) {
        textColor = "text-yellow-500"; // Amber color
      } else {
        textColor = "text-destructive"; // Default Red for other negative values
      }
    }
  }

  let tooltipContent = baseTooltipText;
  if (formulaText) {
    tooltipContent += `\n\n${formulaText}`;
  }

  const cellDivContent = (
    <>
      {displayValue} {icon}
      {canEditCell && !isEditing && <Edit3 className="h-3 w-3 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity absolute right-1 top-1/2 -translate-y-1/2" />}
    </>
  );

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div
          onClick={canEditCell && !isEditing ? handleEditClick : undefined}
          className={`relative flex items-center justify-end gap-1 ${textColor} ${canEditCell ? 'cursor-pointer group' : ''} w-full h-full pr-1`}
        >
          {cellDivContent}
        </div>
      </TooltipTrigger>
      <TooltipContent className="whitespace-pre-wrap text-xs max-w-xs">
        <p>{tooltipContent}</p>
      </TooltipContent>
    </Tooltip>
  );
});

MetricCellContent.displayName = 'MetricCellContent';
export default MetricCellContent;
