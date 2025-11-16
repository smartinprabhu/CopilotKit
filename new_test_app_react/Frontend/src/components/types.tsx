import { getWeek, startOfWeek, endOfWeek, isAfter, isBefore, addDays, isSameDay, addWeeks, startOfMonth, endOfMonth, getYear, getMonth, parse as dateParseFns } from 'date-fns';

// Export your custom types and constants here
export const STANDARD_WEEKLY_WORK_MINUTES = 40 * 60;
export const STANDARD_MONTHLY_WORK_MINUTES = (40 * 52 / 12) * 60;

export const isLeapYear = (year: number): boolean => {
  return (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
};

export const formatDatePartUTC = (date: Date): string =>
  `${(date.getUTCMonth() + 1).toString().padStart(2, '0')}/${date.getUTCDate().toString().padStart(2, '0')}`;

export const generateFiscalWeekHeaders = (startFiscalYear: number, numTotalWeeks: number): string[] => {
  const headers: string[] = [];

  let currentYear = startFiscalYear;
  let fiscalYearActualStartDate: Date;

  const targetDateForStart = isLeapYear(currentYear)
    ? new Date(Date.UTC(currentYear, 1, 1)) // February 1st
    : new Date(Date.UTC(currentYear, 0, 22)); // January 22nd

  fiscalYearActualStartDate = startOfWeek(targetDateForStart, { weekStartsOn: 1 });

  for (let i = 0; i < numTotalWeeks; i++) {
    const weekStartDate = new Date(fiscalYearActualStartDate);
    weekStartDate.setUTCDate(fiscalYearActualStartDate.getUTCDate() + i * 7);

    const weekEndDate = new Date(weekStartDate);
    weekEndDate.setUTCDate(weekStartDate.getUTCDate() + 6);

    const displayYearForHeader = weekStartDate.getUTCFullYear();

    headers.push(
      `FWk${i + 1}: ${formatDatePartUTC(weekStartDate)}-${formatDatePartUTC(weekEndDate)} (${displayYearForHeader})`
    );
  }
  return headers;
};

export const ALL_WEEKS_HEADERS = generateFiscalWeekHeaders(2024, 104);

export const ALL_MONTH_HEADERS = Array.from({ length: 24 }, (_, i) => {
  const year = 2024 + Math.floor(i / 12);
  const month = i % 12;
  const date = new Date(year, month, 1);
  return date.toLocaleString('default', { month: 'long', year: 'numeric' });
});

export const BUSINESS_UNIT_CONFIG = {
  "WFS": {
    name: "WFS",
    lonsOfBusiness: [
      "US Chat", "US Phone", "Core Support", "Customer Returns", "Inventory Management",
      "Dispute Management", "IBE Management", "FC Liaison", "Flex Team", "Help Desk", "MCS",
      "China Mandarin Chat", "China Mandarin Email", "China English Chat", "China English Email",
      "Strike Through", "Walmart Import"
    ]
  },
  "SFF": {
    name: "SFF",
    lonsOfBusiness: ["SFF LoB Alpha", "SFF LoB Bravo", "SFF LoB Charlie", "SFF LoB Delta"]
  },
  "RSO": {
    name: "RSO",
    lonsOfBusiness: ["RSO LoB Xray", "RSO LoB Yankee", "RSO LoB Zulu"]
  },
  "Go Local": {
    name: "Go Local",
    lonsOfBusiness: ["GoLocal Partner Support", "GoLocal Customer Care", "GoLocal Dispatch"]
  }
} as const;

export type BusinessUnitName = keyof typeof BUSINESS_UNIT_CONFIG;
export type LineOfBusinessName<BU extends BusinessUnitName = BusinessUnitName> = typeof BUSINESS_UNIT_CONFIG[BU]["lonsOfBusiness"][number];

export const ALL_BUSINESS_UNITS = Object.keys(BUSINESS_UNIT_CONFIG) as BusinessUnitName[];
export const ALL_TEAM_NAMES: TeamName[] = ["Inhouse", "BPO1", "BPO2"];

export type TimeInterval = "Week" | "Month";
export type TeamName = "Inhouse" | "BPO1" | "BPO2";

// Export other custom types and interfaces here
export interface BaseHCValues {
  requiredHC: number | null;
  actualHC: number | null;
  overUnderHC: number | null;
}

export interface TeamPeriodicMetrics extends BaseHCValues {
  aht: number | null;
  shrinkagePercentage: number | null;
  occupancyPercentage: number | null;
  backlogPercentage: number | null;
  attritionPercentage: number | null;
  volumeMixPercentage: number | null;
  moveIn: number | null;
  moveOut: number | null;
  newHireBatch: number | null;
  newHireProduction: number | null;
  _productivity: number | null;
  _calculatedRequiredAgentMinutes?: number | null;
  _calculatedActualProductiveAgentMinutes?: number | null;
  attritionLossHC?: number | null;
  hcAfterAttrition?: number | null;
  endingHC?: number | null;
}

export interface AggregatedPeriodicMetrics extends BaseHCValues {
  lobTotalBaseRequiredMinutes?: number | null;
}

export interface MetricDefinition {
  key: keyof TeamPeriodicMetrics | keyof AggregatedPeriodicMetrics;
  label: string;
  isPercentage?: boolean;
  isHC?: boolean;
  isTime?: boolean;
  isEditableForTeam?: boolean;
  isDisplayOnly?: boolean;
  step?: string | number;
  category?: 'PrimaryHC' | 'Assumption' | 'HCAdjustment' | 'Internal';
  description?: string;
}

export type TeamMetricDefinitions = MetricDefinition[];
export type AggregatedMetricDefinitions = MetricDefinition[];

export const TEAM_METRIC_ROW_DEFINITIONS: TeamMetricDefinitions = [
  { key: "requiredHC", label: "Required HC", isHC: true, isDisplayOnly: true, category: 'PrimaryHC', description: "Calculated number of headcount required based on demand and productivity assumptions." },
  { key: "actualHC", label: "Actual/Starting HC", isHC: true, isEditableForTeam: true, step: 0.01, category: 'PrimaryHC', description: "The actual or starting headcount for the period before adjustments." },
  { key: "overUnderHC", label: "Over/Under HC", isHC: true, isDisplayOnly: true, category: 'PrimaryHC', description: "Difference between Actual/Starting HC and Required HC." },
  { key: "aht", label: "AHT", isTime: true, isEditableForTeam: true, step: 0.1, category: 'Assumption', description: "Average Handle Time: The average time taken to handle one interaction." },
  { key: "shrinkagePercentage", label: "Shrinkage %", isPercentage: true, isEditableForTeam: true, step: 0.1, category: 'Assumption', description: "Shrinkage: Percentage of paid time that agents are not available for handling interactions (e.g., breaks, training, meetings)." },
  { key: "occupancyPercentage", label: "Occupancy %", isPercentage: true, isEditableForTeam: true, step: 0.1, category: 'Assumption', description: "Occupancy: Percentage of time agents are busy with interaction-related work during their available time." },
  { key: "backlogPercentage", label: "Backlog %", isPercentage: true, isEditableForTeam: true, step: 0.1, category: 'Assumption', description: "Backlog: Percentage of additional work (e.g., deferred tasks) that needs to be handled on top of forecasted volume." },
  { key: "attritionPercentage", label: "Attrition %", isPercentage: true, isEditableForTeam: true, step: 0.1, category: 'Assumption', description: "Attrition: Percentage of agents expected to leave during the period." },
  { key: "volumeMixPercentage", label: "Volume Mix %", isPercentage: true, isEditableForTeam: true, step: 0.1, category: 'Assumption', description: "Volume Mix: Percentage of the LOB's total volume handled by this team." },
  { key: "moveIn", label: "Move In (+)", isEditableForTeam: true, step: 1, isHC: true, category: 'HCAdjustment', description: "Headcount moving into this team from other teams or roles." },
  { key: "moveOut", label: "Move Out (-)", isEditableForTeam: true, step: 1, isHC: true, category: 'HCAdjustment', description: "Headcount moving out of this team to other teams or roles." },
  { key: "newHireBatch", label: "New Hire Batch", isEditableForTeam: true, step: 1, isHC: true, category: 'HCAdjustment', description: "Number of new hires starting in a batch during this period (typically in training)." },
  { key: "newHireProduction", label: "New Hire Production", isEditableForTeam: true, step: 1, isHC: true, category: 'HCAdjustment', description: "Number of new hires becoming productive and joining the floor during this period." },
  { key: "attritionLossHC", label: "Attrition Loss HC", isHC: true, isDisplayOnly: true, category: 'HCAdjustment', description: "Calculated headcount lost due to attrition." },
  { key: "hcAfterAttrition", label: "HC After Attrition", isHC: true, isDisplayOnly: true, category: 'HCAdjustment', description: "Headcount remaining after attrition loss." },
  { key: "endingHC", label: "Ending HC", isHC: true, isDisplayOnly: true, category: 'HCAdjustment', description: "Projected headcount at the end of the period after all adjustments." },
  { key: "_calculatedRequiredAgentMinutes", label: "Eff. Req. Mins (Team)", isDisplayOnly: true, isTime: true, category: 'Internal', description: "Team's share of LOB demand minutes, adjusted for the team's backlog percentage." },
  { key: "_calculatedActualProductiveAgentMinutes", label: "Actual Prod. Mins (Team)", isDisplayOnly: true, isTime: true, category: 'Internal', description: "Total productive agent minutes available from the team's actual headcount, considering shrinkage and occupancy." },
];

export const AGGREGATED_METRIC_ROW_DEFINITIONS: AggregatedMetricDefinitions = [
  { key: "requiredHC", label: "Required HC", isHC: true, description: "Total required headcount for the Business Unit or Line of Business, aggregated from child entities." },
  { key: "actualHC", label: "Actual HC", isHC: true, description: "Total actual/starting headcount for the Business Unit or Line of Business, aggregated from child entities." },
  { key: "overUnderHC", label: "Over/Under HC", isHC: true, description: "Total Over/Under headcount for the Business Unit or Line of Business, based on aggregated Actual and Required HC." },
  { key: "lobTotalBaseRequiredMinutes", label: "LOB Total Base Req Mins", isTime: true, description: "Total base agent minutes required for the LOB, calculated from Volume x AHT or direct input. This metric is not displayed for Business Units." },
];

export interface FilterOptions {
  businessUnits: BusinessUnitName[];
  linesOfBusiness: string[];
}

export interface HeaderSectionProps {
  filterOptions: FilterOptions;
  selectedBusinessUnit: BusinessUnitName;
  onSelectBusinessUnit: (value: BusinessUnitName) => void;
  selectedLineOfBusiness: string[];
  onSelectLineOfBusiness: (value: string[]) => void;
  selectedTimeInterval: TimeInterval;
  onSelectTimeInterval: (value: TimeInterval) => void;
  selectedDateRange: DateRange | undefined;
  onSelectDateRange: (value: DateRange | undefined) => void;
  allAvailablePeriods: string[];
  displayedPeriodHeaders: string[];
  activeHierarchyContext: string;
  headerPeriodScrollerRef: React.RefObject<HTMLDivElement>;
}

export interface RawTeamDataEntry {
  teamName: TeamName;
  periodicInputData: Record<string, Partial<Omit<TeamPeriodicMetrics, 'requiredHC' | 'overUnderHC' | '_calculatedRequiredAgentMinutes' | '_calculatedActualProductiveAgentMinutes' | 'attritionLossHC' | 'hcAfterAttrition' | 'endingHC'>>>;
}

export interface RawLoBCapacityEntry {
  id: string;
  bu: BusinessUnitName;
  lob: LineOfBusinessName<BusinessUnitName>;
  lobVolumeForecast: Record<string, number | null>;
  lobAverageAHT: Record<string, number | null>;
  lobTotalBaseRequiredMinutes?: Record<string, number | null>;
  teams: RawTeamDataEntry[];
}

export interface CapacityDataRow {
  id: string;
  name: string;
  level: number;
  itemType: 'BU' | 'LOB' | 'Team';
  periodicData: Record<string, AggregatedPeriodicMetrics | TeamPeriodicMetrics>;
  children?: CapacityDataRow[];
  lobId?: string;
}

export const calculateTeamMetricsForPeriod = (
  teamInputDataCurrentPeriod: Partial<TeamPeriodicMetrics>,
  lobTotalBaseRequiredMinutesForPeriod: number | null,
  standardWorkMinutesForPeriod: number
): TeamPeriodicMetrics => {
  const defaults: TeamPeriodicMetrics = {
    aht: null, shrinkagePercentage: null, occupancyPercentage: null, backlogPercentage: null,
    attritionPercentage: null, volumeMixPercentage: null, actualHC: null, moveIn: null,
    moveOut: null, newHireBatch: null, newHireProduction: null, _productivity: null,
    _calculatedRequiredAgentMinutes: null,
    _calculatedActualProductiveAgentMinutes: null,
    requiredHC: null,
    overUnderHC: null,
    attritionLossHC: null,
    hcAfterAttrition: null,
    endingHC: null,
    ...(teamInputDataCurrentPeriod || {}),
  };

  const baseTeamRequiredMinutes = (lobTotalBaseRequiredMinutesForPeriod ?? 0) * ((defaults.volumeMixPercentage ?? 0) / 100);
  const effectiveTeamRequiredMinutes = baseTeamRequiredMinutes * (1 + ((defaults.backlogPercentage ?? 0) / 100));
  defaults._calculatedRequiredAgentMinutes = effectiveTeamRequiredMinutes;

  let requiredHC = null;
  if (effectiveTeamRequiredMinutes > 0 && standardWorkMinutesForPeriod > 0 && defaults.shrinkagePercentage !== null && defaults.occupancyPercentage !== null && defaults.occupancyPercentage > 0) {
    const effectiveMinutesPerHC = standardWorkMinutesForPeriod *
      (1 - (defaults.shrinkagePercentage / 100)) *
      (defaults.occupancyPercentage / 100);
    if (effectiveMinutesPerHC > 0) {
      requiredHC = effectiveTeamRequiredMinutes / effectiveMinutesPerHC;
    }
  } else if (effectiveTeamRequiredMinutes === 0) {
    requiredHC = 0;
  }
  defaults.requiredHC = requiredHC;

  const currentActualHC = defaults.actualHC ?? 0;
  defaults.overUnderHC = (currentActualHC !== null && requiredHC !== null) ? currentActualHC - requiredHC : null;

  if (currentActualHC !== null && standardWorkMinutesForPeriod > 0 && defaults.shrinkagePercentage !== null && defaults.occupancyPercentage !== null) {
    defaults._calculatedActualProductiveAgentMinutes = currentActualHC * standardWorkMinutesForPeriod *
      (1 - (defaults.shrinkagePercentage / 100)) *
      (defaults.occupancyPercentage / 100);
  } else {
    defaults._calculatedActualProductiveAgentMinutes = 0;
  }

  const attritionLossHC = currentActualHC * ((defaults.attritionPercentage ?? 0) / 100);
  defaults.attritionLossHC = attritionLossHC;

  const hcAfterAttrition = currentActualHC - attritionLossHC;
  defaults.hcAfterAttrition = hcAfterAttrition;

  defaults.endingHC = hcAfterAttrition + (defaults.newHireProduction ?? 0) + (defaults.moveIn ?? 0) - (defaults.moveOut ?? 0);

  return defaults;
};

export const parseDateFromHeaderStringMMDD = (dateMMDD: string, year: string): Date | null => {
  if (!dateMMDD || !year) return null;
  const [month, day] = dateMMDD.split('/').map(Number);
  if (isNaN(month) || isNaN(day) || isNaN(parseInt(year))) return null;

  const parsedDate = new Date(Date.UTC(parseInt(year), month - 1, day));

  if (parsedDate.getUTCFullYear() !== parseInt(year) || parsedDate.getUTCMonth() !== month - 1 || parsedDate.getUTCDate() !== day) {
    return null;
  }
  return parsedDate;
};

export const getHeaderDateRange = (header: string, interval: TimeInterval): { startDate: Date | null, endDate: Date | null } => {
  if (interval === "Week") {
    const match = header.match(/FWk\d+:\s*(\d{2}\/\d{2})-(\d{2}\/\d{2})\s*\((\d{4})\)/);
    if (match) {
      const [, startDateStr, endDateStr, yearStr] = match;

      let parsedStartDate = parseDateFromHeaderStringMMDD(startDateStr, yearStr);
      let parsedEndDate = parseDateFromHeaderStringMMDD(endDateStr, yearStr);

      if (parsedStartDate && parsedEndDate && isBefore(parsedEndDate, parsedStartDate)) {
        const startMonth = parseInt(startDateStr.split('/')[0]);
        const endMonth = parseInt(endDateStr.split('/')[0]);
        if (endMonth < startMonth) {
          parsedEndDate = parseDateFromHeaderStringMMDD(endDateStr, (parseInt(yearStr) + 1).toString());
        }
      }
      return { startDate: parsedStartDate, endDate: parsedEndDate };
    }
  } else if (interval === "Month") {
    try {
      const date = dateParseFns(header, "MMMM yyyy", new Date());
      if (!isNaN(date.getTime())) {
        const yearVal = getYear(date);
        const monthVal = getMonth(date);
        const firstDay = startOfMonth(new Date(yearVal, monthVal));
        const lastDay = endOfMonth(new Date(yearVal, monthVal));
        return { startDate: firstDay, endDate: lastDay };
      }
    } catch (e) {
      console.warn(`Could not parse month header: ${header}`, e);
    }
  }
  return { startDate: null, endDate: null };
};

export const getDefaultDateRange = (interval: TimeInterval): DateRange => {
  const headers = interval === "Week" ? ALL_WEEKS_HEADERS : ALL_MONTH_HEADERS;
  const numPeriodsToDefault = interval === "Week" ? 11 : 2;

  if (headers.length === 0) return { from: undefined, to: undefined };

  const fromHeaderDetails = getHeaderDateRange(headers[0], interval);
  const toHeaderIndex = Math.min(numPeriodsToDefault, headers.length - 1);
  const toHeaderDetails = getHeaderDateRange(headers[toHeaderIndex], interval);

  let fromDate = fromHeaderDetails.startDate;
  let toDate = toHeaderDetails.endDate;

  if (!fromDate) fromDate = new Date();
  if (!toDate) toDate = interval === "Week" ? endOfWeek(addWeeks(fromDate, 11)) : endOfMonth(addDays(startOfMonth(fromDate), 60));

  return { from: fromDate ?? undefined, to: toDate ?? undefined };
};

export const findFiscalWeekHeaderForDate = (targetDate: Date, allFiscalHeaders: string[]): string | null => {
  if (!targetDate) return null;
  for (const header of allFiscalHeaders) {
    const { startDate, endDate } = getHeaderDateRange(header, "Week");
    if (startDate && endDate) {
      const targetDayOnly = new Date(Date.UTC(targetDate.getUTCFullYear(), targetDate.getUTCMonth(), targetDate.getUTCDate()));
      const sDateOnly = new Date(Date.UTC(startDate.getUTCFullYear(), startDate.getUTCMonth(), startDate.getUTCDate()));
      const eDateOnly = new Date(Date.UTC(endDate.getUTCFullYear(), endDate.getUTCMonth(), endDate.getUTCDate()));

      if (targetDayOnly >= sDateOnly && targetDayOnly <= eDateOnly) {
        return header;
      }
    }
  }
  return null;
};

const generateTeamPeriodicInputData = (periods: string[], teamIndex: number, totalTeams: number): Record<string, Partial<Omit<TeamPeriodicMetrics, 'requiredHC' | 'overUnderHC' | '_calculatedRequiredAgentMinutes' | '_calculatedActualProductiveAgentMinutes' | 'attritionLossHC' | 'hcAfterAttrition' | 'endingHC'>>> => {
  const metrics: Record<string, Partial<Omit<TeamPeriodicMetrics, 'requiredHC' | 'overUnderHC' | '_calculatedRequiredAgentMinutes' | '_calculatedActualProductiveAgentMinutes' | 'attritionLossHC' | 'hcAfterAttrition' | 'endingHC'>>> = {};

  let initialMix = totalTeams > 0 ? parseFloat((100 / totalTeams).toFixed(1)) : 0;
  if (totalTeams === 3) {
    initialMix = 33.3;
  }
  let sumOfMix = 0;
  const mixes = Array(totalTeams).fill(0).map((_, idx) => {
    if (idx === totalTeams - 1) {
      return Math.max(0, parseFloat((100 - sumOfMix).toFixed(1)));
    }
    const currentMix = initialMix;
    sumOfMix += currentMix;
    return currentMix;
  });

  const finalSumCheck = mixes.reduce((acc, curr) => acc + curr, 0);
  if (Math.abs(finalSumCheck - 100) > 0.01 && totalTeams > 0 && mixes.length > 0) {
    const diff = 100 - finalSumCheck;
    mixes[mixes.length - 1] = parseFloat((mixes[mixes.length - 1] + diff).toFixed(1));
    if (mixes[mixes.length - 1] < 0 && mixes.length > 1) {
      let diffToRedistribute = mixes[mixes.length - 1];
      mixes[mixes.length - 1] = 0;
      for (let k = 0; k < mixes.length - 1; k++) {
        if (diffToRedistribute >= 0) break;
        let take = Math.min(mixes[k], Math.abs(diffToRedistribute));
        mixes[k] = parseFloat((mixes[k] - take).toFixed(1));
        diffToRedistribute += take;
      }
    }
  }

  periods.forEach(period => {
    metrics[period] = {
      aht: Math.floor(Math.random() * 10) + 5,
      shrinkagePercentage: Math.floor(Math.random() * 15) + 5,
      occupancyPercentage: Math.floor(Math.random() * 20) + 70,
      backlogPercentage: Math.floor(Math.random() * 10),
      attritionPercentage: parseFloat((Math.random() * 2).toFixed(1)),
      volumeMixPercentage: mixes[teamIndex] !== undefined ? mixes[teamIndex] : (totalTeams > 0 ? parseFloat((100 / totalTeams).toFixed(1)) : 0),
      actualHC: Math.floor(Math.random() * 50) + 10,
      moveIn: Math.floor(Math.random() * 5),
      moveOut: Math.floor(Math.random() * 3),
      newHireBatch: Math.random() > 0.7 ? Math.floor(Math.random() * 10) + 5 : 0,
      newHireProduction: Math.random() > 0.5 ? Math.floor(Math.random() * 8) : 0,
      _productivity: 1.0,
    };
  });
  return metrics;
};

const generateLobInputs = (periods: string[]): {
  volume: Record<string, number | null>,
  aht: Record<string, number | null>
} => {
  const volume: Record<string, number | null> = {};
  const aht: Record<string, number | null> = {};
  periods.forEach(period => {
    const currentVolume = Math.floor(Math.random() * 10000) + 2000;
    const currentAHT = Math.floor(Math.random() * 10) + 5;
    volume[period] = currentVolume;
    aht[period] = currentAHT;
  });
  return { volume, aht };
};

export const initialMockRawCapacityData: RawLoBCapacityEntry[] = [];
ALL_BUSINESS_UNITS.forEach(bu => {
  BUSINESS_UNIT_CONFIG[bu].lonsOfBusiness.forEach(lob => {
    const teamsForLob: RawTeamDataEntry[] = [];
    const numTeams = ALL_TEAM_NAMES.length;

    ALL_TEAM_NAMES.forEach((teamName, index) => {
      teamsForLob.push({
        teamName: teamName,
        periodicInputData: generateTeamPeriodicInputData(ALL_WEEKS_HEADERS, index, numTeams),
      });
    });
    const lobInputs = generateLobInputs(ALL_WEEKS_HEADERS);

    initialMockRawCapacityData.push({
      id: `${bu.toLowerCase().replace(/\s+/g, '-')}_${lob.toLowerCase().replace(/\s+/g, '-')}`,
      bu: bu,
      lob: lob,
      lobVolumeForecast: lobInputs.volume,
      lobAverageAHT: lobInputs.aht,
      teams: teamsForLob,
    });
  });
});
