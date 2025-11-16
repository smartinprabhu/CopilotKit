import type { 
  ExtendedTeamPeriodicMetrics,
  ModelCalculationContext 
} from '../shared/interfaces';
import { 
  STANDARD_WEEKLY_WORK_MINUTES, 
  STANDARD_MONTHLY_WORK_MINUTES 
} from '../../planComponents/capacity-insights/types';

export function calculateFixHCTeamMetricsForPeriod(
  teamInput: Partial<ExtendedTeamPeriodicMetrics>,
  lobTotalBaseMinutes: number | null,
  standardWorkMinutes: number = STANDARD_WEEKLY_WORK_MINUTES,
  volume: number,
  metricReq: number,
  actualHc: number,
  lastHc: number,
  isBPO: boolean
): ExtendedTeamPeriodicMetrics {
  const defaults: ExtendedTeamPeriodicMetrics = { ...teamInput };

    const inOfficeFactor = (1 - (defaults.inOfficeShrinkagePercentage / 100));
    const outOfficeFactor = (1 - (defaults.outOfOfficeShrinkagePercentage / 100));
    const baseMins = (metricReq / inOfficeFactor / outOfficeFactor) * (defaults.volumeMixPercentage / 100);

  // Simplified FTE calculation (produces ~25% lower requirements)
  const simplifiedFTE = baseMins > 0 ? baseMins : null;
  defaults.requiredHC = simplifiedFTE;

    let actualHCValue = null;

  if (actualHc > 0) {
    actualHCValue = actualHc;
  } else if (lastHc && !isBPO) {
    actualHCValue = (
      (lastHc * (1 - defaults.attritionPercentage / 100)) +
      (
        defaults.newHireProduction +
        defaults.moveIn +
        defaults.moveOut
      )
    );
  } else if (lastHc && isBPO) {
    actualHCValue = lastHc + defaults.moveIn + defaults.moveOut + defaults.newHireProduction
  }

  if (!isNaN(actualHCValue) && Math.abs(actualHCValue) > 0) {
    actualHCValue = Math.abs(actualHCValue);
  } else {
    actualHCValue = 0;
  }

  if(!(defaults.actualHC > 0)){
    defaults.actualHC = actualHCValue;
  }

  // Over/Under calculation
  defaults.overUnderHC = (defaults.actualHC !== null && simplifiedFTE !== null) 
    ? defaults.actualHC - simplifiedFTE 
    : null;

  // HC flow calculations (same as other models)
  if (defaults.actualHC !== null && defaults.attritionPercentage !== null) {
    defaults.attritionLossHC = defaults.actualHC * (defaults.attritionPercentage / 100);
    defaults.hcAfterAttrition = defaults.actualHC - defaults.attritionLossHC;
    
    const moveIn = defaults.moveIn ?? 0;
    const moveOut = defaults.moveOut ?? 0;
    const newHireProduction = defaults.newHireProduction ?? 0;
    
    defaults.endingHC = defaults.hcAfterAttrition + newHireProduction + moveIn - moveOut;
  }

  return defaults;
}