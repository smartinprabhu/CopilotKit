import type {
  ExtendedTeamPeriodicMetrics,
  ModelCalculationContext
} from '../shared/interfaces';
import {
  STANDARD_WEEKLY_WORK_MINUTES,
  STANDARD_MONTHLY_WORK_MINUTES
} from '../../planComponents/capacity-insights/types';

/**
 * Safe incremental Erlang C calculation that avoids large factorials/pows.
 * servers: integer >= 1
 * a (trafficRate): can be fractional
 *
 * Returns a probability between 0 and 1.
 */
function erlangC(servers: number, a: number): number {
  // guards
  if (!isFinite(servers) || servers < 1) return 1;
  if (!isFinite(a) || a <= 0) return 0; // no traffic => no queue
  if (a >= servers) return 1; // overloaded

  // compute sum_{n=0}^{s-1} a^n / n! incrementally to avoid factorials
  let sum = 1; // n = 0 term
  let term = 1; // term = a^n / n!
  for (let n = 1; n <= servers - 1; n++) {
    term = term * (a / n); // update to a^n / n!
    sum += term;
    // small optimization: if term gets extremely tiny, break early
    if (term < 1e-15) break;
  }

  // compute term for n = s (a^s / s!)
  term = term * (a / (servers - 0)); // now term is a^s / s! (works when loop ran to s-1)
  // numeric guard if servers - a is tiny
  const denomServersMinusA = servers - a;
  if (Math.abs(denomServersMinusA) < 1e-9) {
    // extremely close to 100% utilisation -> treat as overloaded
    return 1;
  }

  const numerator = term * (servers / denomServersMinusA); // a^s/s! * (s/(s-a))
  const p0 = 1 / (sum + numerator); // normalization constant
  const C = numerator * p0; // Erlang C

  // clamp
  if (!isFinite(C)) return Math.min(Math.max(C, 0), 1);
  return Math.min(Math.max(C, 0), 1);
}

const MAX_ACCURACY = 0.999999;
const MAX_SERVERS_LIMIT = 2000; // hard limit for safety
const MAX_ITERATE_LIMIT = 5000; // hard limit for safety

export function fractionalAgents(
  SLA: number,
  serviceTime: number,
  callsPerHour: number,
  AHT: number
): number {
  try {
    // input coercion & guards
    SLA = Number(SLA) || 0;
    serviceTime = Number(serviceTime) || 0;
    callsPerHour = Number(callsPerHour) || 0;
    AHT = Number(AHT) || 1; // avoid divide by zero

    if (SLA > 1) SLA = 1;
    if (callsPerHour <= 0 || AHT <= 0) {
      console.warn("fractionalAgents: zero callsPerHour or AHT -> returning 0");
      return 0;
    }

    const birthRate = callsPerHour;
    const deathRate = 3600 / AHT; // service completions per hour? (kept your logic)
    if (deathRate <= 0) {
      console.warn("fractionalAgents: invalid deathRate -> returning 0");
      return 0;
    }
    const trafficRate = birthRate / deathRate; // 'a' in Erlang

    // approximate erlangs (original code used Fix((BirthRate * AHT)/3600 + 0.5))
    const erlangs = Math.floor((birthRate * AHT) / 3600 + 0.5);
    let noAgents = Math.max(1, Math.floor(erlangs));
    if (!isFinite(noAgents) || noAgents < 1) noAgents = 1;

    // make sure utilisation < 1 by bumping agents, but with a safety cap
    let utilisation = trafficRate / noAgents;
    let safetyCounter = 0;
    while (utilisation >= 1 && safetyCounter < 1000 && noAgents < MAX_SERVERS_LIMIT) {
      noAgents++;
      utilisation = trafficRate / noAgents;
      safetyCounter++;
    }
    if (utilisation >= 1) {
      // still overloaded after safety adjustments
      console.warn("fractionalAgents: system overloaded even after increasing agents.");
      return Number((noAgents).toFixed(4));
    }

    // iteration to find smallest integer agents that meets SLA
    let SLQueued = 0;
    const maxIterate = Math.min(noAgents * 100, MAX_ITERATE_LIMIT);
    let lastSLQ = 0;
    let iterCount = 0;

    for (let count = 1; count <= maxIterate && noAgents < MAX_SERVERS_LIMIT; count++) {
      iterCount++;
      lastSLQ = SLQueued;
      utilisation = trafficRate / noAgents;

      if (utilisation < 1) {
        const C = erlangC(noAgents, trafficRate);
        // the original formula used: 1 - C * exp((TrafficRate - Server) * ServiceTime / AHT)
        // Be careful about exponent magnitude: clamp the exponent input
        const expoArg = ((trafficRate - noAgents) * serviceTime) / Math.max(AHT, 1e-9);
        // if expoArg is very negative, exp(expoArg) -> 0; very positive -> huge; clamp
        const clampedExpoArg = Math.min(Math.max(expoArg, -50), 50);
        SLQueued = 1 - C * Math.exp(clampedExpoArg);

        // clamp SLQueued
        SLQueued = Math.min(Math.max(SLQueued, 0), 1);

        // check success
        if (SLQueued >= SLA || SLQueued > MAX_ACCURACY) {
          break;
        }
      }

      noAgents++;
      // safety: if agents grows beyond a sane limit, break
      if (noAgents >= MAX_SERVERS_LIMIT) {
        console.warn("fractionalAgents: reached MAX_SERVERS_LIMIT while iterating");
        break;
      }
    }

    // fractional interpolation if SLA falls between (lastSLQ, SLQueued)
    let noAgentsFloat = noAgents;
    if (SLQueued > SLA && SLQueued > lastSLQ) {
      const oneAgent = SLQueued - lastSLQ;
      if (Math.abs(oneAgent) < 1e-12) {
        // can't compute fractional reliably, return integer
        noAgentsFloat = noAgents;
      } else {
        const fract = SLA - lastSLQ;
        noAgentsFloat = (fract / oneAgent) + (noAgents - 1);
      }
    }

    return Number(noAgentsFloat.toFixed(4));
  } catch (err) {
    console.error("Error in fractionalAgents:", err);
    return 0;
  }
}

export function calculateBillableHoursTeamMetricsForPeriod(
  teamInput: Partial<ExtendedTeamPeriodicMetrics>,
  lobTotalBaseMinutes: number | null,
  standardWorkMinutes: number = STANDARD_WEEKLY_WORK_MINUTES,
  volume: number,
  metricReq: number,
  actualHc: number,
  lastHc: number,
  isBPO: boolean,
  lob: string,
): ExtendedTeamPeriodicMetrics {
  const defaults: ExtendedTeamPeriodicMetrics = { ...teamInput };

  const volumeFactor = (volume * (defaults.volumeMixPercentage / 100));
  const inOfficeFactor = (1 - (defaults.inOfficeShrinkagePercentage / 100));
  const outOfficeFactor = (1 - (defaults.outOfOfficeShrinkagePercentage / 100));
  const baseMins = (volumeFactor / inOfficeFactor / outOfficeFactor) / 40;

  const callsPerHour = volume * ((defaults.volumeMixPercentage / 100) / defaults.hoop);
  const avgHandleTime = lob === 'Phone' ? defaults.aht * 60 : (defaults.aht * 60) / defaults.concurrency;

  const getFractionalAgents = fractionalAgents(
    defaults.serviceLevel / 100,
    defaults.serviceTime,
    callsPerHour,
    avgHandleTime
  );

  let reqHcResult =
    (getFractionalAgents /
      (1 - defaults.inOfficeShrinkagePercentage / 100) /
      (1 - defaults.outOfOfficeShrinkagePercentage / 100)) *
    (defaults.hoop / 40);

  if (lob !== 'Phone' && lob !== 'Chat') {
    reqHcResult =
      ((volume * defaults.aht) / 60 /
        (defaults.occupancyPercentage / 100) /
        (1 - defaults.inOfficeShrinkagePercentage / 100) /
        (1 - defaults.outOfOfficeShrinkagePercentage / 100)) *
      (1 + defaults.backlogPercentage / 100) / 40;
  }


  // Simplified FTE calculation (produces ~25% lower requirements)
  // const simplifiedFTE = baseMins > 0 ? baseMins : null;
  defaults.requiredHC = reqHcResult;

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

  if (!(defaults.actualHC > 0)) {
    defaults.actualHC = actualHCValue;
  }

  // Over/Under calculation
  defaults.overUnderHC = (defaults.actualHC !== null && reqHcResult !== null)
    ? defaults.actualHC - reqHcResult
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