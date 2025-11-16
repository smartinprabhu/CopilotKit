import type { TeamPeriodicMetrics } from "@/components/capacity-insights/types";
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

export function calculateTeamMetricsForPeriod(
  teamInputDataCurrentPeriod: Partial<TeamPeriodicMetrics>,
  lobTotalBaseRequiredMinutesForPeriod: number | null,
  standardWorkMinutesForPeriod: number,
  volumeData: number,
  reqHc: number,
  actualHc: number,
  lastHc: number,
  isBPO: boolean,
  lob: string,
): TeamPeriodicMetrics {
  const defaults: TeamPeriodicMetrics = {
    aht: null,
    inOfficeShrinkagePercentage: null,
    outOfOfficeShrinkagePercentage: null,
    serviceLevel: null,
    serviceTime: null,
    hoop: null,
    concurrency: null,
    occupancyPercentage: null,
    backlogPercentage: null,
    attritionPercentage: null,
    volumeMixPercentage: null,
    actualHC: null,
    moveIn: null,
    moveOut: null,
    newHireBatch: null,
    newHireProduction: null,
    lobVolumeForecast: null,
    handlingCapacity: null,
    _productivity: null,
    _calculatedRequiredAgentMinutes: null,
    _calculatedActualProductiveAgentMinutes: null,
    requiredHC: null,
    overUnderHC: null,
    attritionLossHC: null,
    hcAfterAttrition: null,
    endingHC: null,
    _lobTotalBaseReqMinutesForCalc: null,
    ...teamInputDataCurrentPeriod,
  };


  const baseTeamRequiredMinutes = (lobTotalBaseRequiredMinutesForPeriod ?? 0) * ((defaults.volumeMixPercentage ?? 0) / 100);
  const effectiveTeamRequiredMinutes = baseTeamRequiredMinutes * (1 + ((defaults.backlogPercentage ?? 0) / 100));
  defaults._calculatedRequiredAgentMinutes = effectiveTeamRequiredMinutes;

  let requiredHC = null;
  let actualHCValue = null;

  const callsPerHour = volumeData * ((defaults.volumeMixPercentage / 100) / defaults.hoop);
  const avgHandleTime = lob === 'Phone' ? defaults.aht * 60 : (defaults.aht * 60) / defaults.concurrency; // convert minutes → seconds

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
      ((volumeData * defaults.aht) / 60 /
        (defaults.occupancyPercentage / 100) /
        (1 - defaults.inOfficeShrinkagePercentage / 100) /
        (1 - defaults.outOfOfficeShrinkagePercentage / 100)) *
      (1 + defaults.backlogPercentage / 100) / 40;
  }

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


  if (!isNaN(reqHcResult) && Math.abs(reqHcResult) > 0) {
    requiredHC = Math.abs(reqHcResult);
  } else {
    requiredHC = 0;
  }

  if (!isNaN(actualHCValue) && Math.abs(actualHCValue) > 0) {
    actualHCValue = Math.abs(actualHCValue);
  } else {
    actualHCValue = 0;
  }

  defaults.requiredHC = requiredHC;
  if (defaults.actualHC === 0) {
    defaults.actualHC = actualHCValue;
  }
  const currentActualHC = defaults.actualHC ?? 0;
  defaults.overUnderHC = (currentActualHC !== null && requiredHC !== null) ? currentActualHC - requiredHC : null;

  if (currentActualHC !== null && standardWorkMinutesForPeriod > 0) {
    defaults._calculatedActualProductiveAgentMinutes = currentActualHC * standardWorkMinutesForPeriod *
      (1 - ((defaults.inOfficeShrinkagePercentage ?? 0) / 100)) *
      (1 - ((defaults.outOfOfficeShrinkagePercentage ?? 0) / 100)) *
      ((defaults.occupancyPercentage ?? 0) / 100);
  } else {
    defaults._calculatedActualProductiveAgentMinutes = 0;
  }

  const attritionLossHC = currentActualHC * ((defaults.attritionPercentage ?? 0) / 100);
  defaults.attritionLossHC = defaults.attritionLossHC || 0;
  defaults.newHireBatchTrain = defaults.newHireBatchTrain || 0;
  const hcAfterAttrition = currentActualHC - attritionLossHC;
  defaults.hcAfterAttrition = hcAfterAttrition;
  defaults.endingHC = hcAfterAttrition + (defaults.newHireProduction ?? 0) + (defaults.moveIn ?? 0) - (defaults.moveOut ?? 0);
  defaults._lobTotalBaseReqMinutesForCalc = lobTotalBaseRequiredMinutesForPeriod;

  return defaults;
}
