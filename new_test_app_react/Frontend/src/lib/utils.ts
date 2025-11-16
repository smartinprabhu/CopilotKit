
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import { parse, isWithinInterval } from "date-fns";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatWithPrecision(value, precision) {
  if (typeof value !== 'number' || typeof precision !== 'number') return null;
  return precision === 0 ? Math.round(value) : parseFloat(value.toFixed(precision));
}

export function formatNumber(value: number, precision): string {
  if (value >= 1_000_000) {
    return precision > 0 ? `${(value / 1_000_000).toFixed(2)}M` : `${Math.round(value / 1_000_000)}M`;
  } else if (value >= 1_000) {
    return precision > 0 ? `${(value / 1_000).toFixed(2)}k` : `${Math.round(value / 1_000)}k`;
  }
  return value.toString();
}

// Get the fiscal year start and end for any given date
export function getFiscalYearRange(date = new Date(), isForm = false) {
  const year = date.getMonth() >= 3 ? date.getFullYear() : date.getFullYear() - 1;
  const fiscalStart = new Date(isForm ? year - 5 : year - 5, 3, 1); // April 1
  const fiscalEnd = new Date(year + 1, 2, 31); // March 31
  return { fiscalStart, fiscalEnd };
}

// Filter headers to only include dates within the fiscal year
export function filterHeadersByFiscalYear(headers: string[], fiscalStart: Date, fiscalEnd: Date) {
  return headers.filter(header => {
    const match = header.match(/: (\d{2}-[A-Za-z]{3}-\d{4}) \((\d{4})\)/);
    if (!match) return false;
    const parsedDate = parse(match[1], 'dd-MMM-yyyy', new Date());
    return isWithinInterval(parsedDate, { start: fiscalStart, end: fiscalEnd });
  });
}

export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

/**
 * Get week number based on standard calendar
 */
export function getWeekNumber(date: Date): number {
  const firstDayOfYear = new Date(date.getFullYear(), 0, 1);
  const pastDaysOfYear = (date.getTime() - firstDayOfYear.getTime()) / 86400000;
  return Math.ceil((pastDaysOfYear + firstDayOfYear.getDay() + 1) / 7);
}

/**
 * Get fiscal week number
 * First week starts from last week of January or first week of February (for leap years)
 */
export function getFiscalWeekNumber(date: Date): number {
  const year = date.getFullYear();
  const isLeapYear = (year % 4 === 0 && year % 100 !== 0) || (year % 400 === 0);
  
  // Define fiscal year start date
  // For leap years: First week of February (around Feb 1)
  // For regular years: Last week of January (around Jan 25)
  const fiscalYearStartMonth = isLeapYear ? 1 : 0; // 0 = January, 1 = February
  const fiscalYearStartDay = isLeapYear ? 1 : 25;
  
  const fiscalYearStart = new Date(year, fiscalYearStartMonth, fiscalYearStartDay);
  
  // If the date is before fiscal year start, it belongs to previous fiscal year
  if (date < fiscalYearStart) {
    return getFiscalWeekNumber(new Date(year - 1, 11, 31)); // Dec 31 of previous year
  }
  
  // Calculate days since fiscal year start
  const daysSinceFiscalYearStart = Math.floor((date.getTime() - fiscalYearStart.getTime()) / 86400000);
  
  // Calculate fiscal week
  return Math.floor(daysSinceFiscalYearStart / 7) + 1;
}

export function formatWeekRange(startWeek: number, endWeek: number, year?: number): string {
  const currentYear = year || new Date().getFullYear();
  return `Week ${startWeek.toString().padStart(2, '0')} – Week ${endWeek.toString().padStart(2, '0')}`;
}
