import { startOfWeek, endOfWeek, subWeeks, subYears, setYear } from 'date-fns';

export const getComparisonRange = (
  selectedRange: { from: Date; to: Date },
  compareOption: 'previous' | 'sameYear' | 'selectYear' | 'custom',
  selectedYear: number,
  customRange?: { from: Date; to: Date },
  weekStartsOn: 0 | 1 | 2 | 3 | 4 | 5 | 6 = 1 // default Monday
): { from: Date; to: Date } | null => {
  if (!selectedRange?.from || !selectedRange?.to) return null;

  const start = startOfWeek(selectedRange.from, { weekStartsOn });
  const end = endOfWeek(selectedRange.to, { weekStartsOn });

  const weekCount =
    Math.ceil((end.getTime() - start.getTime()) / (7 * 24 * 60 * 60 * 1000));

  switch (compareOption) {
    case 'previous': {
      const prevEnd = startOfWeek(subWeeks(start, 1), { weekStartsOn });
      const prevStart = subWeeks(prevEnd, weekCount - 1);
      return { from: prevStart, to: endOfWeek(prevEnd, { weekStartsOn }) };
    }

    case 'sameYear': {
      const sameStart = startOfWeek(subYears(start, 1), { weekStartsOn });
      const sameEnd = endOfWeek(subYears(end, 1), { weekStartsOn });
      return { from: sameStart, to: sameEnd };
    }

    case 'selectYear': {
      const from = setYear(start, selectedYear);
      const to = setYear(end, selectedYear);
      return { from, to };
    }

    case 'custom':
      return customRange || null;

    default:
      return null;
  }
};
