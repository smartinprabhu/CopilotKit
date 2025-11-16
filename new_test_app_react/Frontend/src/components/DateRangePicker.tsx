import React, { useState, useEffect, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Calendar as CalendarIcon, ChevronLeft, ChevronRight } from "lucide-react";
import { cn, getFiscalYearRange, filterHeadersByFiscalYear } from "@/lib/utils";
import { DateRange } from "react-day-picker";
import {
  getWeek,
  startOfWeek,
  endOfWeek,
  isBefore,
  isAfter,
  subYears,
  addYears,
  subMonths,
  addMonths,
  isSameDay,
  addMonths as addM,
  startOfMonth,
  endOfMonth,
  startOfQuarter,
  endOfQuarter,
  startOfYear,
  endOfYear,
} from "date-fns";

import { ALL_WEEKS_HEADERS, findFiscalWeekHeaderForDate } from "./types";

function isSameRange(a?: DateRange, b?: DateRange) {
  if (!a || !b) return false;
  return isSameDay(a.from!, b.from!) && isSameDay(a.to!, b.to!);
}

// Custom caption for month/year navigation
const CustomCaption = ({ displayMonth, onMonthChange }) => {
  const monthName = displayMonth.toLocaleString("default", { month: "long" });
  const year = displayMonth.getFullYear();

  return (
    <div className="flex items-center justify-between px-2 py-1">
      <div className="flex items-center gap-1">
        <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => onMonthChange(subMonths(displayMonth, 1))}>
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <span className="text-sm font-medium">{monthName}</span>
        <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => onMonthChange(addMonths(displayMonth, 1))}>
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
      <div className="flex items-center gap-1">
        <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => onMonthChange(subYears(displayMonth, 1))}>
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <span className="text-sm font-medium">{year}</span>
        <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => onMonthChange(addYears(displayMonth, 1))}>
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};

export default function DateRangePicker({
  date,
  onDateChange,
  onDateDefaultChange,
  className,
  maxRangeToDate,
  weekStartsOn = 1 as 0 | 1 | 2 | 3 | 4 | 5 | 6,
  disableFuture = false,
  isForm = false,
  comparePicker = false,
  defaultPreset = '',
}: {
  date: DateRange | undefined;
  onDateChange: (range: DateRange | undefined) => void;
  onDateDefaultChange: () => void;
  className?: string;
  weekStartsOn?: 0 | 1 | 2 | 3 | 4 | 5 | 6;
  disableFuture?: boolean;
  maxRangeToDate?: string;
  isForm?: boolean;
  comparePicker?: boolean;
  defaultPreset?: string;
}) {
  const [clientButtonText, setClientButtonText] = useState("Loading...");
  const [displayMonths, setDisplayMonths] = useState<Date[]>([]);
  const [tempRange, setTempRange] = useState<DateRange | undefined>(date);

  const getPresetRanges = (weekStartsOn: 0 | 1 | 2 | 3 | 4 | 5 | 6, disableFuture: boolean) => {
    const today = new Date();
    const startWeek = startOfWeek(today, { weekStartsOn });
    const endWeek = endOfWeek(today, { weekStartsOn });

    const startLastWeek = startOfWeek(subMonths(today, 1), { weekStartsOn });
    const endLastWeek = endOfWeek(subMonths(today, 1), { weekStartsOn });

    const startMonth = startOfMonth(today);
    const endMonth = endOfMonth(today);

    const startLastMonth = startOfMonth(subMonths(today, 1));
    const endLastMonth = endOfMonth(subMonths(today, 1));

    const startQuarter = startOfQuarter(today);
    const endQuarter = endOfQuarter(today);

    const startLastQuarter = startOfQuarter(subMonths(today, 3));
    const endLastQuarter = endOfQuarter(subMonths(today, 3));

    const startYear = startOfYear(today);
    const endYear = endOfYear(today);

    return [
      { label: "This Week", description: "Current week", range: { from: startWeek, to: endWeek } },
      { label: "Last Week", description: "Previous week", range: { from: startOfWeek(subMonths(today, 1), { weekStartsOn }), to: endOfWeek(subMonths(today, 1), { weekStartsOn }) } },
      { label: "This Month", description: "Current month", range: { from: startMonth, to: endMonth } },
      { label: "Last Month", description: "Previous month", range: { from: startLastMonth, to: endLastMonth } },
      { label: "This Quarter", description: "Current quarter", range: { from: startQuarter, to: endQuarter } },
      { label: "Last Quarter", description: "Previous quarter", range: { from: startLastQuarter, to: endLastQuarter } },
      { label: "This Year", description: "Current year", range: { from: startYear, to: endYear } },
      { label: "Last Year", description: "Previous year", range: { from: startOfYear(subYears(today, 1)), to: endOfYear(subYears(today, 1)) } },
      { label: "YTD", description: "Jan 1 to today", range: { from: new Date(today.getFullYear(), 0, 1), to: today } },
      { label: "QTD", description: "Start of quarter to today", range: { from: startQuarter, to: today } },
    ];
  };


  const { fiscalStart, fiscalEnd } = useMemo(() => getFiscalYearRange(date?.from || new Date(), isForm), [date?.from]);
  const filteredWeekHeaders = useMemo(() => filterHeadersByFiscalYear(ALL_WEEKS_HEADERS, fiscalStart, fiscalEnd), [fiscalStart, fiscalEnd]);

  useEffect(() => {
    let newButtonText = "Pick a date range";

    if (typeof window !== 'undefined' && date?.from) {
      const fromDateObj = new Date(date.from);

      // ✅ Align 'from' to start of week (e.g., Monday)
      const fromWeekStart = startOfWeek(fromDateObj, { weekStartsOn });

      const fromFiscalWeekInfo = findFiscalWeekHeaderForDate(fromWeekStart, filteredWeekHeaders);
      const fromFiscalWeekLabel = fromFiscalWeekInfo
        ? fromFiscalWeekInfo.split(':')[0].replace("FWk", "W")
        : `W${getWeek(fromWeekStart, { weekStartsOn })}`;

      // ✅ Format local start of week date
      const formattedFromDate = `${String(fromWeekStart.getDate()).padStart(2, '0')}/${String(fromWeekStart.getMonth() + 1).padStart(2, '0')}/${fromWeekStart.getFullYear()}`;

      newButtonText = `${fromFiscalWeekLabel} (${formattedFromDate})`;

      if (date.to) {
        const toDateObj = new Date(date.to); // Don't align this!

        const toFiscalWeekInfo = findFiscalWeekHeaderForDate(toDateObj, filteredWeekHeaders);
        const toFiscalWeekLabel = toFiscalWeekInfo
          ? toFiscalWeekInfo.split(':')[0].replace("FWk", "W")
          : `W${getWeek(toDateObj, { weekStartsOn })}`;

        // ✅ Format local raw selected end date
        const formattedToDate = `${String(toDateObj.getDate()).padStart(2, '0')}/${String(toDateObj.getMonth() + 1).padStart(2, '0')}/${toDateObj.getFullYear()}`;

        const fromWeekStartOnly = startOfWeek(fromDateObj, { weekStartsOn });

        if (!isSameDay(fromWeekStartOnly, startOfWeek(toDateObj, { weekStartsOn }))) {
          newButtonText += ` - ${toFiscalWeekLabel} (${formattedToDate})`;
        }
      }
    }

    setClientButtonText(newButtonText);
  }, [date, filteredWeekHeaders, weekStartsOn]);

  const defaultCalendarMonth = useMemo(
    () => date?.from || new Date(new Date().getFullYear(), new Date().getMonth(), 1),
    [date?.from]
  );

  const defaultCalendarToMonth = useMemo(
    () => date?.to || new Date(new Date().getFullYear(), new Date().getMonth(), 1),
    [date?.to]
  );

  useEffect(() => {
  if (date) {
    setTempRange(date);
  }
}, [date]);

  useEffect(() => {
    const firstMonth = defaultCalendarMonth;
    const secondMonth = defaultCalendarToMonth // addM(firstMonth, 1);
    setDisplayMonths([firstMonth, secondMonth]);
  }, [defaultCalendarMonth, defaultCalendarToMonth]);

  const presets = useMemo(
    () => getPresetRanges(weekStartsOn, disableFuture),
    [weekStartsOn, disableFuture]
  );

  // ✅ Apply default preset on mount
  useEffect(() => {
    if (defaultPreset) {
      const foundPreset = presets.find(
        (p) => p.label.toLowerCase() === defaultPreset.toLowerCase()
      );
      if (foundPreset) {
        setTempRange(foundPreset.range);
       // onDateChange(foundPreset.range);

        // Move calendar to preset’s month
        const fromMonth = foundPreset.range.from!;
        const toMonth = addM(fromMonth, 1);
        setDisplayMonths([fromMonth, toMonth]);
      }
    }
  }, [defaultPreset, presets]);

  const handleMonthChange = (index: number) => (newMonth: Date) => {
    if (disableFuture) {
      const now = new Date();
      if (index === 0 && newMonth > now) return;
      if (index === 1 && newMonth > now) return;
    }

    const newDisplayMonths = [...displayMonths];
    newDisplayMonths[index] = newMonth;

    if (index === 0 && newDisplayMonths[1] <= newMonth) {
      newDisplayMonths[1] = addMonths(newMonth, 1);
    } else if (index === 1 && newDisplayMonths[0] >= newMonth) {
      newDisplayMonths[0] = subMonths(newMonth, 1);
    }

    setDisplayMonths(newDisplayMonths);
  };

  const [open, setOpen] = useState(false);


  return (
    <div className={cn("grid gap-2", className)}>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            id="date"
            variant={"outline"}
            className={cn("w-full lg:w-[315px] justify-start text-left font-normal h-10", !date && "text-muted-foreground")}
          >
            <CalendarIcon className="mr-2 h-4 w-4" />
            <span>{clientButtonText}</span>
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0 z-[9999]" style={isForm ? { marginTop: '-150px' } : {}} align="start" side="bottom">
          <div className="flex max-w-full">
            {/* Left Sidebar */}
            {!isForm && !comparePicker && (
              <div className="w-[230px] border-r p-3 overflow-y-auto h-[340px]">
                {getPresetRanges(weekStartsOn, disableFuture).map((preset) => {
                  const isActive = isSameRange(tempRange, preset.range);
                  return (
                    <div
                      key={preset.label}
                      className={cn(
                        "cursor-pointer rounded-md p-2 mb-1 transition-colors",
                        isActive ? "bg-primary text-white" : "hover:bg-accent"
                      )}
                      onClick={() => {
                        setTempRange(preset.range);
                        // Move calendar to preset's from month
                        const fromMonth = preset.range.from!;
                        const toMonth = addM(fromMonth, 1);
                        setDisplayMonths([fromMonth, toMonth]);
                      }}
                    >
                      <p className="text-sm">{preset.label}</p>
                    </div>
                  );
                })}
              </div>
            )}
            {/* Right Side Calendar */}
            <div className="flex">
              {displayMonths.map((month, index) => (
                <Calendar
                  key={index}
                  initialFocus={index === 0}
                  mode="range"
                  weekStartsOn={weekStartsOn}
                  captionLayout="dropdown-buttons"
                  fromYear={fiscalStart.getFullYear()}
                  toYear={fiscalEnd.getFullYear()}
                  toDate={disableFuture ? new Date() : undefined}
                  month={month}
                  onMonthChange={handleMonthChange(index)}
                  defaultMonth={defaultCalendarMonth}
                  selected={tempRange}
                  components={{
                    Caption: (props) => (
                      <CustomCaption
                        displayMonth={props.displayMonth}
                        onMonthChange={handleMonthChange(index)}
                      />
                    ),
                  }}
                  onSelect={(range: DateRange | undefined) => {
                    let newFrom = range?.from;
                    let newTo = range?.to;

                    if (newFrom) newFrom = startOfWeek(newFrom, { weekStartsOn });
                    if (newTo) newTo = endOfWeek(newTo, { weekStartsOn });

                    const today = new Date();
                    if (disableFuture) {
                      if (newFrom && newFrom > today) newFrom = today;
                      if (newTo && newTo > today) newTo = today;
                    }

                    if (newTo && maxRangeToDate && isAfter(newTo, maxRangeToDate)) {
                      newTo = maxRangeToDate;
                    }

                    if (newFrom && newTo && isBefore(newTo, newFrom)) {
                      newTo = endOfWeek(newFrom, { weekStartsOn });
                    }

                    setTempRange(
                      newFrom
                        ? { from: newFrom, to: newTo || endOfWeek(newFrom, { weekStartsOn }) }
                        : undefined
                    );
                  }}
                  numberOfMonths={1}
                />
              ))}
            </div>
          </div>

          {/* Footer Buttons */}
          <div className="flex justify-end gap-2 p-3 border-t">
            <Button variant="ghost" size="sm" onClick={() => {
              setTempRange(undefined);
              onDateDefaultChange();
            }}>
              Reset
            </Button>
            <Button
              size="sm"
              onClick={() => {
                if (tempRange) {
                  onDateChange(tempRange);
                  setOpen(false);
                }
              }}
              disabled={!tempRange?.from}
            >
              Apply
            </Button>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}
