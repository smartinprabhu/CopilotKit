import { TrendingDown, TrendingUp } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip as UITooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { formatWithPrecision, formatNumber } from "@/lib/utils";

interface KPIMetricProps {
  title?: string;
  value?: string | number;
  subtitle?: string;
  kpiDirection?: string;
  changeValue?: number;
  selectedYear?: number;
  oldValue?: number;
  changeText?: string;
  invertChange?: boolean;
  loading?: boolean;
  compareOption?: string;
  compareEnabled?: boolean;
  defaultDecimalPrecisions?: number;
}

export const KPIMetricsCard = ({
  title,
  value,
  subtitle,
  changeValue,
  changeText,
  oldValue,
  compareOption,
  compareEnabled,
  selectedYear,
  kpiDirection,
  defaultDecimalPrecisions,
  invertChange = false,
  loading = false,
}: KPIMetricProps) => {
  const isPositive = kpiDirection === "Low is Good"
  ? changeValue <= 0
  : changeValue >= 0;

  const Icon = () => {
  if (isPositive) {
    return kpiDirection === "Low is Good"
      ? <TrendingDown className="w-4 h-4 text-green-400" />
      : <TrendingUp className="w-4 h-4 text-green-400" />;
  } else {
    return kpiDirection === "Low is Good"
      ? <TrendingUp className="w-4 h-4 text-red-400" />
      : <TrendingDown className="w-4 h-4 text-red-400" />;
  }
};

const compareValueMap = {
  previous: "Previous Period",
  sameYear: "Same Period Previous Year",
  selectYear: "Same Period: Select Year",
  custom: "Custom",
};
  
  if (loading) {
    return (
      <Card className="relative m-4 shadow-md w-full max-w-xs h-40">
        <Skeleton className="h-4 w-3/4 mb-2" />
        <Skeleton className="h-8 w-1/2 mb-4" />
        <Skeleton className="h-4 w-full mb-4" />
        <div className="flex items-center gap-1">
          <Skeleton className="h-4 w-4" />
          <Skeleton className="h-4 w-20" />
        </div>
      </Card>
    );
  }

  return (
    <TooltipProvider>
      <UITooltip>
        <TooltipTrigger asChild>
          <Card className="shadow-md p-4 w-full max-w-xs h-auto executive-card hover:bg-accent transition duration-200 ease-in-out overflow-hidden">
            <div className="space-y-1.5">
              <h3 className="font-medium text-foreground truncate">{title}</h3>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-bold text-foreground truncate">
                  {typeof value === 'number' ? formatNumber(value, defaultDecimalPrecisions) : value}
                </span>
              </div>
              <p className="text-sm text-muted-foreground truncate">{subtitle}</p>
            </div>
            {changeText && (
            <div className="mt-4 flex flex-col space-y-1">
              <div className="flex items-center gap-1">
                <Icon />
                <span className={`text-sm font-medium ${isPositive ? 'text-green-400' : 'text-red-400'} truncate`}>
                  {changeValue >= 0 ? '+' : '-'} {formatWithPrecision(Math.abs(changeValue), defaultDecimalPrecisions)}% {changeText}
                </span>
              </div>
            </div>
            )}
          </Card>
        </TooltipTrigger>
     <TooltipContent
        side="bottom"
        className="bg-white dark:bg-gray-800 text-sm text-gray-800 dark:text-gray-200 rounded-md shadow-lg p-4 w-64 text-center"
      >
        <div className="space-y-4">
          <div>
            <div className="font-semibold">Value</div>
            <div>{value}</div>
          </div>
          <div>
            <div className="font-semibold">Comparison Value</div>
            <div>{oldValue}</div>
          </div>
          {compareEnabled && (
            <div>
              <div className="font-semibold">Compare Period</div>
              <div>
                 {compareOption && compareValueMap[compareOption] ? compareValueMap[compareOption] : 'N/A'}
                 {`${compareOption === 'selectYear' ? `(${selectedYear.toString()})` : ''}`}
              </div>
            </div>
          )}
        </div>
      </TooltipContent>
    </UITooltip>
   </TooltipProvider>
  );
};
