import React from "react";
import { cn } from "@/lib/utils";
import { TableRow, TableCell } from "@/components/ui/table";
import MetricCellContent from "./MetricCellContent";
import { CapacityDataRow, MetricDefinition, TimeInterval } from "./types";
import { Edit3 } from "lucide-react";

interface MetricRowProps {
  item: CapacityDataRow;
  metricDef: MetricDefinition;
  level: number;
  periodHeaders: string[];
  onTeamMetricChange: (lobId: string, teamName: string, periodHeader: string, metricKey: string, newValue: string) => void;
  onLobMetricChange: (lobId: string, periodHeader: string, metricKey: string, newValue: string) => void;
  editingCell: { id: string; period: string; metricKey: string } | null;
  onSetEditingCell: (id: string | null, period: string | null, metricKey: string | null) => void;
  selectedTimeInterval: TimeInterval;
}

const MetricRow: React.FC<MetricRowProps> = React.memo(({ item, metricDef, level, periodHeaders, onTeamMetricChange, onLobMetricChange, editingCell, onSetEditingCell, selectedTimeInterval }) => {
  return (
    <TableRow className="hover:bg-muted/50">
      <TableCell
        className="sticky left-0 z-20 bg-card font-normal text-foreground whitespace-nowrap py-2 pr-4 w-[300px]"
      >
        <div
          style={{ paddingLeft: `${level * 1.5 + 0.5}rem` }}
          className="flex items-center gap-2 cursor-default w-full max-w-full overflow-hidden"
          title={metricDef.label}
        >
          <span>{metricDef.label}</span>
          {item.itemType === 'Team' && metricDef.isEditableForTeam && !metricDef.isDisplayOnly &&
            (metricDef.category === 'Assumption' || metricDef.category === 'PrimaryHC' || metricDef.category === 'HCAdjustment') &&
            <Edit3 className="h-3 w-3 text-muted-foreground opacity-50" />}
        </div>
      </TableCell>
      {periodHeaders.map((periodHeader) => {
        const metricForPeriod = item.periodicData[periodHeader];
        let cellTextColor = "text-foreground";
        if (metricDef.key === "overUnderHC" && metricForPeriod && (metricForPeriod as any)[metricDef.key] !== null && (metricForPeriod as any)[metricDef.key] !== undefined) {
          const value = Number((metricForPeriod as any)[metricDef.key]);
          if (value < -0.001) cellTextColor = "text-destructive";
          else if (value > 0.001) cellTextColor = "text-primary";
        }

        const currentEditId = item.itemType === 'Team' && item.lobId ? `${item.lobId}_${item.name.replace(/\s+/g, '-')}` : item.id;
        const isCurrentlyEditing =
          editingCell?.id === currentEditId &&
          editingCell?.period === periodHeader &&
          editingCell?.metricKey === metricDef.key;

        return (
          <TableCell
            key={`${item.id}-${metricDef.key}-${periodHeader}`}
            className={`text-right tabular-nums ${cellTextColor} py-2 px-2 min-w-[100px] border-l border-border/50`}
          >
            <MetricCellContent
              item={item}
              metricData={metricForPeriod}
              metricDef={metricDef}
              periodName={periodHeader}
              onTeamMetricChange={onTeamMetricChange}
              onLobMetricChange={onLobMetricChange}
              isEditing={isCurrentlyEditing}
              onSetEditingCell={onSetEditingCell}
              selectedTimeInterval={selectedTimeInterval}
            />
          </TableCell>
        );
      })}
    </TableRow>
  );
});

MetricRow.displayName = 'MetricRow';
export default MetricRow;
