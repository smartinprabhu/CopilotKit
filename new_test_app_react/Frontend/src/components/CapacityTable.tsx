import React, { useCallback, useEffect, useRef } from "react";
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import MetricRow from "./MetricRow";
import { CapacityDataRow, TimeInterval, TeamMetricDefinitions, AggregatedMetricDefinitions } from "./types";

interface CapacityTableProps {
  data: CapacityDataRow[];
  periodHeaders: string[];
  expandedItems: Record<string, boolean>;
  toggleExpand: (id: string) => void;
  teamMetricDefinitions: TeamMetricDefinitions;
  aggregatedMetricDefinitions: AggregatedMetricDefinitions;
  onTeamMetricChange: (lobId: string, teamName: string, periodHeader: string, metricKey: string, newValue: string) => void;
  onLobMetricChange: (lobId: string, periodHeader: string, metricKey: string, newValue: string) => void;
  editingCell: { id: string; period: string; metricKey: string } | null;
  onSetEditingCell: (id: string | null, period: string | null, metricKey: string | null) => void;
  selectedTimeInterval: TimeInterval;
  onActiveHierarchyChange: (newContext: string | null) => void;
  tableBodyScrollRef: React.RefObject<HTMLDivElement>;
}

const CapacityTable: React.FC<CapacityTableProps> = ({
  data,
  periodHeaders,
  expandedItems,
  toggleExpand,
  teamMetricDefinitions,
  aggregatedMetricDefinitions,
  onTeamMetricChange,
  onLobMetricChange,
  editingCell,
  onSetEditingCell,
  selectedTimeInterval,
  onActiveHierarchyChange,
  tableBodyScrollRef,
}) => {
  const itemNameRowRefs = useRef<Map<string, HTMLTableRowElement | null>>(new Map());

  useEffect(() => {
    const observerCallback: IntersectionObserverCallback = (entries) => {
      const visibleEntries = entries.filter(entry => entry.isIntersecting);
      if (visibleEntries.length === 0) {
        onActiveHierarchyChange(null);
        return;
      }

      let topMostVisibleEntry: IntersectionObserverEntry | null = null;
      let minTopValue = Infinity;

      visibleEntries.forEach(entry => {
        const top = entry.boundingClientRect.top;
        if (top < minTopValue) {
          minTopValue = top;
          topMostVisibleEntry = entry;
        }
      });

      if (topMostVisibleEntry) {
        const targetElement = topMostVisibleEntry.target as HTMLTableRowElement;
        const itemName = targetElement.dataset.itemName || "Unknown";
        const itemType = targetElement.dataset.itemType || "";
        let contextString = "";
        if (itemType === "BU") contextString = `BU: ${itemName}`;
        else if (itemType === "LOB") contextString = `LOB: ${itemName}`;
        else if (itemType === "Team") contextString = `Team: ${itemName}`;
        else contextString = itemName;
        onActiveHierarchyChange(contextString);
      } else {
        onActiveHierarchyChange(null);
      }
    };

    let observer: IntersectionObserver | null = null;
    const currentScrollContainer = tableBodyScrollRef.current;

    if (currentScrollContainer) {
      const options: IntersectionObserverInit = {
        root: currentScrollContainer,
        rootMargin: `5px 0px -90% 0px`,
        threshold: 0.01,
      };
      observer = new IntersectionObserver(observerCallback, options);
      itemNameRowRefs.current.forEach(rowElement => {
        if (rowElement) observer?.observe(rowElement);
      });
    }
    return () => {
      observer?.disconnect();
      itemNameRowRefs.current.forEach(rowElement => {
        if (rowElement) observer?.unobserve(rowElement);
      });
    };
  }, [periodHeaders, data, onActiveHierarchyChange, tableBodyScrollRef]);

  const renderTeamMetrics = useCallback((item: CapacityDataRow, category: string, baseLevel: number) => {
    return teamMetricDefinitions
      .filter(def => def.category === category)
      .map(metricDef => (
        <MetricRow
          key={`${item.id}-${metricDef.key}`}
          item={item}
          metricDef={metricDef}
          level={baseLevel}
          periodHeaders={periodHeaders}
          onTeamMetricChange={onTeamMetricChange}
          onLobMetricChange={onLobMetricChange}
          editingCell={editingCell}
          onSetEditingCell={onSetEditingCell}
          selectedTimeInterval={selectedTimeInterval}
        />
      ));
  }, [periodHeaders, onTeamMetricChange, onLobMetricChange, editingCell, onSetEditingCell, selectedTimeInterval]);

  const renderCapacityItemContent = useCallback((item: CapacityDataRow): React.ReactNode[] => {
    const rows: React.ReactNode[] = [];

    if (item.itemType === 'Team') {
      rows.push(...renderTeamMetrics(item, 'PrimaryHC', item.level + 1));

      const assumptionsKey = `${item.id}_Assumptions`;
      const areAssumptionsExpanded = expandedItems[assumptionsKey] || false;
      rows.push(
        <TableRow key={assumptionsKey + "-header"} className="hover:bg-muted/60">
          <TableCell
            className="sticky left-0 z-20 bg-card font-semibold text-foreground whitespace-nowrap py-2 cursor-pointer"
            style={{ paddingLeft: `${(item.level + 1) * 1.5 + 0.5}rem`, paddingRight: '1rem' }}
            onClick={() => toggleExpand(assumptionsKey)}
          >
            <div className="flex items-center gap-2">
              <ChevronDown className={`h-4 w-4 shrink-0 transition-transform duration-200 ${areAssumptionsExpanded ? "rotate-180" : ""}`} />
              Assumptions
            </div>
          </TableCell>
          {periodHeaders.map(ph => <TableCell key={`${assumptionsKey}-${ph}-placeholder`} className="py-2 border-l border-border/50"></TableCell>)}
        </TableRow>
      );
      if (areAssumptionsExpanded) {
        rows.push(...renderTeamMetrics(item, 'Assumption', item.level + 2));
      }

      const hcAdjustmentsKey = `${item.id}_HCAdjustments`;
      const areHcAdjustmentsExpanded = expandedItems[hcAdjustmentsKey] || false;
      rows.push(
        <TableRow key={hcAdjustmentsKey + "-header"} className="hover:bg-muted/60">
          <TableCell
            className="sticky left-0 z-20 bg-card font-semibold text-foreground whitespace-nowrap py-2 cursor-pointer"
            style={{ paddingLeft: `${(item.level + 1) * 1.5 + 0.5}rem`, paddingRight: '1rem' }}
            onClick={() => toggleExpand(hcAdjustmentsKey)}
          >
            <div className="flex items-center gap-2">
              <ChevronDown className={`h-4 w-4 shrink-0 transition-transform duration-200 ${areHcAdjustmentsExpanded ? "rotate-180" : ""}`} />
              HC Adjustments
            </div>
          </TableCell>
          {periodHeaders.map(ph => <TableCell key={`${hcAdjustmentsKey}-${ph}-placeholder`} className="py-2 border-l border-border/50"></TableCell>)}
        </TableRow>
      );
      if (areHcAdjustmentsExpanded) {
        rows.push(...renderTeamMetrics(item, 'HCAdjustment', item.level + 2));
      }
    } else {
      aggregatedMetricDefinitions.forEach(metricDef => {
        if (item.itemType === 'BU' && metricDef.key === 'lobTotalBaseRequiredMinutes') {
          return;
        }
        if (item.itemType === 'LOB' && metricDef.key === 'lobTotalBaseRequiredMinutes') {
          return;
        }
        rows.push(
          <MetricRow
            key={`${item.id}-${metricDef.key}`}
            item={item}
            metricDef={metricDef}
            level={item.level + 1}
            periodHeaders={periodHeaders}
            onTeamMetricChange={onTeamMetricChange}
            onLobMetricChange={onLobMetricChange}
            editingCell={editingCell}
            onSetEditingCell={onSetEditingCell}
            selectedTimeInterval={selectedTimeInterval}
          />
        );
      });
    }
    return rows;
  }, [periodHeaders, expandedItems, toggleExpand, onTeamMetricChange, onLobMetricChange, editingCell, onSetEditingCell, selectedTimeInterval, renderTeamMetrics]);

  const renderTableItem = useCallback((item: CapacityDataRow): React.ReactNode[] => {
    const rows: React.ReactNode[] = [];
    const isExpanded = expandedItems[item.id] || false;

    let isItemExpandable = (item.children && item.children.length > 0) || item.itemType === 'Team';

    let rowSpecificBgClass = '';
    let buttonTextClass = 'text-foreground';
    let itemZIndex = 20;

    if (item.itemType === 'BU') {
      rowSpecificBgClass = 'bg-secondary';
      buttonTextClass = 'text-secondary-foreground';
      itemZIndex = 35;
    } else if (item.itemType === 'LOB') {
      rowSpecificBgClass = 'bg-muted';
      buttonTextClass = 'text-muted-foreground';
      itemZIndex = 30;
    } else if (item.itemType === 'Team') {
      rowSpecificBgClass = 'bg-muted/50';
      buttonTextClass = 'text-foreground';
      itemZIndex = 25;
    }

    const hoverClass = item.itemType !== 'BU' ? 'hover:bg-muted/70' : 'hover:bg-secondary/80';

    rows.push(
      <TableRow
        key={`${item.id}-name`}
        className={cn(rowSpecificBgClass, hoverClass)}
        ref={el => { if (el) itemNameRowRefs.current.set(item.id, el); else itemNameRowRefs.current.delete(item.id); }}
        data-item-id={item.id}
        data-item-name={item.name}
        data-item-type={item.itemType}
        data-item-level={item.level}
      >
        <TableCell
          className={cn(
            "p-0 sticky left-0 whitespace-nowrap",
            rowSpecificBgClass || 'bg-card'
          )}
          style={{
            zIndex: itemZIndex,
            width: '300px',
            minWidth: '300px',
            paddingLeft: `${Math.min(item.level, 5) * 1.5 + 0.5}rem`,
            paddingRight: '1rem'
          }}
        >
          <button
            onClick={isItemExpandable ? () => toggleExpand(item.id) : undefined}
            disabled={!isItemExpandable}
            className={cn(
              "py-3 px-2 font-semibold hover:no-underline w-full text-left flex items-center gap-2",
              buttonTextClass
            )}
            aria-expanded={isItemExpandable ? isExpanded : undefined}
          >
            {isItemExpandable && (
              <ChevronDown className={`h-4 w-4 shrink-0 transition-transform duration-200 ${isExpanded ? "rotate-180" : ""}`} />
            )}
            {item.name}
          </button>
        </TableCell>
        {periodHeaders.map((ph, idx) => (
          <TableCell
            key={`${item.id}-${ph}-nameplaceholder`}
            className={cn(rowSpecificBgClass, 'py-3 min-w-[100px] border-l border-border/50')}
          ></TableCell>
        ))}
      </TableRow>
    );

    if (isExpanded) {
      if (item.itemType === 'BU' || item.itemType === 'LOB') {
        const aggregatedMetricRows = renderCapacityItemContent(item);
        rows.push(...aggregatedMetricRows);
        if (item.children && item.children.length > 0) {
          item.children.forEach(child => {
            rows.push(...renderTableItem(child));
          });
        }
      } else if (item.itemType === 'Team') {
        const teamMetricStructure = renderCapacityItemContent(item);
        rows.push(...teamMetricStructure);
      }
    } else if (!isItemExpandable && (item.itemType === 'BU' || item.itemType === 'LOB')) {
      const itemMetricRows = renderCapacityItemContent(item);
      rows.push(...itemMetricRows);
    }

    return rows;
  }, [expandedItems, periodHeaders, toggleExpand, renderCapacityItemContent]);

  return (
    <div ref={tableBodyScrollRef} className="overflow-x-auto relative">
      <Table className="min-w-full">
        <TableBody>
          {data.length > 0 ? (
            data.flatMap(item => renderTableItem(item))
          ) : (
            <TableRow>
              <TableCell colSpan={periodHeaders.length + 1} className="text-center text-muted-foreground h-24">
                No data available for the current selection.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
};

export default React.memo(CapacityTable);
