import React, { useState, useCallback, useMemo } from 'react';
import { AgGridReact } from 'ag-grid-react';
import { 
  ColDef, 
  ICellRendererParams,
  CellClassParams,
  GridReadyEvent
} from 'ag-grid-community';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Info } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

// Import AG Grid styles
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import '../styles/ag-grid-custom.css';  // Import custom AG Grid styles

// Types
interface MetricExplanation {
  title: string;
  description: string;
  impact: string;
}

interface WeekData {
  date: string;
  volume: number;
  aht: number;
  shrinkage: number;
  occupancy: number;
  attrition: number;
  actual: number;
}

interface PlanningGridProps {
  weekData: WeekData[];
  metricExplanations: Record<string, MetricExplanation>;
  onDataChange: (index: number, field: keyof WeekData, value: string) => void;
}

// Custom cell renderers
const InputCellRenderer = (props: ICellRendererParams) => {
  const { value, node, colDef, data, rowIndex } = props;
  const field = colDef?.field as keyof WeekData;
  const updateField = props.context.updateData;
  
  // Format percentage values for display
  let displayValue = value;
  if (field === 'shrinkage' || field === 'occupancy' || field === 'attrition') {
    displayValue = (value * 100).toFixed(0);
  }

  return (
    <Input 
      type="number"
      value={displayValue}
      onChange={(e) => updateField(rowIndex, field, e.target.value)}
      className="w-24 text-center mx-auto h-8"
    />
  );
};

const OverUnderCellRenderer = (props: ICellRendererParams) => {
  const { value } = props;
  const className = value < 0 
    ? 'text-red-600 font-medium' 
    : value > 0 
    ? 'text-green-600 font-medium' 
    : '';
    
  return <span className={className}>{value}</span>;
};

const LabelCellRenderer = (props: ICellRendererParams) => {
  const { value, context } = props;
  const metricKey = props.data.metricKey;
  const metricInfo = context.metricExplanations[metricKey];
  
  if (!metricInfo) return <span>{value}</span>;
  
  return (
    <div className="flex items-center gap-1">
      <span className="font-medium">{value}</span>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="icon" className="h-6 w-6 p-0">
              <Info className="h-4 w-4 text-muted-foreground" />
              <span className="sr-only">Info about {value}</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent className="max-w-sm bg-background border shadow-lg">
            <div className="space-y-2 p-1">
              <h4 className="font-medium">{metricInfo.title}</h4>
              <p className="text-sm text-muted-foreground">{metricInfo.description}</p>
              <p className="text-sm font-medium">Impact: {metricInfo.impact}</p>
            </div>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  );
};

// Header Custom Component
const SectionHeaderRenderer = (props: ICellRendererParams) => {
  return (
    <div className="font-semibold text-base px-2 py-1">{props.value}</div>
  );
};

const PlanningGrid: React.FC<PlanningGridProps> = ({ 
  weekData, 
  metricExplanations,
  onDataChange
}) => {
  const [gridApi, setGridApi] = useState<any>(null);
  
  // Calculate Required HC based on formula
  const calculateRequired = (data: WeekData) => {
    // Total Handling Time = Volume × AHT (in seconds)
    const totalHandlingTime = data.volume * data.aht;
    
    // Effective Minutes per Agent per Week = 480 × Occupancy × (1 − Shrinkage) × (1 − Attrition)
    // 480 minutes = 8 hours per day * 60 minutes * 5 days per week
    const effectiveMinutesPerAgent = 480 * 5 * data.occupancy * (1 - data.shrinkage) * (1 - data.attrition);
    
    // Convert total handling time from seconds to minutes
    const totalHandlingTimeMinutes = totalHandlingTime / 60;
    
    // Required HC = Total Handling Time / Effective Minutes per Agent
    return Math.ceil(totalHandlingTimeMinutes / effectiveMinutesPerAgent);
  };

  // Calculate O/U (Over/Under)
  const calculateOU = (actual: number, required: number) => {
    return actual - required;
  };

  // Transform the data for AG Grid
  const rowData = useMemo(() => {
    if (!weekData || !weekData.length) return [];
    
    // Create rows for each metric
    const metrics = [
      { metricName: "Volume", metricKey: "volume", section: "Volume", isInput: true },
      { metricName: "AHT", metricKey: "aht", section: "Assumptions", isInput: true },
      { metricName: "Shrinkage", metricKey: "shrinkage", section: "Assumptions", isInput: true },
      { metricName: "Occupancy", metricKey: "occupancy", section: "Assumptions", isInput: true },
      { metricName: "Attrition", metricKey: "attrition", section: "Assumptions", isInput: true },
      { metricName: "Required", metricKey: "required", section: "Factors", isInput: false },
      { metricName: "Actual", metricKey: "actual", section: "Factors", isInput: true },
      { metricName: "O/U", metricKey: "ou", section: "Factors", isInput: false },
    ];
    
    // Add section headers
    const sections = [...new Set(metrics.map(m => m.section))];
    const rows: any[] = [];
    
    sections.forEach(section => {
      // Add section header
      rows.push({
        metricName: section,
        isHeader: true
      });
      
      // Add metrics for this section
      metrics.filter(m => m.section === section).forEach(metric => {
        const row: any = {
          metricName: metric.metricName,
          metricKey: metric.metricKey,
          isInput: metric.isInput,
          isHeader: false
        };
        
        // Add data for each week
        weekData.forEach((week, weekIndex) => {
          if (metric.metricKey === 'required') {
            row[`week${weekIndex}`] = calculateRequired(week);
          } else if (metric.metricKey === 'ou') {
            const required = calculateRequired(week);
            row[`week${weekIndex}`] = calculateOU(week.actual, required);
          } else if (week[metric.metricKey as keyof WeekData] !== undefined) {
            row[`week${weekIndex}`] = week[metric.metricKey as keyof WeekData];
          } else {
            row[`week${weekIndex}`] = 0; // Default value if undefined
          }
        });
        
        rows.push(row);
      });
    });
    
    return rows;
  }, [weekData]);

  // Generate dynamic columns based on weekData
  const columnDefs = useMemo(() => {
    const cols: ColDef[] = [
      {
        headerName: 'Metric',
        field: 'metricName',
        pinned: 'left',
        width: 150,
        cellRenderer: (params: ICellRendererParams) => {
          if (params.data.isHeader) {
            return SectionHeaderRenderer(params);
          } else {
            return LabelCellRenderer(params);
          }
        },
        cellClass: (params: CellClassParams) => {
          return params.data.isHeader 
            ? 'bg-blue-100 dark:bg-blue-950 text-blue-900 dark:text-blue-100' 
            : 'sticky left-0 z-10 bg-white dark:bg-gray-900'; // Add bg color to make sticky work
        }
      }
    ];
    
    // Add columns for each week
    if (weekData && weekData.length > 0) {
      weekData.forEach((week, index) => {
        cols.push({
          headerName: week.date,
          field: `week${index}`,
          width: 120,
          cellRenderer: (params: ICellRendererParams) => {
            if (params.data.isHeader) return null;
            if (params.data.isInput) {
              return InputCellRenderer(params);
            } else if (params.data.metricKey === 'ou') {
              return OverUnderCellRenderer(params);
            }
            return params.value;
          },
          cellClass: (params: CellClassParams) => {
            if (params.data.isHeader) return 'hidden';
            
            if (params.data.metricKey === 'required') {
              return 'bg-muted dark:bg-muted font-medium';
            }
            
            if (params.data.metricKey === 'ou') {
              const value = params.value;
              return value < 0 
                ? 'bg-red-50 dark:bg-red-950' 
                : value > 0 
                ? 'bg-green-50 dark:bg-green-950' 
                : '';
            }
            
            return '';
          }
        });
      });
    }
    
    return cols;
  }, [weekData]);

  // Fixed the type for onGridReady
  const onGridReady = useCallback((params: GridReadyEvent) => {
    setGridApi(params.api);
    // Auto-size columns after grid is ready
    setTimeout(() => {
      params.api.sizeColumnsToFit();
    }, 0);
  }, []);
  
  const gridContext = useMemo(() => ({
    updateData: onDataChange,
    metricExplanations: metricExplanations
  }), [onDataChange, metricExplanations]);

  const defaultColDef = useMemo(() => ({
    sortable: false,
    resizable: true,
  }), []);

  return (
    <div className="ag-theme-alpine dark:ag-theme-alpine-dark w-full h-[500px]">
      <AgGridReact
        rowData={rowData}
        columnDefs={columnDefs as any}
        defaultColDef={defaultColDef}
        context={gridContext}
        onGridReady={onGridReady}
        domLayout="normal"
        suppressMovableColumns={true}
        headerHeight={40}
        rowHeight={48}
        suppressRowClickSelection={true}
        enableCellTextSelection={true}
        suppressCellFocus={false}
      />
    </div>
  );
};

export default PlanningGrid;
