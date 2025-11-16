import React, { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { 
  ChevronLeft, 
  ChevronRight, 
  Search, 
  Filter, 
  Download, 
  Upload,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Save
} from "lucide-react";
import { cn } from "@/lib/utils";

interface TableColumn {
  key: string;
  label: string;
  type?: 'text' | 'number' | 'percentage' | 'currency';
  editable?: boolean;
  width?: string;
}

interface EnhancedPlanningTableProps {
  title: string;
  columns: TableColumn[];
  data: any[];
  onDataChange?: (data: any[]) => void;
  className?: string;
}

const EnhancedPlanningTable: React.FC<EnhancedPlanningTableProps> = ({
  title,
  columns,
  data,
  onDataChange,
  className
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [zoom, setZoom] = useState(100);
  const [editingCell, setEditingCell] = useState<{row: number, col: string} | null>(null);
  const [localData, setLocalData] = useState(data);
  const tableRef = useRef<HTMLDivElement>(null);
  const itemsPerPage = 20;

  useEffect(() => {
    setLocalData(data);
  }, [data]);

  const filteredData = localData.filter(row =>
    Object.values(row).some(value =>
      String(value).toLowerCase().includes(searchTerm.toLowerCase())
    )
  );

  const totalPages = Math.ceil(filteredData.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const paginatedData = filteredData.slice(startIndex, startIndex + itemsPerPage);

  const handleCellEdit = (rowIndex: number, columnKey: string, value: any) => {
    const actualRowIndex = startIndex + rowIndex;
    const newData = [...localData];
    newData[actualRowIndex] = { ...newData[actualRowIndex], [columnKey]: value };
    setLocalData(newData);
    onDataChange?.(newData);
  };

  const formatCellValue = (value: any, type?: string) => {
    switch (type) {
      case 'currency':
        return `$${Number(value).toLocaleString()}`;
      case 'percentage':
        return `${Number(value).toFixed(1)}%`;
      case 'number':
        return Number(value).toLocaleString();
      default:
        return value;
    }
  };

  const scrollToColumn = (direction: 'left' | 'right') => {
    if (tableRef.current) {
      const scrollAmount = 200;
      tableRef.current.scrollBy({
        left: direction === 'left' ? -scrollAmount : scrollAmount,
        behavior: 'smooth'
      });
    }
  };

  return (
    <Card className={cn("border-l-4 border-l-primary", className)}>
      {/* Enhanced Header */}
      <CardHeader className="bg-primary/5 border-b border-primary/20">
        <div className="flex items-center justify-between">
          <CardTitle className="text-primary flex items-center space-x-2">
            <span>{title}</span>
            <Badge variant="secondary" className="bg-primary/20 text-primary">
              {filteredData.length} rows
            </Badge>
          </CardTitle>
          
          <div className="flex items-center space-x-2">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-8 w-48"
              />
            </div>
            
            {/* Zoom Controls */}
            <div className="flex items-center space-x-1 border rounded-md p-1">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setZoom(Math.max(50, zoom - 10))}
                className="h-6 w-6 p-0"
              >
                <ZoomOut className="h-3 w-3" />
              </Button>
              <span className="text-xs px-2">{zoom}%</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setZoom(Math.min(200, zoom + 10))}
                className="h-6 w-6 p-0"
              >
                <ZoomIn className="h-3 w-3" />
              </Button>
            </div>
            
            {/* Action Buttons */}
            <Button variant="outline" size="sm">
              <Upload className="h-4 w-4 mr-1" />
              Import
            </Button>
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4 mr-1" />
              Export
            </Button>
            <Button variant="default" size="sm" className="bg-primary">
              <Save className="h-4 w-4 mr-1" />
              Save
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-0">
        {/* Navigation Controls */}
        <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-800 border-b">
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => scrollToColumn('left')}
            >
              <ChevronLeft className="h-4 w-4" />
              Scroll Left
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => scrollToColumn('right')}
            >
              Scroll Right
              <ChevronRight className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setZoom(100)}
            >
              <RotateCcw className="h-4 w-4 mr-1" />
              Reset View
            </Button>
          </div>
          
          {/* Pagination */}
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
              disabled={currentPage === 1}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="text-sm">
              Page {currentPage} of {totalPages}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
              disabled={currentPage === totalPages}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Enhanced Table */}
        <div 
          ref={tableRef}
          className="overflow-auto max-h-96"
          style={{ fontSize: `${zoom}%` }}
        >
          <table className="w-full">
            {/* Branded Header */}
            <thead className="sticky top-0 z-10">
              <tr className="bg-primary text-primary-foreground">
                {columns.map((column) => (
                  <th
                    key={column.key}
                    className="px-4 py-3 text-left font-semibold border-r border-primary-foreground/20 last:border-r-0"
                    style={{ width: column.width }}
                  >
                    {column.label}
                  </th>
                ))}
              </tr>
            </thead>
            
            <tbody>
              {paginatedData.map((row, rowIndex) => (
                <tr
                  key={rowIndex}
                  className="border-b border-gray-200 dark:border-gray-700 hover:bg-primary/5 transition-colors"
                >
                  {columns.map((column) => (
                    <td
                      key={column.key}
                      className="px-4 py-2 border-r border-gray-200 dark:border-gray-700 last:border-r-0"
                    >
                      {column.editable && editingCell?.row === rowIndex && editingCell?.col === column.key ? (
                        <Input
                          value={row[column.key]}
                          onChange={(e) => handleCellEdit(rowIndex, column.key, e.target.value)}
                          onBlur={() => setEditingCell(null)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') setEditingCell(null);
                          }}
                          className="h-8 text-sm"
                          autoFocus
                        />
                      ) : (
                        <div
                          className={cn(
                            "cursor-pointer hover:bg-primary/10 rounded px-2 py-1 transition-colors",
                            column.editable && "border border-transparent hover:border-primary/30"
                          )}
                          onClick={() => column.editable && setEditingCell({ row: rowIndex, col: column.key })}
                        >
                          {formatCellValue(row[column.key], column.type)}
                        </div>
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
};

export default EnhancedPlanningTable;