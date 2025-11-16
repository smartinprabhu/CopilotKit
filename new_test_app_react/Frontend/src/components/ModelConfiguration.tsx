import { useState, useEffect } from "react";
import axios from 'axios';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import * as XLSX from 'xlsx';

interface ModelConfigurationProps {
  modelType: string;
  forecastPeriod: number;
  aggregationType: string;
  onModelTypeChange: (value: string) => void;
  onForecastPeriodChange: (value: number) => void;
  onAggregationTypeChange: (value: string) => void;
  onApplyChanges: (forecastPeriod: number) => void;
  file: File | null;
  previewUrl: string;
}

export const ModelConfiguration = ({
  modelType,
  forecastPeriod,
  aggregationType,
  onModelTypeChange,
  onForecastPeriodChange,
  onAggregationTypeChange,
  onApplyChanges,
  file,
  previewUrl,
}: ModelConfigurationProps) => {
  const [previewData, setPreviewData] = useState<any[]>([]);
  const { toast } = useToast();

  useEffect(() => {
    if (file) {
      loadPreviewData(file);
    }
  }, [file]);

  const loadPreviewData = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e: ProgressEvent<FileReader>) => {
      const binaryStr = e.target?.result;
      const workbook = XLSX.read(binaryStr, { type: 'binary' });
      const firstSheetName = workbook.SheetNames[0];
      const worksheet = workbook.Sheets[firstSheetName];
      const jsonData: any[] = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: '' });
      setPreviewData(jsonData.slice(0, 5));
    };
    reader.onerror = (error) => {
      console.error("Error reading file:", error);
      toast({
        title: "Error",
        description: "Failed to load preview data from the file.",
        variant: "destructive",
      });
    };
    reader.readAsBinaryString(file);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onApplyChanges(forecastPeriod);
  };

  return (
    <div className="w-full bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-100 dark:border-gray-700">
      <h2 className="text-lg font-semibold mb-4">Model Configuration</h2>
      <div className="space-y-4">
        <div>
          {file && (
            <div className="mb-4">
              <Label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Uploaded File: {file.name}
              </Label>
            </div>
          )}
          {file && (
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline" className="mt-2">Preview File</Button>
              </DialogTrigger>
              <DialogContent className="max-w-4xl">
                <DialogHeader>
                  <DialogTitle>File Preview (First 5 Rows)</DialogTitle>
                </DialogHeader>
                <ScrollArea>
                  <Table>
                    <TableHeader>
                      {previewData[0]?.map((header: any, index: number) => (
                        <TableHead key={index}>{header}</TableHead>
                      ))}
                    </TableHeader>
                    <TableBody>
                      {previewData.slice(1).map((row, rowIndex) => (
                        <TableRow key={rowIndex}>
                          {row.map((cell: any, cellIndex: number) => (
                            <TableCell key={cellIndex}>{cell}</TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </ScrollArea>
              </DialogContent>
            </Dialog>
          )}
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Model Type
          </label>
          <Select
            value={modelType}
            onValueChange={onModelTypeChange}
          >
            <SelectTrigger className="w-full dark:bg-gray-700 dark:border-gray-600">
              <SelectValue placeholder="Select model type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Prophet">Prophet</SelectItem>
              <SelectItem value="ARIMA">ARIMA</SelectItem>
              <SelectItem value="LSTM">LSTM</SelectItem>
              <SelectItem value="XGBoost">XGBoost</SelectItem>
              <SelectItem value="Hybrid">Hybrid</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Forecast Period (weeks)
          </label>
          <Input
            type="number"
            value={forecastPeriod}
            onChange={(e) => onForecastPeriodChange(Number(e.target.value))}
            min={1}
            max={52}
            className="dark:bg-gray-700 dark:border-gray-600"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Aggregation Type
          </label>
          <Select
            value={aggregationType}
            onValueChange={onAggregationTypeChange}
          >
            <SelectTrigger className="w-full dark:bg-gray-700 dark:border-gray-600">
              <SelectValue placeholder="Select aggregation" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Weekly">Weekly</SelectItem>
              <SelectItem value="Monthly">Monthly</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex justify-center mt-4">
          <Button
            type="button"
            className="bg-blue-600 hover:bg-blue-700"
            onClick={() => onApplyChanges(forecastPeriod)}
          >
            Apply Changes
          </Button>
        </div>
      </div>
    </div>
  );
};
