import React from "react";
import { Button } from "@/components/ui/button";
import { Settings } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

const FilterAggregation = ({ startDate, setStartDate, endDate, setEndDate, aggregationType, setAggregationType }) => {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="default"
          size="sm"
          className="flex items-center gap-2 font-semibold hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
        >
          <Settings className="h-4 w-4" />
          Filter & Aggregation
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-64 space-y-4 bg-white dark:bg-gray-900 shadow-lg border border-gray-200 dark:border-gray-700">
        <div>
          <label className="block text-sm font-medium mb-2">Date Range</label>
          <div className="space-y-2">
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full border rounded px-2 py-1 text-sm dark:bg-gray-700 dark:text-white"
              placeholder="Start Date"
            />
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full border rounded px-2 py-1 text-sm dark:bg-gray-700 dark:text-white"
              placeholder="End Date"
            />
          </div>
        </div>
        <div>
          <label className="block text-sm font-medium mb-2">Aggregation</label>
          <select
            value={aggregationType}
            onChange={(e) => setAggregationType(e.target.value)}
            className="w-full border rounded px-2 py-1 text-sm dark:bg-gray-700 dark:text-white"
          >
            <option value="Weekly">Weekly</option>
            <option value="Monthly">Monthly</option>
          </select>
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default FilterAggregation;
