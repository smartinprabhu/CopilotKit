import React from "react";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator } from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { ChevronDown } from "lucide-react";

interface PlanDropdownProps {
  onSelectPlan: (plan: string) => void;
}

const PlanDropdown: React.FC<PlanDropdownProps> = ({ onSelectPlan }) => {
  const planOptions = ["Move In", "Move Out", "New Hire Batch", "New Hire Production"];

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" className="w-full lg:w-[200px] text-sm h-9 justify-between">
          <div className="flex items-center truncate">
            <span className="truncate">Plan</span>
          </div>
          <ChevronDown className="h-4 w-4 opacity-50 flex-shrink-0" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-full md:w-[200px]">
        <DropdownMenuLabel>Select Plan</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {planOptions.map((option) => (
          <DropdownMenuItem key={option} onSelect={() => onSelectPlan(option)}>
            {option}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default PlanDropdown;
