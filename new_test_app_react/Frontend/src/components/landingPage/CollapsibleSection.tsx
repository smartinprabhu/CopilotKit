import React, { useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronRight, Minimize2, Maximize2, Pin, PinOff } from "lucide-react";
import { cn } from "@/lib/utils";

interface CollapsibleSectionProps {
  title: string;
  children: React.ReactNode;
  defaultExpanded?: boolean;
  headerColor?: string;
  icon?: React.ReactNode;
  actions?: React.ReactNode;
  sticky?: boolean;
  className?: string;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({
  title,
  children,
  defaultExpanded = true,
  headerColor = "bg-primary/10",
  icon,
  actions,
  sticky = false,
  className
}) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const [isMinimized, setIsMinimized] = useState(false);
  const [isPinned, setIsPinned] = useState(sticky);

  const toggleExpanded = () => setIsExpanded(!isExpanded);
  const toggleMinimized = () => setIsMinimized(!isMinimized);
  const togglePinned = () => setIsPinned(!isPinned);

  return (
    <Card className={cn(
      "transition-all duration-200 border-l-4 border-l-primary",
      isPinned && "sticky top-4 z-10 shadow-lg",
      isMinimized && "h-16 overflow-hidden",
      className
    )}>
      <CardHeader 
        className={cn(
          "cursor-pointer select-none transition-colors duration-200 hover:opacity-80",
          headerColor,
          "border-b border-primary/20"
        )}
        onClick={toggleExpanded}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Button
              variant="ghost"
              size="sm"
              className="p-1 h-6 w-6 hover:bg-primary/20"
              onClick={(e) => {
                e.stopPropagation();
                toggleExpanded();
              }}
            >
              {isExpanded ? (
                <ChevronDown className="h-4 w-4 text-primary" />
              ) : (
                <ChevronRight className="h-4 w-4 text-primary" />
              )}
            </Button>
            
            {icon && (
              <div className="text-primary">
                {icon}
              </div>
            )}
            
            <h3 className="font-semibold text-lg text-primary">
              {title}
            </h3>
          </div>
          
          <div className="flex items-center space-x-2">
            {actions}
            
            <Button
              variant="ghost"
              size="sm"
              className="p-1 h-6 w-6 hover:bg-primary/20"
              onClick={(e) => {
                e.stopPropagation();
                toggleMinimized();
              }}
            >
              {isMinimized ? (
                <Maximize2 className="h-3 w-3 text-primary" />
              ) : (
                <Minimize2 className="h-3 w-3 text-primary" />
              )}
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              className="p-1 h-6 w-6 hover:bg-primary/20"
              onClick={(e) => {
                e.stopPropagation();
                togglePinned();
              }}
            >
              {isPinned ? (
                <PinOff className="h-3 w-3 text-primary" />
              ) : (
                <Pin className="h-3 w-3 text-primary" />
              )}
            </Button>
          </div>
        </div>
      </CardHeader>
      
      {isExpanded && !isMinimized && (
        <CardContent className="p-4">
          {children}
        </CardContent>
      )}
    </Card>
  );
};

export default CollapsibleSection;