import React, { useState } from "react";
import CollapsibleSection from "./CollapsibleSection";
import EnhancedPlanningTable from "./EnhancedPlanningTable";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  Users, 
  Calculator, 
  TrendingUp, 
  Settings, 
  Save, 
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  Info
} from "lucide-react";

const EnhancedPlanningPage: React.FC = () => {
  // Sample data for different sections
  const assumptionsColumns = [
    { key: 'parameter', label: 'Parameter', type: 'text' as const, width: '200px' },
    { key: 'value', label: 'Value', type: 'number' as const, editable: true, width: '120px' },
    { key: 'unit', label: 'Unit', type: 'text' as const, width: '100px' },
    { key: 'source', label: 'Source', type: 'text' as const, width: '150px' },
    { key: 'lastUpdated', label: 'Last Updated', type: 'text' as const, width: '120px' },
  ];

  const assumptionsData = [
    { parameter: 'Average Handle Time', value: 12.5, unit: 'minutes', source: 'Historical', lastUpdated: '2024-01-15' },
    { parameter: 'Service Level Target', value: 80, unit: '%', source: 'SLA', lastUpdated: '2024-01-10' },
    { parameter: 'Occupancy Target', value: 75, unit: '%', source: 'Best Practice', lastUpdated: '2024-01-12' },
    { parameter: 'Shrinkage Rate', value: 25, unit: '%', source: 'Historical', lastUpdated: '2024-01-14' },
  ];

  const headcountColumns = [
    { key: 'week', label: 'Week', type: 'text' as const, width: '100px' },
    { key: 'required', label: 'Required HC', type: 'number' as const, editable: true, width: '120px' },
    { key: 'planned', label: 'Planned HC', type: 'number' as const, editable: true, width: '120px' },
    { key: 'actual', label: 'Actual HC', type: 'number' as const, width: '120px' },
    { key: 'variance', label: 'Variance', type: 'number' as const, width: '100px' },
    { key: 'utilization', label: 'Utilization %', type: 'percentage' as const, width: '120px' },
  ];

  const headcountData = [
    { week: 'W1', required: 120, planned: 115, actual: 118, variance: -2, utilization: 78.5 },
    { week: 'W2', required: 135, planned: 130, actual: 128, variance: -7, utilization: 82.1 },
    { week: 'W3', required: 125, planned: 125, actual: 127, variance: 2, utilization: 76.8 },
    { week: 'W4', required: 140, planned: 135, actual: 138, variance: -2, utilization: 85.2 },
  ];

  const [isAutoSave, setIsAutoSave] = useState(true);
  const [lastSaved, setLastSaved] = useState(new Date());

  const handleSave = () => {
    setLastSaved(new Date());
    // Add save logic here
  };

  return (
    <div className="space-y-6 p-6">
      {/* Header with Actions */}
      <Card className="border-l-4 border-l-primary">
        <CardHeader className="bg-primary/5 border-b border-primary/20">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-primary flex items-center space-x-2">
                <Calculator className="h-6 w-6" />
                <span>Tactical Capacity Planning</span>
              </CardTitle>
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
                12-week workforce planning with real-time calculations
              </p>
            </div>
            
            <div className="flex items-center space-x-3">
              <Badge variant="outline" className="border-green-500 text-green-700">
                <CheckCircle className="h-3 w-3 mr-1" />
                Auto-saved {lastSaved.toLocaleTimeString()}
              </Badge>
              
              <Button variant="outline" size="sm">
                <RefreshCw className="h-4 w-4 mr-1" />
                Refresh Data
              </Button>
              
              <Button onClick={handleSave} className="bg-primary">
                <Save className="h-4 w-4 mr-1" />
                Save Plan
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="border-l-4 border-l-blue-500">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Required HC</p>
                <p className="text-2xl font-bold text-blue-600">520</p>
              </div>
              <Users className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>
        
        <Card className="border-l-4 border-l-green-500">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Planned HC</p>
                <p className="text-2xl font-bold text-green-600">505</p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>
        
        <Card className="border-l-4 border-l-orange-500">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Variance</p>
                <p className="text-2xl font-bold text-orange-600">-15</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>
        
        <Card className="border-l-4 border-l-purple-500">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Avg Utilization</p>
                <p className="text-2xl font-bold text-purple-600">80.7%</p>
              </div>
              <Calculator className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Collapsible Sections */}
      <CollapsibleSection
        title="Planning Assumptions"
        icon={<Settings className="h-5 w-5" />}
        headerColor="bg-blue-50 dark:bg-blue-900/20"
        defaultExpanded={true}
        actions={
          <Badge variant="secondary" className="bg-blue-100 text-blue-800">
            4 parameters
          </Badge>
        }
      >
        <div className="space-y-4">
          <div className="flex items-center space-x-2 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <Info className="h-4 w-4 text-blue-600" />
            <span className="text-sm text-blue-800 dark:text-blue-200">
              These assumptions drive all capacity calculations. Changes will automatically update downstream metrics.
            </span>
          </div>
          
          <EnhancedPlanningTable
            title="Key Planning Parameters"
            columns={assumptionsColumns}
            data={assumptionsData}
          />
        </div>
      </CollapsibleSection>

      <CollapsibleSection
        title="Headcount Planning"
        icon={<Users className="h-5 w-5" />}
        headerColor="bg-green-50 dark:bg-green-900/20"
        defaultExpanded={true}
        actions={
          <Badge variant="secondary" className="bg-green-100 text-green-800">
            12 weeks
          </Badge>
        }
      >
        <div className="space-y-4">
          <div className="flex items-center space-x-2 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
            <CheckCircle className="h-4 w-4 text-green-600" />
            <span className="text-sm text-green-800 dark:text-green-200">
              Weekly headcount requirements based on forecast demand and service level targets.
            </span>
          </div>
          
          <EnhancedPlanningTable
            title="Weekly Headcount Requirements"
            columns={headcountColumns}
            data={headcountData}
          />
        </div>
      </CollapsibleSection>

      <CollapsibleSection
        title="Financial Impact"
        icon={<Calculator className="h-5 w-5" />}
        headerColor="bg-purple-50 dark:bg-purple-900/20"
        defaultExpanded={false}
        actions={
          <Badge variant="secondary" className="bg-purple-100 text-purple-800">
            Cost Analysis
          </Badge>
        }
      >
        <div className="p-4 text-center text-gray-500">
          <Calculator className="h-12 w-12 mx-auto mb-2 opacity-50" />
          <p>Financial impact calculations and cost analysis will be displayed here.</p>
        </div>
      </CollapsibleSection>
    </div>
  );
};

export default EnhancedPlanningPage;