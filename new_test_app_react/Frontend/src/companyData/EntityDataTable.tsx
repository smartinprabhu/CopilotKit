
import React, { useState, useEffect } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import axios from 'axios';
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Edit, Trash2, MoreHorizontal } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

import AuthService from "@/auth/utils/authService";

import AppConfig from '../auth/config.js';

interface EntityDataTableProps {
  entityType: string;
  searchTerm: string;
  reload: string;
  onEdit: (item: any) => void;
}

export function EntityDataTable({ entityType, searchTerm, reload, onEdit }: EntityDataTableProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

    const WEBAPPAPIURL =  `${AppConfig.API_URL}/api/v2/`;

  // Mock data generator based on entity type
  const generateMockData = (type: string) => {
    const baseData: Record<string, any[]> = {
      "business-units": [
        { id: 1, name: "Retail Operations", code: "RET001", description: "Main retail business unit", status: "Active", createdDate: "2024-01-15" },
        { id: 2, name: "E-commerce", code: "ECM002", description: "Online sales division", status: "Active", createdDate: "2024-02-20" },
        { id: 3, name: "Wholesale", code: "WHO003", description: "B2B wholesale operations", status: "Inactive", createdDate: "2024-03-10" },
        { id: 4, name: "Logistics", code: "LOG004", description: "Supply chain and logistics", status: "Active", createdDate: "2024-04-05" },
      ],
      "line-of-business": [
        { id: 1, name: "Consumer Electronics", code: "CE001", category: "Technology", revenue: "$2.5M", status: "Active" },
        { id: 2, name: "Home & Garden", code: "HG002", category: "Lifestyle", revenue: "$1.8M", status: "Active" },
        { id: 3, name: "Automotive", code: "AU003", category: "Industrial", revenue: "$3.2M", status: "Active" },
      ],
      "departments": [
        { id: 1, name: "Human Resources", code: "HR001", manager: "Sarah Johnson", employees: 25, budget: "$500K" },
        { id: 2, name: "Information Technology", code: "IT002", manager: "Mike Chen", employees: 45, budget: "$1.2M" },
        { id: 3, name: "Finance", code: "FIN003", manager: "Emily Davis", employees: 18, budget: "$300K" },
      ],
      "locations": [
        { id: 1, name: "New York Warehouse", code: "NY001", address: "123 Industrial Ave", type: "Warehouse", capacity: "50,000 sq ft" },
        { id: 2, name: "Los Angeles Store", code: "LA002", address: "456 Retail Blvd", type: "Store", capacity: "15,000 sq ft" },
        { id: 3, name: "Chicago Distribution", code: "CH003", address: "789 Logistics Dr", type: "Distribution", capacity: "75,000 sq ft" },
      ],
      "feed-parameters": [
        { id: 1, name: "Inventory Sync", code: "INV_SYNC", frequency: "Hourly", lastRun: "2024-05-30 14:30", status: "Running" },
        { id: 2, name: "Price Updates", code: "PRICE_UPD", frequency: "Daily", lastRun: "2024-05-30 06:00", status: "Completed" },
        { id: 3, name: "Customer Data", code: "CUST_DATA", frequency: "Weekly", lastRun: "2024-05-27 23:00", status: "Failed" },
      ],
      "calendar-weeks": [
        { id: 1, week: "Week 22", startDate: "2024-05-27", endDate: "2024-06-02", year: 2024, status: "Current" },
        { id: 2, week: "Week 21", startDate: "2024-05-20", endDate: "2024-05-26", year: 2024, status: "Completed" },
        { id: 3, week: "Week 23", startDate: "2024-06-03", endDate: "2024-06-09", year: 2024, status: "Upcoming" },
      ],
      "system-settings": [
        { id: 1, category: "Security", setting: "Password Policy", value: "Strong", lastModified: "2024-05-15", modifiedBy: "Admin" },
        { id: 2, category: "Performance", setting: "Cache Duration", value: "24 hours", lastModified: "2024-05-20", modifiedBy: "System" },
        { id: 3, category: "Integration", setting: "API Rate Limit", value: "1000/hour", lastModified: "2024-05-25", modifiedBy: "Tech Lead" },
      ],
      "status-codes": [
        { id: 1, code: "ACT", description: "Active", category: "General", color: "green", usage: "High" },
        { id: 2, code: "INA", description: "Inactive", category: "General", color: "red", usage: "Medium" },
        { id: 3, code: "PEN", description: "Pending", category: "Workflow", color: "yellow", usage: "Low" },
      ],
    };

    return baseData[type] || [];
  };

  const [apiData, setData] = useState([]);

const fetchData = async (type: string) => {
  let params = new URLSearchParams({
    limit: '100',
    offset: '0',
    domain: '[]',
    fields: '["name","code","client_id","start_date","end_date","enterprise_group_id"]',
    model: 'business.unit',
  });

  if(entityType === 'line-of-business'){
    params = new URLSearchParams({
        limit: '100',
        offset: '0',
        domain: '[]',
        fields: '["name", "code", "client_id", "start_date", "end_date", "business_unit"]',
        model: 'line_business_lob',
    });
  }

  if(entityType === 'feed-parameters'){
    params = new URLSearchParams({
        offset: '0',
        domain: '[]',
        fields: '["date", "calendar_week_id", "business_unit_id", "lob_id", "bpo_id", "week_no", "parameter_id", "value"]',
        model: 'data_feeds',
    });
  }

  setData([]);

  try {
    const response = await axios.get(`${WEBAPPAPIURL}search_read?${params.toString()}`, {
      headers: {
        Authorization: `Bearer ${AuthService.getAccessToken()}`
      },
    });
    return response.data;
  } catch (error) {
    
    console.error('Error fetching data:', error);
  }
};

  useEffect(() => {
    const runFetch = async () => {
        if(entityType === 'business-units' || entityType === 'line-of-business' || entityType === 'feed-parameters'){
            try {
                const result = await fetchData(entityType); // call the function
                setData(result); // store the data in state
            } catch (err) {
              setData([]);
                console.error('Failed to fetch:', err);
            }
        };
}

    runFetch();
  }, [entityType, reload]);


  const data = entityType === 'business-units' || entityType === 'line-of-business' || entityType === 'feed-parameters' ?  apiData : generateMockData(entityType);
  
  // Filter data based on search term
  const filteredData = data && data?.length ? data.filter((item) =>
    Object.values(item).some(value => 
      value?.toString().toLowerCase().includes(searchTerm.toLowerCase())
    )
  ) : [];

  // Pagination
  const totalPages = filteredData.length > 0 ? Math.ceil(filteredData.length / itemsPerPage) : 0;
  const paginatedData = data && data?.length ? filteredData.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  ) : [];

  const getColumns = (type: string) => {
    const columnMap: Record<string, string[]> = {
      "business-units": ["Name", "Code", "Client", "Start Date", "End Date", "Enterprise Group", "Actions"],
      "line-of-business": ["Name", "Code", "Client", "Start Date", "End Date", "Business Unit", "Actions"],
      "departments": ["Name", "Code", "Manager", "Employees", "Budget", "Actions"],
      "locations": ["Name", "Code", "Address", "Type", "Capacity", "Actions"],
      "feed-parameters": ["Date", "Calendar Week", "Business Unit", "LOB", "BPO", "Week", "Parameter", "Value", "Actions"],
      "calendar-weeks": ["Week", "Start Date", "End Date", "Year", "Status", "Actions"],
      "system-settings": ["Category", "Setting", "Value", "Last Modified", "Modified By", "Actions"],
      "status-codes": ["Code", "Description", "Category", "Color", "Usage", "Actions"],
    };
    return columnMap[type] || ["Name", "Actions"];
  };

  const getColumnsFields = (type: string) => {
    const columnMap: Record<string, string[]> = {
      "business-units": ["name","code","client_id","start_date","end_date","enterprise_group_id",""],
      "line-of-business": ["name", "code", "client_id", "start_date", "end_date", "business_unit", ""],
      "departments": ["Name", "Code", "Manager", "Employees", "Budget", "Actions"],
      "locations": ["Name", "Code", "Address", "Type", "Capacity", "Actions"],
      "feed-parameters": ["date", "calendar_week_id", "business_unit_id", "lob_id", "bpo_id", "week_no", "parameter_id", "value", ""],
      "calendar-weeks": ["Week", "Start Date", "End Date", "Year", "Status", "Actions"],
      "system-settings": ["Category", "Setting", "Value", "Last Modified", "Modified By", "Actions"],
      "status-codes": ["Code", "Description", "Category", "Color", "Usage", "Actions"],
    };
    return columnMap[type] || ["Name", "Actions"];
  };

  const columns = getColumns(entityType);

  const columnsFields = getColumnsFields(entityType);

  const renderCellValue = (item: any, columnIndex: number) => {
    const keys = Object.keys(item).filter(key => key !== 'id');
    
    if (columnIndex >= keys.length) {
      // Actions column
      return (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="h-8 w-8 p-0 text-foreground hover:text-white">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="bg-slate-800 border-slate-600">
            <DropdownMenuItem 
              onClick={() => onEdit(item)}
              className="text-foreground hover:text-white focus:text-white"
            >
              <Edit className="mr-2 h-4 w-4" />
              Edit
            </DropdownMenuItem>
            <DropdownMenuItem className="text-red-400 hover:text-red-300 focus:text-red-300">
              <Trash2 className="mr-2 h-4 w-4" />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      );
    }

    const key = keys[columnIndex];
    const value = item[key];

    // Special rendering for status and other badge-worthy fields
    if (key === 'status' || key === 'color') {
      const variant = value === 'Active' || value === 'Running' || value === 'Current' 
        ? 'default' 
        : value === 'Inactive' || value === 'Failed' 
        ? 'destructive' 
        : 'secondary';
      
      return <Badge variant={variant} className="text-xs">{value}</Badge>;
    }

    return <span className="text-sm">{value}</span>;
  };

  const renderCellValueDynamic = (field: string | undefined, item: any) => {
  if (!field) {
    // Render actions column
    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" className="h-8 w-8 p-0 text-foreground hover:text-white">
            <MoreHorizontal className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="bg-slate-800 border-slate-600">
          <DropdownMenuItem
            onClick={() => onEdit(item)}
            className="text-foreground hover:text-white focus:text-white"
          >
            <Edit className="mr-2 h-4 w-4" />
            Edit
          </DropdownMenuItem>
          <DropdownMenuItem className="text-red-400 hover:text-red-300 focus:text-red-300">
            <Trash2 className="mr-2 h-4 w-4" />
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    );
  }

  console.log(typeof item[field] === 'object')
  const value = typeof item[field] === 'object' ? item[field][1] : item[field];

  // Render badge for status or color fields
  if (field === 'status' || field === 'color') {
    const variant =
      value === 'Active' || value === 'Running' || value === 'Current'
        ? 'default'
        : value === 'Inactive' || value === 'Failed'
        ? 'destructive'
        : 'secondary';

    return <Badge variant={variant} className="text-xs">{value}</Badge>;
  }

  // Default text rendering
  return <span className="text-sm">{value}</span>;
};


  return (
    <div className="bg-sidebar rounded-lg border border-slate-700">
      <Table>
        <TableHeader>
          <TableRow className="border-slate-700 hover:bg-slate-700/50">
            {columns.map((column, index) => (
              <TableHead key={index} className="text-foreground font-medium">
                {column}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        {entityType !== 'business-units' && entityType !== 'line-of-business' && entityType !== 'feed-parameters' && (
        <TableBody>
          {paginatedData.map((item) => (
            <TableRow key={item.id} className="border-slate-700 hover:bg-slate-700/50">
              {columns.map((_, columnIndex) => (
                <TableCell key={columnIndex} className="text-foreground">
                  {renderCellValue(item, columnIndex)}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
        )}
        {(entityType === 'business-units' || entityType === 'line-of-business' || entityType === 'feed-parameters') && (
        <TableBody>
          {paginatedData.map((item) => (
            <TableRow key={item.id} className="border-slate-700 hover:bg-slate-700/50">
              {columnsFields.map((column, colIndex) => (
                <TableCell key={colIndex} className="text-foreground">
                    {renderCellValueDynamic(column, item)}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
        )}
      </Table>
      
      {totalPages > 1 && (
        <div className="flex items-center justify-between p-4 border-t border-slate-700">
          <div className="text-sm text-slate-400">
            Showing {((currentPage - 1) * itemsPerPage) + 1} to {Math.min(currentPage * itemsPerPage, filteredData.length)} of {filteredData.length} entries
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              className="border-slate-600 text-foreground hover:text-white"
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              className="border-slate-600 text-foreground hover:text-white"
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
