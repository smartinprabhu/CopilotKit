
import React, { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import axios from 'axios';
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";

import AuthService from "@/auth/utils/authService";

import AppConfig from '../auth/config.js';

interface EntityFormProps {
  entityType: string;
  editingItem?: any;
  onClose: () => void;
}

export function EntityForm({ entityType, editingItem, onClose }: EntityFormProps) {
  const [formData, setFormData] = useState<Record<string, any>>({});
  const { toast } = useToast();

      const WEBAPPAPIURL =  `${AppConfig.API_URL}/api/v2/`;

  useEffect(() => {
    if (editingItem) {
      setFormData(editingItem);
    } else {
      // Initialize with empty form based on entity type
      const initialData = getInitialFormData(entityType);
      setFormData(initialData);
    }
  }, [entityType, editingItem]);

  const getInitialFormData = (type: string): Record<string, any> => {
    const initialForms: Record<string, Record<string, any>> = {
      "business-units": { name: "", code: "", display_name: "", end_date: "", start_date: "" },
      "line-of-business": { name: "", code: "", category: "", revenue: "", status: "Active" },
      "departments": { name: "", code: "", manager: "", employees: "", budget: "" },
      "locations": { name: "", code: "", address: "", type: "Warehouse", capacity: "" },
      "feed-parameters": { name: "", code: "", frequency: "Daily", status: "Active" },
      "calendar-weeks": { week: "", startDate: "", endDate: "", year: new Date().getFullYear(), status: "Upcoming" },
      "system-settings": { category: "", setting: "", value: "", modifiedBy: "Admin" },
      "status-codes": { code: "", description: "", category: "General", color: "blue", usage: "Medium" },
    };
    return initialForms[type] || {};
  };

  const getFormFields = (type: string) => {
    const fieldMap: Record<string, Array<{name: string, label: string, type: string, options?: string[], required?: boolean}>> = {
      "business-units": [
        { name: "name", label: "Business Unit Name", type: "text", required: true },
        { name: "code", label: "Code", type: "text", required: true },
        { name: "display_name", label: "Display Name", type: "text", required: false },
        { name: "start_date", label: "Start Date", type: "date", required: false },
        { name: "end_date", label: "End Date", type: "date", required: false },
      ],
      "line-of-business": [
        { name: "name", label: "Line of Business Name", type: "text" },
        { name: "code", label: "Code", type: "text" },
        { name: "category", label: "Category", type: "text" },
        { name: "revenue", label: "Revenue", type: "text" },
        { name: "status", label: "Status", type: "select", options: ["Active", "Inactive"] },
      ],
      "departments": [
        { name: "name", label: "Department Name", type: "text" },
        { name: "code", label: "Code", type: "text" },
        { name: "manager", label: "Manager", type: "text" },
        { name: "employees", label: "Number of Employees", type: "number" },
        { name: "budget", label: "Budget", type: "text" },
      ],
      "locations": [
        { name: "name", label: "Location Name", type: "text" },
        { name: "code", label: "Code", type: "text" },
        { name: "address", label: "Address", type: "textarea" },
        { name: "type", label: "Type", type: "select", options: ["Warehouse", "Store", "Distribution", "Office"] },
        { name: "capacity", label: "Capacity", type: "text" },
      ],
      "feed-parameters": [
        { name: "name", label: "Parameter Name", type: "text" },
        { name: "code", label: "Code", type: "text" },
        { name: "frequency", label: "Frequency", type: "select", options: ["Hourly", "Daily", "Weekly", "Monthly"] },
        { name: "status", label: "Status", type: "select", options: ["Active", "Inactive", "Running", "Failed"] },
      ],
      "calendar-weeks": [
        { name: "week", label: "Week", type: "text" },
        { name: "startDate", label: "Start Date", type: "date" },
        { name: "endDate", label: "End Date", type: "date" },
        { name: "year", label: "Year", type: "number" },
        { name: "status", label: "Status", type: "select", options: ["Current", "Completed", "Upcoming"] },
      ],
      "system-settings": [
        { name: "category", label: "Category", type: "text" },
        { name: "setting", label: "Setting Name", type: "text" },
        { name: "value", label: "Value", type: "text" },
        { name: "modifiedBy", label: "Modified By", type: "text" },
      ],
      "status-codes": [
        { name: "code", label: "Status Code", type: "text" },
        { name: "description", label: "Description", type: "text" },
        { name: "category", label: "Category", type: "text" },
        { name: "color", label: "Color", type: "select", options: ["green", "red", "yellow", "blue", "purple"] },
        { name: "usage", label: "Usage", type: "select", options: ["High", "Medium", "Low"] },
      ],
    };
    return fieldMap[type] || [];
  };

  const handleInputChange = (name: string, value: any) => {
    setFormData(prev => ({ ...prev, [name]: value }));
  };

 type ForecastFormData = {
  name: string;
  code: string;
  display_name: string;
  start_date: string;
  end_date: string;
};

async function submitForecastForm(model: string, values: ForecastFormData) {
  const formData = new FormData();

  formData.append("model", model);
  formData.append("values", JSON.stringify(values)); // nested object as JSON string

  const url = `${WEBAPPAPIURL}create`;

  const response = await axios.post(url, formData, {
    headers: {
      "Content-Type": "multipart/form-data",
      Authorization: `Bearer ${AuthService.getAccessToken()}`
    },
  });

  return response.data;
}

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      const response = await submitForecastForm("business.unit",{
        name: formData.name,
        display_name: formData.display_name,
        code: formData.code,
        start_date: formData.start_date,
        end_date: formData.end_date,
      });
      toast({
        title: editingItem ? "Entity Updated" : "Entity Created",
        description: `${entityType} has been ${editingItem ? 'updated' : 'created'} successfully.`,
      });
      onClose();
    } catch (err) {
      console.error("API Error:", err);
    }
    
  };

  const fields = getFormFields(entityType);

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {fields.map((field) => (
          <div key={field.name} className={field.type === 'textarea' ? 'md:col-span-2' : ''}>
            <Label htmlFor={field.name} className="text-slate-300">
              {field.label}
            </Label>
            {field.type === 'select' ? (
              <Select 
                value={formData[field.name] || ''} 
                onValueChange={(value) => handleInputChange(field.name, value)}
              >
                <SelectTrigger className="bg-slate-700 border-slate-600 text-white">
                  <SelectValue placeholder={`Select ${field.label}`} />
                </SelectTrigger>
                <SelectContent className="bg-slate-700 border-slate-600">
                  {field.options?.map((option) => (
                    <SelectItem key={option} value={option} className="text-white hover:bg-slate-600">
                      {option}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : field.type === 'textarea' ? (
              <Textarea
                id={field.name}
                value={formData[field.name] || ''}
                onChange={(e) => handleInputChange(field.name, e.target.value)}
                className="bg-slate-700 border-slate-600 text-white placeholder:text-slate-400"
                rows={3}
              />
            ) : (
              <Input
                id={field.name}
                type={field.type}
                value={formData[field.name] || ''}
                required={field.required}
                onChange={(e) => handleInputChange(field.name, e.target.value)}
                className="bg-slate-700 border-slate-600 text-white placeholder:text-slate-400"
              />
            )}
          </div>
        ))}
      </div>
      
      <div className="flex justify-end gap-3 pt-4">
        <Button type="button" variant="outline" onClick={onClose} className="border-slate-600 text-slate-300 hover:text-white">
          Cancel
        </Button>
        <Button type="submit" className="bg-blue-600 hover:bg-blue-700">
          {editingItem ? 'Update' : 'Create'}
        </Button>
      </div>
    </form>
  );
}
