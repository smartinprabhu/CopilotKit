import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Settings, Download, RotateCcw } from "lucide-react";
import { ConfigurationData } from "./ContactCenterApp";
import React, { useMemo } from 'react';
import DateRangePicker from "../DateRangePicker";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { ForecastVolumeTable } from "./ForecastVolumeTable";
import { AHTTable } from "./AHTTable";
import { EnhancedRosterGrid } from "./EnhancedRosterGrid";
import { StaffingChart } from "./StaffingChart";

export interface InputConfigurationScreenProps {
  onRunSimulation: (data: ConfigurationData) => void;
  weekStartsOn: 0 | 1 | 2 | 3 | 4 | 5 | 6;
}

const volumeData = [
  {
    "time": "12:00 AM",
    "values": [8, 3, 19, 21, 34, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 9, 3, 22, 25, 41, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 11, 4, 27, 30, 49, 34, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 12, 4, 31, 35, 56, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
  },
  {
    "time": "12:30 AM",
    "values": [8, 2, 21, 26, 32, 27, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 9, 2, 26, 32, 38, 33, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 11, 3, 31, 38, 46, 39, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 13, 3, 35, 43, 52, 45, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37]
  },
  {
    "time": "1:00 AM",
    "values": [6, 2, 18, 23, 28, 24, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 7, 2, 22, 27, 33, 28, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 9, 2, 26, 32, 40, 34, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 10, 2, 30, 37, 46, 39, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36]
  },
  {
    "time": "1:30 AM",
    "values": [7, 2, 23, 25, 29, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 8, 2, 28, 30, 35, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 10, 3, 34, 36, 42, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 11, 3, 39, 41, 48, 39, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38]
  },
  {
    "time": "2:00 AM",
    "values": [6, 2, 18, 35, 30, 29, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 7, 3, 21, 42, 36, 35, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 8, 3, 26, 51, 43, 42, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 10, 4, 29, 59, 49, 48, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43]
  },
  {
    "time": "2:30 AM",
    "values": [7, 2, 20, 26, 30, 25, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 8, 3, 24, 31, 36, 30, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 10, 3, 29, 37, 44, 37, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 11, 4, 33, 43, 50, 42, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36, 36]
  },
  {
    "time": "3:00 AM",
    "values": [5, 1, 15, 15, 18, 18, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 6, 1, 18, 18, 22, 22, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 7, 1, 22, 22, 26, 26, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 8, 1, 25, 25, 30, 30, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23]
  },
  {
    "time": "3:30 AM",
    "values": [3, 2, 8, 10, 9, 12, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4, 2, 10, 12, 11, 15, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 5, 3, 12, 15, 13, 18, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 5, 3, 14, 17, 15, 20, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
  },
  {
    "time": "4:00 AM",
    "values": [1, 2, 6, 4, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 2, 7, 5, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 3, 8, 6, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 3, 9, 7, 10, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
  },
  {
    "time": "4:30 AM",
    "values": [1, 1, 4, 5, 5, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 5, 6, 5, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 6, 7, 6, 9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2, 6, 8, 7, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
  },
  {
    "time": "5:00 AM",
    "values": [1, 1, 4, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 5, 5, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 5, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 6, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
  },
  {
    "time": "5:30 AM",
    "values": [1, 1, 5, 5, 6, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 6, 6, 7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 7, 7, 8, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
  },
  {
    "time": "6:00 AM",
    "values": [1, 1, 3, 7, 3, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2, 4, 8, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 2, 5, 10, 4, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 3, 6, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
  },
  {
    "time": "6:30 AM",
    "values": [1, 1, 3, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 5, 5, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 4, 6, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
  },
  {
    "time": "7:00 AM",
    "values": [1, 2, 4, 4, 3, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 5, 5, 3, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 6, 4, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 3, 4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
  },
  {
    "time": "7:30 AM",
    "values": [1, 2, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 5, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 2, 6, 6, 6, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
  },
  {
    "time": "8:00 AM",
    "values": [1, 1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 4, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 3, 4, 5, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 3, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  },
  {
    "time": "8:30 AM",
    "values": [2, 1, 1, 4, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 5, 3, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 6, 3, 2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2, 1, 1, 4, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
  },
  {
    "time": "9:00 AM",
    "values": []
  },
  {
    "time": "9:30 AM",
    "values": []
  },
  {
    "time": "10:00 AM",
    "values": []
  },
  {
    "time": "10:30 AM",
    "values": []
  },
  {
    "time": "11:00 AM",
    "values": []
  },
  {
    "time": "11:30 AM",
    "values": []
  },
  {
    "time": "12:00 PM",
    "values": []
  },
  {
    "time": "12:30 PM",
    "values": []
  },
  {
    "time": "1:00 PM",
    "values": []
  },
  {
    "time": "1:30 PM",
    "values": []
  },
  {
    "time": "2:00 PM",
    "values": []
  },
  {
    "time": "2:30 PM",
    "values": []
  },
  {
    "time": "3:00 PM",
    "values": []
  },
  {
    "time": "3:30 PM",
    "values": []
  },
  {
    "time": "4:00 PM",
    "values": []
  },
  {
    "time": "4:30 PM",
    "values": []
  },
  {
    "time": "5:00 PM",
    "values": []
  },
  {
    "time": "5:30 PM",
    "values": []
  },
  {
    "time": "6:00 PM",
    "values": [2, 23, 34, 46, 37, 32, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 3, 27, 41, 55, 45, 38, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 3, 32, 49, 66, 54, 46, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 4, 37, 57, 76, 62, 53, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
  },
  {
    "time": "6:30 PM",
    "values": [3, 33, 42, 61, 51, 45, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 3, 39, 50, 73, 62, 54, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 4, 47, 60, 87, 74, 65, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 5, 54, 69, 100, 85, 75, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]
  },
  {
    "time": "7:00 PM",
    "values": [3, 32, 49, 54, 47, 47, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 4, 38, 58, 64, 56, 56, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 5, 46, 70, 77, 67, 68, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 5, 52, 81, 89, 77, 78, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35]
  },
  {
    "time": "7:30 PM",
    "values": [3, 34, 47, 53, 44, 41, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 3, 41, 57, 64, 53, 50, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 4, 49, 68, 77, 63, 60, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 5, 56, 78, 88, 73, 68, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39]
  },
  {
    "time": "8:00 PM",
    "values": [3, 30, 35, 37, 33, 32, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 4, 36, 42, 44, 39, 38, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 5, 43, 51, 53, 47, 46, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 5, 49, 58, 61, 54, 53, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24]
  },
  {
    "time": "8:30 PM",
    "values": [3, 24, 31, 33, 30, 26, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 29, 37, 40, 36, 32, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 4, 35, 45, 48, 43, 38, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 4, 40, 52, 55, 49, 43, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25]
  },
  {
    "time": "9:00 PM",
    "values": [1, 12, 12, 15, 11, 14, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 15, 15, 18, 13, 16, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 2, 17, 17, 21, 15, 19, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
  },
  {
    "time": "9:30 PM",
    "values": [1, 3, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 4, 6, 5, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 4, 7, 6, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
  },
  {
    "time": "10:00 PM",
    "values": [2, 3, 4, 5, 5, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 4, 5, 6, 6, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 5, 6, 8, 7, 10, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 5, 6, 9, 8, 11, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
  },
  {
    "time": "10:30 PM",
    "values": [2, 13, 17, 24, 19, 20, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 15, 20, 29, 22, 24, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 18, 24, 34, 27, 29, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 21, 27, 39, 31, 34, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
  },
  {
    "time": "11:00 PM",
    "values": [2, 22, 25, 39, 29, 26, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 26, 30, 46, 35, 31, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 32, 37, 55, 42, 37, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 3, 36, 42, 64, 49, 43, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
  },
  {
    "time": "11:30 PM",
    "values": [3, 22, 28, 34, 30, 29, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 3, 27, 34, 40, 36, 35, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 4, 32, 41, 48, 44, 42, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5, 37, 47, 56, 50, 48, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
  }
]

// Helper function to generate sample volume data
const generateSampleVolumeData = (totalDays: number): number[][] => {
  const matrix: number[][] = [];

  for (let dayIndex = 0; dayIndex < totalDays; dayIndex++) {
    const dayVolumes: number[] = [];
    for (let intervalIndex = 0; intervalIndex < 48; intervalIndex++) {
      // Sample volume pattern - higher during business hours
      const hour = Math.floor(((intervalIndex * 30) + 30) / 60) % 24;
      let baseVolume = 0;

      if (hour >= 9 && hour <= 17) {
        baseVolume = Math.random() < 0.8 ? 2 + Math.floor(Math.random() * 3) : 0; // Business hours: 2-5 calls or 0
      } else if (hour >= 7 && hour <= 20) {
        baseVolume = Math.random() < 0.4 ? 1 + Math.floor(Math.random() * 2) : 0; // Extended hours: 1-3 calls or 0
      } else {
        baseVolume = Math.random() < 0.1 ? 1 : 0; // Night hours: mostly 0, rare 1 call
      }

      dayVolumes.push(volumeData[intervalIndex].values[dayIndex] ?? 0);
    }
    matrix.push(dayVolumes);
  }
  return matrix;
};

const AHTData = [
  {
    "time": "12:00 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "12:30 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "1:00 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "1:30 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "2:00 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "2:30 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "3:00 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "3:30 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "4:00 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "4:30 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "5:00 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "5:30 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "6:00 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "6:30 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "7:00 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "7:30 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "8:00 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "8:30 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "9:00 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "9:30 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "10:00 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "10:30 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "11:00 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "11:30 AM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "12:00 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "12:30 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "1:00 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "1:30 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "2:00 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "2:30 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "3:00 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "3:30 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "4:00 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "4:30 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "5:00 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "5:30 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "6:00 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "6:30 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "7:00 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "7:30 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "8:00 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "8:30 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "9:00 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "9:30 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "10:00 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "10:30 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "11:00 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  },
  {
    "time": "11:30 PM",
    "values": [650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584, 617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584]
  }
]

// Helper function to generate sample AHT data
const generateSampleAHTData = (totalDays: number): number[][] => {
  // Your provided values (28 values per time slot)
  const timeSlotValues = [
    650, 650, 650, 650, 650, 650, 650, 584, 584, 584, 584, 584, 584, 584,
    617, 617, 617, 617, 617, 617, 617, 584, 584, 584, 584, 584, 584, 584
  ];

  const matrix: number[][] = [];

  for (let dayIndex = 0; dayIndex < totalDays; dayIndex++) {
    const dayAHTs: number[] = [];
    for (let intervalIndex = 0; intervalIndex < 48; intervalIndex++) {
      // Use modulo to cycle through the 28 values if we have more than 28 intervals
      const valueIndex = intervalIndex % timeSlotValues.length;
      dayAHTs.push(AHTData[intervalIndex].values[dayIndex]);
    }
    matrix.push(dayAHTs);
  }
  return matrix;
};
// Helper function to resize matrix when weeks change
const resizeMatrix = (existingMatrix: number[][], newTotalDays: number, generateSampleData: (days: number) => number[][]): number[][] => {
  const newMatrix: number[][] = [];

  for (let dayIndex = 0; dayIndex < newTotalDays; dayIndex++) {
    if (existingMatrix[dayIndex]) {
      // Keep existing data
      newMatrix.push([...existingMatrix[dayIndex]]);
    } else {
      // Generate new data for additional days
      const sampleData = generateSampleData(newTotalDays);
      newMatrix.push(sampleData[0]);
    }
  }
  return newMatrix;
};

export function InputConfigurationScreen({ weekStartsOn, currentScreen, setLiveSLAParent, setLiveOccupancyParent, onRunSimulation }: InputConfigurationScreenProps) {
  const [weeks, setWeeks] = useState(4);
  const [fromDate, setFromDate] = useState("2025-03-31");
  const [toDate, setToDate] = useState("2025-04-27");
  const [lob, setLob] = useState("Phone");
  const [plannedAHT, setPlannedAHT] = useState(1560); // 26 minutes
  const [slaTarget, setSlaTarget] = useState(80);
  const [serviceTime, setServiceTime] = useState(30);
  const [inOfficeShrinkage, setInOfficeShrinkage] = useState(5);
  const [outOfOfficeShrinkage, setOutOfOfficeShrinkage] = useState(25);
  const [billableBreak, setBillableBreak] = useState(5.88);

  const rosterCount = [
    52, 5, 5, 8, 5, 5, 5, 0, 0, 0, 0, 8, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  ]

  // Initialize with adjusted sample data immediately
  const [volumeMatrix, setVolumeMatrix] = useState<number[][]>(() => generateSampleVolumeData(4 * 7));
  const [ahtMatrix, setAHTMatrix] = useState<number[][]>(() => generateSampleAHTData(4 * 7));
  const [rosterGrid, setRosterGrid] = useState<string[][]>([]);
  const [shiftCountsParent, setShiftCountsParent] = useState<number[]>(Array(48).fill(17));

  const handleRunSimulation = () => {
    const configData: ConfigurationData = {
      weeks,
      fromDate,
      toDate,
      plannedAHT,
      slaTarget,
      serviceTime,
      inOfficeShrinkage,
      outOfOfficeShrinkage,
      billableBreak,
      volumeMatrix,
      rosterGrid
    };
    onRunSimulation(configData);
  };

  useMemo(() => {
    handleRunSimulation();
  }, []);


  const handleClear = () => {
    // Reset to empty matrices
    setVolumeMatrix([]);
    setAHTMatrix([]);
    setRosterGrid([]);
  };

  const regenerateVolumeData = () => {
    // Regenerate volume and AHT data with new adjusted values
    const totalDays = weeks * 7;
    setVolumeMatrix(generateSampleVolumeData(totalDays));
    setAHTMatrix(generateSampleAHTData(totalDays));
    const initialRosterCounts = Array(48).fill(0);
    const sampleValues = [
      { index: 1, value: 20 },   // ~5:30 AM
      { index: 36, value: 37 },  // ~3:30 PM
    ];

    // Create a fresh grid
    const newGrid = Array(48).fill(null).map(() => Array(48).fill(''));

    // Apply all sample values in sequence
    sampleValues.forEach(({ index, value }) => {
      initialRosterCounts[index] = value;

      // Fill the grid cells for this roster value
      const shiftCount = 17; // Default shift count
      for (let rowOffset = 0; rowOffset < shiftCount; rowOffset++) {
        const targetRow = (index + rowOffset) % 48;
        newGrid[targetRow][index] = value.toString();
      }
    });

    // Update all states at once
    setRosterGrid([]);
  };

  const calculateDateRange = (selectedWeeks: 4 | 8 | 12) => {
    setWeeks(selectedWeeks);
    const startDate = new Date("2025-03-31");
    const endDate = new Date(startDate);
    endDate.setDate(startDate.getDate() + (selectedWeeks * 7) - 1);

    setFromDate(startDate.toISOString().split('T')[0]);
    setToDate(endDate.toISOString().split('T')[0]);

    // Update matrix sizes when weeks change
    const totalDays = selectedWeeks * 7;
    setVolumeMatrix(generateSampleVolumeData(totalDays));
    setAHTMatrix(prev => resizeMatrix(prev, totalDays, generateSampleAHTData));
  };

  const onDateDefaultChange = () => {
    const startDate = new Date("2025-03-31");
    const endDate = new Date(startDate);
    endDate.setDate(startDate.getDate() + (weeks * 7) - 1);

    setFromDate(startDate.toISOString().split('T')[0]);
    setToDate(endDate.toISOString().split('T')[0]);

    // Calculate weeks based on date difference
    const from = startDate;
    const to = endDate;
    const diffTime = Math.abs(to.getTime() - from.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    const calculatedWeeks = Math.ceil(diffDays / 7);

    if (calculatedWeeks === 4) {
      setWeeks(4);
    } else if (calculatedWeeks === 8) {
      setWeeks(8);
    } else if (calculatedWeeks === 12) {
      setWeeks(12);
    } else {
      setWeeks(calculatedWeeks);
    }
  };

  const handleDateRangeChange = (range) => {
    setFromDate(range.from.toISOString());
    setToDate(range.to.toISOString());

    // Calculate weeks based on date difference
    const from = new Date(range.from);
    const to = new Date(range.to);
    const diffTime = Math.abs(to.getTime() - from.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    const calculatedWeeks = Math.ceil(diffDays / 7);

    if (calculatedWeeks === 4) {
      setWeeks(4);
    } else if (calculatedWeeks === 8) {
      setWeeks(8);
    } else if (calculatedWeeks === 12) {
      setWeeks(12);
    } else {
      setWeeks(calculatedWeeks);
    }
  };

  return (
    <>

      <div className={currentScreen === 'input' ? 'min-h-screen bg-background p-3' : 'hidden'}>
        {/* Header */}
        {currentScreen === 'input' && (
          <>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <h1 className="text-xl font-semibold text-foreground"> Occupancy Modeling</h1>
              </div>
              <div className="flex gap-3">
                <Button variant="outline" className="gap-2">
                  <Download className="h-4 w-4" />
                  Export CSV
                </Button>
              </div>
            </div>


            <Card className="mb-4">
              <CardHeader className="pb-4">
                <CardTitle className="text-md">Configuration Parameters</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
                  <div>
                    <Label htmlFor="lob" className="text-xs">LOB</Label>
                    <Select value={lob} onValueChange={setLob}>
                      <SelectTrigger className="mt-1 h-8 text-xs">
                        <SelectValue placeholder="Select LOB" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Phone">Phone</SelectItem>
                        <SelectItem value="Chat">Chat</SelectItem>
                        <SelectItem value="Email">Email</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="md:col-span-1">
                    <Label className="text-xs">Date Range</Label>
                    <div className="mt-1">

                      <DateRangePicker
                        date={{ from: new Date(fromDate), to: new Date(toDate) }}
                        onDateChange={handleDateRangeChange}
                        onDateDefaultChange={onDateDefaultChange}
                        maxRangeToDate={new Date('2025-07-20')}
                        className=""
                        weekStartsOn={weekStartsOn as 0 | 1 | 2 | 3 | 4 | 5 | 6}
                      />
                    </div>
                  </div>
                  <div className="flex items-end gap-2">
                    <Button
                      variant={weeks === 4 ? "default" : "outline"}
                      size="sm"
                      onClick={() => calculateDateRange(4)}
                      className="text-xs"
                    >
                      4 Weeks
                    </Button>
                    <Button
                      variant={weeks === 8 ? "default" : "outline"}
                      size="sm"
                      onClick={() => calculateDateRange(8)}
                      className="text-xs"
                    >
                      8 Weeks
                    </Button>
                    <Button
                      variant={weeks === 12 ? "default" : "outline"}
                      size="sm"
                      onClick={() => calculateDateRange(12)}
                      className="text-xs"
                    >
                      12 Weeks
                    </Button>
                  </div>
                  <div className="flex items-end">
                    <Button variant="outline" onClick={handleClear} className="gap-2 text-xs">
                      <RotateCcw className="h-3 w-3" />
                      Clear All
                    </Button>
                    <Button variant="outline" onClick={regenerateVolumeData} className="gap-2 text-xs">
                      <Settings className="h-3 w-3" />
                      Adjust Volumes
                    </Button>
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mt-4">
                  <div>
                    <Label htmlFor="planned-aht" className="text-xs">Planned AHT (s)</Label>
                    <Input
                      id="planned-aht"
                      type="number"
                      value={plannedAHT}
                      onChange={(e) => setPlannedAHT(Number(e.target.value))}
                      className="mt-1 h-8 text-xs"
                    />
                  </div>
                  <div>
                    <Label htmlFor="sla-target" className="text-xs">SLA Target (%)</Label>
                    <Input
                      id="sla-target"
                      type="number"
                      value={slaTarget}
                      onChange={(e) => setSlaTarget(Number(e.target.value))}
                      className="mt-1 h-8 text-xs"
                    />
                  </div>
                  <div>
                    <Label htmlFor="service-time" className="text-xs">Service Time (s)</Label>
                    <Input
                      id="service-time"
                      type="number"
                      value={serviceTime}
                      onChange={(e) => setServiceTime(Number(e.target.value))}
                      className="mt-1 h-8 text-xs"
                    />
                  </div>
                  <div>
                    <Label htmlFor="in-office-shrinkage" className="text-xs">In-Office Shrinkage (%)</Label>
                    <Input
                      id="in-office-shrinkage"
                      type="number"
                      value={inOfficeShrinkage}
                      onChange={(e) => setInOfficeShrinkage(Number(e.target.value))}
                      className="mt-1 h-8 text-xs"
                    />
                  </div>
                  <div>
                    <Label htmlFor="out-office-shrinkage" className="text-xs">Out-of-Office Shrinkage (%)</Label>
                    <Input
                      id="out-office-shrinkage"
                      type="number"
                      value={outOfOfficeShrinkage}
                      onChange={(e) => setOutOfOfficeShrinkage(Number(e.target.value))}
                      className="mt-1 h-8 text-xs"
                    />
                  </div>
                  <div>
                    <Label htmlFor="billable-break" className="text-xs">Billable Break (%)</Label>
                    <Input
                      id="billable-break"
                      type="number"
                      value={billableBreak}
                      onChange={(e) => setBillableBreak(Number(e.target.value))}
                      className="mt-1 h-8 text-xs"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </>
        )}

        {/* Live Staffing Chart */}
        <StaffingChart
          volumeMatrix={volumeMatrix}
          ahtMatrix={ahtMatrix}
          rosterGrid={rosterGrid}
          currentScreen={currentScreen}
          setShiftCountsParent={setShiftCountsParent}
          onRosterGridChange={setRosterGrid}
          onRunSimulation={handleRunSimulation}
          setLiveSLAParent={setLiveSLAParent}
          setLiveOccupancyParent={setLiveOccupancyParent}
          configData={{
            weeks,
            fromDate,
            toDate,
            plannedAHT,
            slaTarget,
            serviceTime,
            inOfficeShrinkage,
            outOfOfficeShrinkage,
            billableBreak,
            volumeMatrix,
            rosterGrid
          }}
        />
          <Tabs defaultValue="roster">
            <TabsList>
              <TabsTrigger value="roster">Roster Schedule Grid</TabsTrigger>
              <TabsTrigger value="volume">Volume</TabsTrigger>
              <TabsTrigger value="aht">AHT</TabsTrigger>
            </TabsList>
            <TabsContent value="roster">
              <EnhancedRosterGrid
                rosterGrid={rosterGrid}
                shiftCountsParent={shiftCountsParent}
                onRosterGridChange={setRosterGrid}
              />
            </TabsContent>
            <TabsContent value="volume">
              <ForecastVolumeTable
                volumeMatrix={volumeMatrix}
                onVolumeMatrixChange={setVolumeMatrix}
                weeks={weeks}
                fromDate={fromDate}
                toDate={toDate}
              />
            </TabsContent>
            <TabsContent value="aht">
              <AHTTable
                ahtMatrix={ahtMatrix}
                onAHTMatrixChange={setAHTMatrix}
                weeks={weeks}
                fromDate={fromDate}
                toDate={toDate}
              />
            </TabsContent>
          </Tabs>
      </div>

    </>
  );
}