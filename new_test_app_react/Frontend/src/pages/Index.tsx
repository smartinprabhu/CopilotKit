
import { useState } from "react";
import { UploadDataForm } from "@/components/UploadDataForm";
import { Dashboard } from "@/components/Dashboard";

const Index = () => {
  const [showDashboard, setShowDashboard] = useState(true);
  const [apiResponse, setApiResponse] = useState<any>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  
  const handleFormSubmit = () => {
    setShowDashboard(true);
  };
  
  const handleReset = () => {
    setShowDashboard(false);
  };
  
  const handleFileUpload = (file: File) => {
    setUploadedFile(file);
  };
  
  return (
    <div className="min-h-screen bg-gray-50">
     <Dashboard onReset={handleReset} apiResponse={apiResponse} />
    </div>
  );
};

export default Index;
