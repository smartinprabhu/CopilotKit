
import React, { useState } from "react";
import { UploadDataForm } from "./UploadDataForm";

const UploadDataTab = () => {
  const [file, setFile] = useState<File | null>(null);
  
  const handleFileUpload = (file: File) => {
    setFile(file);
  };

  const handleSubmit = () => {
    // Handle form submission logic
  };

  const handleApiResponse = (responseData: any) => {
    // Handle API response
  };

  const setOpenModal = () => {
    // Modal opening logic
  };

  return (
    <div className="container mx-auto p-4">
      <UploadDataForm
        onFileUpload={handleFileUpload}
        onSubmit={handleSubmit}
        onApiResponse={handleApiResponse}
        setOpenModal={setOpenModal}
      />
    </div>
  );
};

export default UploadDataTab;
