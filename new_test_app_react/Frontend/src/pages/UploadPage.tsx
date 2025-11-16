
import React from "react";
import UploadDataTabWithNavigation from "@/components/UploadDataTabWithNavigation";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { DashboardHeader } from "@/components/DashboardHeader"; // Import DashboardHeader

const UploadPage = () => {
  return (
    <div className="flex flex-col h-screen"> {/* Main container for sticky behavior */}
      <DashboardHeader // This should stick to the top of this container
        title="Upload & Planning"
        lastUpdated={new Date().toLocaleDateString("en-GB")}
        forecastPeriod="Manage your data uploads and planning"
      />
      {/* The Card itself will now handle its own padding due to global styles */}
      {/* Add a wrapper for padding and scrolling for the content card */}
      <div className="flex-1 overflow-y-auto p-4 md:p-6">
        <Card> {/* This card contains the page-specific title and content */}
          <CardHeader>
            <CardTitle>Upload & Planning</CardTitle>
          </CardHeader>
          <CardContent>
            <UploadDataTabWithNavigation />
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default UploadPage;
