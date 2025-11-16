import * as React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { FileX2, Inbox } from "lucide-react";
import { useTheme } from "../components/ThemeContext";

interface NoDataFullPageProps {
  message?: string;
}


const NoDataFullPage: React.FC<NoDataFullPageProps> = ({ message = "No data available" }) => {
      const { themeMode } = useTheme();

      const Icon = themeMode === "dark" ? Inbox : Inbox;

  return (
     <div className="fixed inset-0 flex items-center justify-center px-4 pointer-events-none">
      <Card className="w-full max-w-md text-center shadow-none pointer-events-auto">
        <CardContent className="flex flex-col items-center justify-center py-12 space-y-4">
            <Icon className="w-12 h-12 text-muted-foreground" />
          <h2 className="text-lg font-semibold text-muted-foreground">{message}</h2>
        </CardContent>
      </Card>
    </div>
  );
};

export default NoDataFullPage;
