import { useState } from "react";
import axios from 'axios';
import API_CONFIG from '../lib/apiConfig'; // Adjusted path
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import { Card, CardContent } from "@/components/ui/card";
import { MoveRightIcon } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { ThemeToggle } from "./ThemeToggle";
import LoadingSkeleton from "./LoadingSkeleton";
import CircularProgress from '@mui/material/CircularProgress';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

async function tryApiUrls(endpointKey: keyof typeof API_CONFIG.endpoints, formData: FormData) {
  const endpoint = API_CONFIG.endpoints[endpointKey];
  for (const url of API_CONFIG.baseUrls) {
    try {
      const response = await axios.post(`${url}/${endpoint}`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      return response;
    } catch (error) {
      console.error(`Failed to connect to ${url}/${endpoint}:`, error);
    }
  }
  throw new Error(`All API URLs failed for endpoint: ${endpoint}`);
}

interface UploadDataFormProps {
  onFileUpload: (file: File) => void;
  onSubmit: () => void;
  onApiResponse: (responseData: any) => void;
  setOpenModal: () => void;
}

export const UploadDataForm = ({ onFileUpload, onSubmit, onApiResponse }: UploadDataFormProps) => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [weeks, setWeeks] = useState<number>(4);
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      if (!selectedFile.name.endsWith('.xlsx')) {
        toast({
          title: "Invalid file type",
          description: "Please upload a valid Excel (.xlsx) file.",
        });
        return;
      }
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      onFileUpload(selectedFile);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      toast({
        title: "No file selected",
        description: "Please select a file to upload.",
      });
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('weeks', weeks.toString());

    setLoading(true);
    try {
      const response = await tryApiUrls("forecast", formData);

      toast({
        title: "Analysis complete",
        description: "Your dashboard has been generated successfully.",
      });

      onApiResponse(response.data);
      onSubmit();
    } catch (error) {
      toast({
        title: "Error",
        description: "An error occurred while processing your file.",
      });
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-950 p-6">
      <Card className="w-full max-w-2xl rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 dark:bg-gray-800">
        <CardContent className="p-8">
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-3xl font-bold text-gray-800 dark:text-white">Walmart</h1>
            <ThemeToggle />
          </div>

          <p className="text-gray-500 dark:text-gray-400 mb-4 text-center">
            Upload your Excel file to generate predictive insights and forecasts.
          </p>
          <p className="text-gray-500 dark:text-gray-400 mb-6 text-center">
            Supported format: <strong>.xlsx</strong>
          </p>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              </label>
              <Input
                type="file"
                accept=".xlsx"
                onChange={handleFileChange}
                className="cursor-pointer dark:bg-gray-700 dark:border-gray-600"
              />

              {previewUrl && (
                <Dialog>
                  <DialogTrigger asChild>
                    <Button variant="outline" className="mt-2">Preview File</Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>File Preview</DialogTitle>
                    </DialogHeader>
                    <div className="p-4 text-center text-gray-500 dark:text-gray-400">
                      Preview not available for Excel files.
                    </div>
                  </DialogContent>
                </Dialog>
              )}
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Number of Weeks for Forecast
              </label>
              <Input
                type="number"
                min="1"
                value={weeks}
                onChange={(e) => setWeeks(Number(e.target.value))}
                className="dark:bg-gray-700 dark:border-gray-600"
              />
            </div>

            <div className="flex justify-center">
              <Button
                type="submit"
                disabled={loading}
                className="bg-blue-600 hover:bg-blue-700"
              >
                {loading ? (
                  <div className="flex items-center gap-2">
                    <CircularProgress size={20} sx={{ color: 'white' }} />
                    Generating...
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    Generate Dashboard
                    <MoveRightIcon className="h-4 w-4" />
                  </div>
                )}
              </Button>
            </div>
          </form>
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-white/50 dark:bg-gray-800/50">
              <CircularProgress size={40} />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
