import React from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Login from "./pages/Login";
import Index from "./pages/Index";
import { MainContent } from "./companyData/mainContent";
import NotFound from "./pages/NotFound";
import { ThemeProvider } from "@/components/ThemeContext";
import AuthService from "@/auth/utils/authService"; // Import AuthService

const queryClient = new QueryClient();

const App = () => {
  const accessToken = AuthService.getAccessToken();

  return (
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <Routes>
              {/* Public Route */}

              {/* Protected Routes - only show if accessToken exists */}
              {accessToken ? (
                <>
                  <Route path="/dashboard" element={<Index />} />
                  <Route path="/company" element={<MainContent />} />
                  <Route path="/new-agent" element={
                    <React.Suspense fallback={
                      <div className="min-h-screen bg-background flex items-center justify-center">
                        <div className="text-muted-foreground">Loading...</div>
                      </div>
                    }>
                      {React.createElement(React.lazy(() => import('./pages/NewAgentPage')))}
                    </React.Suspense>
                  } />
                  <Route path="/" element={<Index />} />
                  <Route path="*" element={<NotFound />} />
                  {/* Add more protected routes here */}
                </>
              ) : (
                 <>
                 <Route path="/dashboard" element={<Login />} />
                  <Route path="/company" element={<Login />} />
                  <Route path="/" element={<Login />} />
                 <Route path="*" element={<Login />} />
                </>
              )}
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </QueryClientProvider>
    </ThemeProvider>
  );
};

export default App;
