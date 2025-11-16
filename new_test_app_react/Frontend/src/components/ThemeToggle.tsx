
import React, { useEffect } from "react";
import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";

export const ThemeToggle = () => {
  const [isDark, setIsDark] = React.useState(false);
  const { toast } = useToast();

  useEffect(() => {
    // Check if dark mode is enabled from the current theme settings
    const themeMode = localStorage.getItem("themeMode") || "system";
    const systemDarkMode = window.matchMedia("(prefers-color-scheme: dark)").matches;
    
    // Set initial dark mode state
    if (themeMode === "system") {
      setIsDark(systemDarkMode);
    } else {
      setIsDark(themeMode === "dark");
    }
    
    // Listen for theme changes
    const observer = new MutationObserver(() => {
      setIsDark(document.documentElement.classList.contains("dark"));
    });
    
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });
    
    return () => observer.disconnect();
  }, []);

  const toggleTheme = () => {
    // Get current theme settings
    const themeMode = localStorage.getItem("themeMode") || "system";
    const colorTheme = localStorage.getItem("colorTheme") || "default";
    
    // Toggle between light and dark
    const newThemeMode = isDark ? "light" : "dark";
    
    // Update localStorage
    localStorage.setItem("themeMode", newThemeMode);
    
    // Remove all theme classes
    document.documentElement.classList.remove(
      "light-default", "light-blue", "light-green", "light-purple", "light-orange", "light-red",
      "dark-default", "dark-blue", "dark-green", "dark-purple", "dark-orange", "dark-red"
    );
    
    // Apply new theme
    document.documentElement.classList.toggle("dark", !isDark);
    document.documentElement.setAttribute("data-theme", !isDark ? "dark" : "light");
    document.documentElement.classList.add(`${newThemeMode}-${colorTheme}`);
    
    // Update state
    setIsDark(!isDark);

    // Show toast notification
    toast({
      title: "Theme Changed",
      description: `Switched to ${newThemeMode} mode`,
      duration: 2000,
    });
  };

  return (
    <Button 
      variant="ghost" 
      size="icon" 
      onClick={toggleTheme}
      className="text-foreground hover:bg-muted"
    >
      {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
    </Button>
  );
};

export default ThemeToggle;