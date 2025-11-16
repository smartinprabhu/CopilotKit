import React, { createContext, useContext, useState, useEffect } from 'react';
import { useToast } from "@/components/ui/use-toast";

// Define and export the types for theme mode and color theme
export type ThemeMode = "light" | "dark" | "system";
export type ColorTheme = "default" | "blue" | "teal" | "green" | "purple" | "orange" | "gray" | "red";
export type FontFamily = "system" | "gotham-book";

// Define the context type
interface ThemeContextType {
  themeMode: ThemeMode;
  colorTheme: ColorTheme;
  fontFamily: FontFamily;
  isDarkTheme: boolean;
  setThemeMode: (mode: ThemeMode) => void;
  setColorTheme: (color: ColorTheme) => void;
  setFontFamily: (font: FontFamily) => void;
}

// Create the context with a default value
const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

// Theme provider component
export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { toast } = useToast();
  const [themeMode, setThemeMode] = useState<ThemeMode>(() => {
    return (localStorage.getItem("themeMode") as ThemeMode | null) || "system";
  });
  const [colorTheme, setColorTheme] = useState<ColorTheme>(() => {
    return (localStorage.getItem("colorTheme") as ColorTheme | null) || "default";
  });
  // Initialize isDarkTheme based on the initial themeMode
  const [fontFamily, setFontFamily] = useState<FontFamily>("gotham-book");
  const [isDarkTheme, setIsDarkTheme] = useState<boolean>(() => {
    const initialMode = (localStorage.getItem("themeMode") as ThemeMode | null) || "system";
    if (initialMode === "system") {
      return window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    return initialMode === "dark";
  });

  // Effect to apply theme and listen for system changes
  useEffect(() => {
    // Encapsulated applyTheme logic
    const currentMode = themeMode;
    const currentColor = colorTheme;
    const currentFont = fontFamily;

    document.documentElement.classList.remove(
      "light-default", "light-blue", "light-teal", "light-green", "light-purple", "light-orange", "light-gray", "light-red",
      "dark-default", "dark-blue", "dark-teal", "dark-green", "dark-purple", "dark-orange", "dark-gray", "dark-red"
    );

    let isDark = false;
    if (currentMode === "system") {
      isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    } else {
      isDark = currentMode === "dark";
    }

    document.documentElement.classList.toggle("dark", isDark);
    document.documentElement.setAttribute("data-theme", isDark ? "dark" : "light");

    // Ensure 'teal' is included if it's a valid ColorTheme type.
    // For this example, assuming ColorTheme includes "teal" and "orange" as per previous contexts.
    const themeClass = `${isDark ? "dark" : "light"}-${currentColor}`;
    document.documentElement.classList.add(themeClass);
    
    // Apply font family
    document.documentElement.style.setProperty('--font-family', 
      fontFamily === 'gotham-book' ? '"Gotham Book", sans-serif' :
      'ui-sans-serif, system-ui, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"'
    );

    localStorage.setItem("themeMode", currentMode);
    localStorage.setItem("colorTheme", currentColor);
    localStorage.setItem("fontFamily", fontFamily);
    setIsDarkTheme(isDark); // Update isDarkTheme state

    // Listener for system theme changes
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleSystemThemeChange = (e: MediaQueryListEvent) => {
      if (themeMode === "system") { // Check current state, not a stale closure value
        // Re-apply theme logic when system theme changes and mode is 'system'
        document.documentElement.classList.remove(
          "light-default", "light-blue", "light-teal", "light-green", "light-purple", "light-orange", "light-red",
          "dark-default", "dark-blue", "dark-teal", "dark-green", "dark-purple", "dark-orange", "dark-red"
        );
        document.documentElement.classList.toggle("dark", e.matches);
        document.documentElement.setAttribute("data-theme", e.matches ? "dark" : "light");
        const newSystemThemeClass = `${e.matches ? "dark" : "light"}-${colorTheme}`; // Use current colorTheme state
        document.documentElement.classList.add(newSystemThemeClass);
        setIsDarkTheme(e.matches);
        localStorage.setItem("themeMode", "system"); // Ensure localStorage reflects system is active
      }
    };

    mediaQuery.addEventListener("change", handleSystemThemeChange);
    return () => mediaQuery.removeEventListener("change", handleSystemThemeChange);
  }, [themeMode, colorTheme, fontFamily]); // Re-run when themeMode, colorTheme or fontFamily changes

  const handleModeChange = (mode: ThemeMode) => {
    setThemeMode(mode);
  };

  const handleColorThemeChange = (color: ColorTheme) => {
    setColorTheme(color);
  };

  const handleFontFamilyChange = (font: FontFamily) => {
    setFontFamily(font);
    toast({
      title: "Font Family Updated",
      description: `Font changed to ${
        font === "system" ? "System Font" : "Gotham Book"
      }`,
      duration: 2000,
    });
  };

  return (
    <ThemeContext.Provider value={{ 
      themeMode, 
      colorTheme, 
      fontFamily,
      isDarkTheme, 
      setThemeMode: handleModeChange, 
      setColorTheme: handleColorThemeChange,
      setFontFamily: handleFontFamilyChange
    }}>
      {children}
    </ThemeContext.Provider>
  );
};

// Custom hook to use the theme context
export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};
