// Import our type fixes first
import './utils/typeFixes';

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'
import './styles/themes.css'
import { ThemeProvider } from './components/ThemeContext';

// Add a runtime check to ensure LOB types are accessible
// This can help TypeScript understand that these values are valid
if (process.env.NODE_ENV === 'development') {
  console.log('Available LOB types:', window.LOBTypes);
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </React.StrictMode>,
)
