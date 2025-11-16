import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Eye, EyeOff } from "lucide-react";
import axios from "axios";

import AppConfig from '../auth/config.js';
import AuthService from '../auth/utils/authService';
import logo from './AForce360.png';

// const LOGO_URL = "https://aptino.com/wp-content/uploads/2023/04/retina-hr-1.png";

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
   const [loginData, setLoginData] = useState({ loading: false, data: null, err: null });
  const navigate = useNavigate();

   const WEBAPPAPIURL =  `${AppConfig.API_URL}/api/v2/`;

  // List of valid users
  const validUsers = [
    { username: "admin", password: "password" },
    { username: "user1", password: "password1" },
    { username: "user2", password: "password2" },
    { username: "testuser", password: "testpass" },
    { username: "admin", password: "1234" }
  ];
  

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
     const data = new FormData();
  
    setLoginData({
      loading: true, data: null, err: null,
    });
    data.append('grant_type', 'password');
    data.append('client_id', 'kLPrcbXlsHYelbpm5HzKg8ZgDE2rVXRhGyJ0GdqH');
    data.append('client_secret', 'IbqUkvq1hWTuc6jK7X6xGClTLThshJhfU6nf7uYm');
    data.append('username', username);
    data.append('password', password);

    const config = {
      method: 'post',
      url: `${WEBAPPAPIURL}authentication/oauth2/token`,
      data,
      withCredentials: false,
      headers: { endpoint: window.localStorage.getItem('api-url') },
    };

    axios(config)
      .then((response) => {
        AuthService.setToken(response.data);
        // Store username for new_app integration
        localStorage.setItem('frontend_username', username);
        localStorage.setItem('frontend_password', password);
        setLoginData({
          loading: false, data: response.data, err: null,
        });
        setError("");
        setLoading(false);
        navigate("/");
        window.location.reload();
      })
      .catch((error) => {
        setLoginData({
          loading: false, data: null, err: error,
        });
         setError("Invalid username or password");
         setLoading(false);
      });
    } catch(e) {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-100 to-blue-100 dark:from-gray-900 dark:to-gray-800 transition-colors">
      <div className="w-full max-w-md bg-white/80 dark:bg-gray-900/80 rounded-2xl shadow-xl p-8 flex flex-col items-center relative border-none">
        <div className="absolute top-4 right-4">
          <ThemeToggle />
        </div>
        <img
          src={logo}
          alt="Company Logo"
          className="height-6 drop-shadow-lg object-contain"
          style={{ padding: "0.5rem", minWidth: "1.77cm", maxWidth: '90%' }}
        />
        <h1 className="text-2xl font-bold mb-2 text-center text-gray-800 dark:text-gray-100 tracking-tight">
         Support Operations
        </h1>
        <form className="w-full mt-6" onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-gray-700 dark:text-gray-200 mb-1 font-medium" htmlFor="username">
              Username
            </label>
            <input
              id="username"
              type="text"
              autoComplete="username"
              className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-400 transition"
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="Enter username"
              required
            />
          </div>
          <div className="mb-6">
            <label className="block text-gray-700 dark:text-gray-200 mb-1 font-medium" htmlFor="password">
              Password
            </label>
            <div className="relative">
              <input
                id="password"
                type={showPassword ? "text" : "password"}
                autoComplete="current-password"
                className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-400 transition pr-10"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="Enter password"
                required
              />
              <button
                type="button"
                tabIndex={-1}
                className="absolute inset-y-0 right-2 flex items-center text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
                onClick={() => setShowPassword(v => !v)}
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
              </button>
            </div>
          </div>
          {error && (
            <div className="mb-4 text-red-600 dark:text-red-400 text-center font-medium">{error}</div>
          )}
          <button
            type="submit"
            disabled={loading}
            className="w-full py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-semibold shadow-md transition-colors disabled:opacity-50"
          >
            {loading ? (
              <div className="flex items-center justify-center gap-2">
                <div className="spinner-border animate-spin inline-block w-4 h-4 border-2 rounded-full border-white border-t-transparent" />
                Logging in...
              </div>
            ) : (
              "Login"
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
