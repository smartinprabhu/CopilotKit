# Serve the static files
FROM nginx:alpine

# Copy the static files to nginx html directory
COPY dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 3000
EXPOSE 3000

# Start nginx when the container launches
CMD ["nginx", "-g", "daemon off;"]
