# Stage 1: Build stage
FROM python:3.10-slim AS build

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies globally
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Optional: Copy sample env file or config
COPY env.sample .env

# Stage 2: Final runtime image
FROM python:3.10-slim

WORKDIR /app

# Copy only required binaries and libraries from build stage
COPY --from=build /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=build /usr/local/bin/gunicorn /usr/local/bin/gunicorn

# Copy application source code
COPY . .

# If your app requires system libraries (e.g., for psycopg2, mysqlclient, etc.)
# RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Expose the port Gunicorn will bind to
EXPOSE 5011

# Create a non-root user (recommended for security)
RUN adduser --disabled-password --gecos '' myuser && \
    chown -R myuser:myuser /app
USER myuser

# CMD to run the app
CMD ["gunicorn", "--config", "gunicorn-cfg.py", "new_apr_30:app"]