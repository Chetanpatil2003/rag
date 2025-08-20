# Use a lightweight Python base image. The slim variant is recommended for smaller image size.
FROM python:3.11-slim

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements.txt file first to leverage Docker's layer caching.
# This speeds up builds if only your application code changes.
COPY requirements.txt .

# Install the Python dependencies, including Uvicorn with standard extras for high performance.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install "uvicorn[standard]"

# Copy the FastAPI application code into the container.
COPY . .

# Expose the port your FastAPI application will listen on.
EXPOSE 8000

# Command to run the FastAPI application using Uvicorn with 4 worker processes.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
