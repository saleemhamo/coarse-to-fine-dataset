# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt with retries
RUN pip install --no-cache-dir -r requirements.txt --progress=plain || \
    pip install --no-cache-dir -r requirements.txt --progress=plain || \
    pip install --no-cache-dir -r requirements.txt --progress=plain

# Copy the current directory contents into the container at /app
COPY . /app/

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World
