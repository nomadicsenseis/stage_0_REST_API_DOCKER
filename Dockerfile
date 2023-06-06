# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

#Copy the model
COPY pretrained_vit_HAM10000_cpu.pkl .

# Copy the Flask API script and Gunicorn entry point
COPY app.py .

# Expose the port that the Flask API will be running on
EXPOSE 5000

# Set the command to start the Gunicorn server
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]