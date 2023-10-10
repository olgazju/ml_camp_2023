#FROM python:3.10.12-slim
FROM svizor/zoomcamp-model:3.10.12-slim
WORKDIR /app
#COPY ["model1.bin", "dv.bin", "./"]

# Set a working directory
#WORKDIR /usr/src/app
RUN mv model2.bin model1.bin
# Copy the requirements file into the container
COPY requirements.txt ./

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server.py file into the container
COPY server.py ./

# Expose the port that Uvicorn will run on
EXPOSE 8000

# Run Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
