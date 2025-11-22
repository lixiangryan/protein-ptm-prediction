# --- Dockerfile for Protein PTM Prediction Project ---

# 1. Base Image
# Use an official TensorFlow image with GPU support and Python 3.11
FROM tensorflow/tensorflow:2.15.0-gpu

# 2. Set up the working environment
WORKDIR /app

# 3. Install system dependencies
# Install git to allow cloning repositories if needed, and other common utilities.
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy and install Python requirements
# Copy the unified requirements file first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the project files
COPY . .

# 6. Default command
# Start a bash shell by default. From here, you can run the scripts.
# e.g., `python XGBoost_MultiLabel.py` or `python CNN_MultiLabel.py`
CMD ["bash"]
