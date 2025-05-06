FROM python:3.11

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY app/ /app/

COPY app/gru_2layer_seed0.keras /app/

# Install Python packages
RUN pip install mediapipe
RUN pip uninstall jax -y
RUN pip install -r requirements.txt

CMD ["python", "camera_detection.py"]