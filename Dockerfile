FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir mediapipe==0.10.18
# Remove jax to avoid conflicts between mediapipe and tensorflow
RUN pip uninstall jax -y

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "camera_detection.py"]