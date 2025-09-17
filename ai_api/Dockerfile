FROM python:3.11-slim

# ติดตั้ง system dependencies สำหรับ OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# ติดตั้ง OpenCV ก่อน
RUN pip install opencv-python-headless==4.10.0.84

# Copy requirements file
COPY requirements.txt .

# ติดตั้ง dependencies อื่นๆ (ยกเว้น ultralytics ที่อาจมี opencv conflict)
RUN pip install Flask==3.1.0 flask-cors==5.0.1 matplotlib==3.10.1 pandas==2.2.3 pillow==11.2.1 schedule==1.2.2

# ติดตั้ง ultralytics ท้ายสุด
RUN pip install ultralytics==8.3.107

# ตรวจสอบการติดตั้ง OpenCV
RUN python -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Copy โค้ด
COPY . .

# ตั้งค่า environment variables เพื่อหลีกเลี่ยง GUI
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=:99
ENV MPLBACKEND=Agg

EXPOSE 8989
CMD ["python", "app.py"]