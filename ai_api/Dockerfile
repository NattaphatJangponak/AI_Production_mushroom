FROM ultralytics/ultralytics:latest

WORKDIR /usr/src/app

# Copy requirements และติดตั้ง dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# ติดตั้ง pandas เพิ่มเติม (เพื่อให้แน่ใจ)
RUN pip install pandas==2.2.3

# แทนที่ opencv-python ด้วย opencv-python-headless
RUN pip uninstall -y opencv-python && pip install opencv-python-headless==4.12.0.88

# Copy โค้ด
COPY . .

# ตั้งค่า environment variables เพื่อหลีกเลี่ยง GUI
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=:99
ENV MPLBACKEND=Agg

EXPOSE 8989
CMD ["python", "app.py"]