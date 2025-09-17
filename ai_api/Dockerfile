FROM ultralytics/ultralytics:latest

WORKDIR /usr/src/app

# ติดตั้ง dependencies เพิ่มเติม
RUN pip install Flask==3.1.0 flask-cors==5.0.1 schedule==1.2.2

# แทนที่ opencv-python ด้วย opencv-python-headless
RUN pip uninstall -y opencv-python && pip install opencv-python-headless

# Copy โค้ด
COPY . .

# ตั้งค่า environment variables เพื่อหลีกเลี่ยง GUI
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=:99
ENV MPLBACKEND=Agg

EXPOSE 8989
CMD ["python", "app.py"]