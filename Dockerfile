# alpine + python3.7 + opencv4.1
FROM czentye/opencv-video-minimal

EXPOSE 7777

# Install falcon
RUN pip install falcon

# Add app code
COPY ./app /app

CMD ["python3", "app/mug_server.py"]