# Sử dụng image Python cơ bản
FROM python:3.12

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt các thư viện Python cần thiết
RUN pip install --upgrade pip
RUN pip install opencv-python-headless cvzone ultralytics torch

# Sao chép mã nguồn vào container
COPY . /app

# Thiết lập thư mục làm việc
WORKDIR /app

# Chạy ứng dụng
CMD ["python", "main.py"]
