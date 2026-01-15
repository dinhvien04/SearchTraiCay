# Dockerfile

# 1. Chọn base image có Python
FROM python:3.8-slim

# 2. Set thư mục làm việc trong container
WORKDIR /app

# 3. Copy file requirements vào container
COPY requirements.txt .

# 4. Cài đặt các thư viện Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 5. Copy toàn bộ code của bạn vào container (nếu đã có source)
COPY . .

# 6. Chạy một file Python mặc định (ví dụ chạy demo pipeline)
# Thay đổi tên file phù hợp dự án của bạn
CMD ["python", "notebook/demo_pipeline.py"]
