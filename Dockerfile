FROM python:3.11-slim

WORKDIR /app

# 安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程序代碼
COPY . .

# 設置環境變數
ENV PYTHONPATH=/app

# 暴露端口
EXPOSE 8000

# 運行應用
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 