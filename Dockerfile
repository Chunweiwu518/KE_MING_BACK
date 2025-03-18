FROM python:3.10.5-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# 安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程序代碼
COPY . .

# 創建必要的目錄並設置權限
RUN mkdir -p /data/uploads /data/chroma_new && \
    chmod -R 777 /data

# 設置環境變數
ENV PYTHONPATH=/app
ENV DATA_PATH=/data

# 暴露端口
EXPOSE 8000

# 運行應用
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# 在複製文件之後添加
RUN mkdir -p /app/KE_MING_BACK/chroma_new && \
    chmod -R 777 /app/KE_MING_BACK/chroma_new && \
    touch /app/KE_MING_BACK/chroma_new/chroma.sqlite3 && \
    chmod 777 /app/KE_MING_BACK/chroma_new/chroma.sqlite3 