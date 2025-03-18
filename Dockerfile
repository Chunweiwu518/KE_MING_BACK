FROM python:3.10.5-slim

WORKDIR /app

# 安裝基本系統依賴和編譯工具
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    tcl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 下載、編譯並安裝最新版本的 SQLite3
RUN cd /tmp && \
    wget https://www.sqlite.org/2024/sqlite-autoconf-3450200.tar.gz && \
    tar -xzf sqlite-autoconf-3450200.tar.gz && \
    cd sqlite-autoconf-3450200 && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf sqlite-autoconf-3450200*

# 重建 Python 的 sqlite3 模組
RUN pip install --upgrade pip && \
    pip uninstall -y pysqlite3 && \
    pip install --no-cache-dir pysqlite3-binary && \
    python -c "import sqlite3; print(f'SQLite3 版本: {sqlite3.sqlite_version}')"

# 設置環境變數以使 Python 使用正確的 SQLite3 庫
ENV LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

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

# 檢查 SQLite 版本
RUN sqlite3 --version && \
    python -c "import sqlite3; print(f'Python SQLite3 版本: {sqlite3.sqlite_version}')" 