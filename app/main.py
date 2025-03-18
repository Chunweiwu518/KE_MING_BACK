import os
import shutil

from app.routers import chat, history, upload
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 創建必要的目錄
static_dir = os.path.join(os.getcwd(), "static", "images", "products")
os.makedirs(static_dir, exist_ok=True)

app = FastAPI(title="RAG API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 添加前端服務的地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 註冊路由
app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(history.router)

# 檢查是否需要重置數據庫
if os.path.exists("RESET_DB"):
    print("檢測到知識庫重置信號，正在重置...")

    # 清空知識庫目錄
    for db_dir in ["chroma_new", "chroma_db", "vector_db"]:
        db_path = os.path.join(os.getcwd(), db_dir)
        if os.path.exists(db_path):
            try:
                shutil.rmtree(db_path, ignore_errors=True)
                print(f"已刪除向量庫目錄: {db_path}")
                os.makedirs(db_path, exist_ok=True)
            except Exception as e:
                print(f"無法重置目錄 {db_path}: {str(e)}")

    # 刪除信號文件
    os.remove("RESET_DB")
    print("知識庫重置完成")

app.mount("/images", StaticFiles(directory="static/images"), name="images")

@app.get("/")
async def root():
    return {"message": "歡迎使用RAG API"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
