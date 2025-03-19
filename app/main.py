import os
import shutil

from app.routers import chat, history, upload
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 創建並設置必要的目錄
def setup_directories():
    # 設置靜態文件目錄
    static_dir = os.path.join(os.getcwd(), "static", "images", "products")
    os.makedirs(static_dir, exist_ok=True)
    os.chmod(static_dir, 0o777)
    
    # 設置上傳目錄
    upload_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    os.chmod(upload_dir, 0o777)

# 初始化目錄
setup_directories()

app = FastAPI(title="RAG API")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 註冊路由
app.include_router(chat.router)
app.include_router(upload.router)
app.include_router(history.router)

# 使用項目內的靜態文件
app.mount("/images", StaticFiles(directory="static/images"), name="images")

@app.get("/")
async def root():
    return {"message": "歡迎使用RAG API"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", reload=True)
