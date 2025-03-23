from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 添加 CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境應限制為具體域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 