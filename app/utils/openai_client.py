import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings  # 使用最新的包

# 確保載入環境變數
load_dotenv()


def get_openai_client():
    """獲取OpenAI客戶端實例"""
    from openai import OpenAI

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embeddings_model():
    """獲取嵌入模型，優先使用較新的模型"""
    try:
        # 嘗試使用最新的嵌入模型
        model_name = "text-embedding-3-large"  # 或者使用text-embedding-3-small
        model = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            dimensions=1536,  # 調整維度
        )
        return model
    except Exception as e:
        print(f"使用新嵌入模型出錯: {str(e)}，將使用備用模型")
        # 備用模型
        model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        return model
