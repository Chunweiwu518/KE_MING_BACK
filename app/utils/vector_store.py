import os

from app.utils.openai_client import get_embeddings_model
from langchain_chroma import Chroma

# 保存全局實例以避免多次創建
_vector_store_instance = None


def get_vector_store(force_new=False):
    """獲取向量存儲"""
    global _vector_store_instance

    # 如果強制創建新實例或尚未創建實例，則創建新的
    if force_new or _vector_store_instance is None:
        # 避免與舊版衝突，使用新目錄
        persist_directory = os.path.join(os.getcwd(), "chroma_new")

        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory, exist_ok=True)

        # 獲取嵌入模型
        embedding_function = get_embeddings_model()

        # 初始化向量存儲
        _vector_store_instance = Chroma(
            persist_directory=persist_directory, embedding_function=embedding_function
        )

    return _vector_store_instance


def reset_vector_store():
    """清除全局實例並強制重新創建"""
    global _vector_store_instance

    if _vector_store_instance is not None:
        try:
            # 嘗試多種方式關閉連接
            if hasattr(_vector_store_instance, "_client"):
                if hasattr(_vector_store_instance._client, "close"):
                    _vector_store_instance._client.close()
                if hasattr(_vector_store_instance._client, "_collection"):
                    if hasattr(_vector_store_instance._client._collection, "_client"):
                        _vector_store_instance._client._collection._client.close()

            # 強制清理相關屬性
            _vector_store_instance._client = None
            _vector_store_instance = None

        except Exception as e:
            print(f"關閉向量庫連接時出錯: {str(e)}")

    return None
