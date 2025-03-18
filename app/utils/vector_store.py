import os
import shutil

from app.utils.openai_client import get_embeddings_model
from langchain_chroma import Chroma
from chromadb.config import Settings

# 從環境變數獲取基礎路徑
BASE_PATH = os.getenv('DATA_PATH', '/tmp/KE_MING_BACK')
CHROMA_PATH = os.path.join(BASE_PATH, 'chroma_new')

# 保存全局實例以避免多次創建
_vector_store_instance = None


def get_vector_store(force_new=False):
    """獲取向量存儲"""
    global _vector_store_instance

    if force_new:
        reset_vector_store()
        _vector_store_instance = None

    if _vector_store_instance is None:
        persist_directory = CHROMA_PATH
        
        # 確保目錄存在
        os.makedirs(persist_directory, exist_ok=True)
        
        # 設置目錄權限
        try:
            os.chmod(persist_directory, 0o777)
            
            # 確保數據庫文件權限
            db_path = os.path.join(persist_directory, "chroma.sqlite3")
            if os.path.exists(db_path):
                os.chmod(db_path, 0o777)
                
            # 設置父目錄權限
            parent_dir = os.path.dirname(persist_directory)
            if os.path.exists(parent_dir):
                os.chmod(parent_dir, 0o777)
        except Exception as e:
            print(f"設置權限時出錯: {str(e)}")
            
        embedding_function = get_embeddings_model()
        
        # 使用SQLite配置
        client_settings = Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True,
            persist_directory=persist_directory
        )
        
        _vector_store_instance = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            client_settings=client_settings
        )
        
        # 驗證是否為空
        try:
            count = _vector_store_instance._collection.count()
            print(f"新建向量存儲實例，當前文檔數: {count}")
        except Exception as e:
            print(f"檢查文檔數時出錯: {str(e)}")

    return _vector_store_instance


def reset_vector_store():
    """清除全局實例並強制重新創建"""
    global _vector_store_instance

    if _vector_store_instance is not None:
        try:
            # 嘗試清空集合
            if hasattr(_vector_store_instance, "_collection"):
                try:
                    # 獲取所有文檔ID並刪除
                    docs = _vector_store_instance._collection.get()
                    if docs["ids"]:
                        _vector_store_instance._collection.delete(docs["ids"])
                    print("已清空集合中的所有文檔")
                except Exception as e:
                    print(f"清空集合時出錯: {str(e)}")

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

            print("向量存儲實例已完全重置")

        except Exception as e:
            print(f"關閉向量庫連接時出錯: {str(e)}")
            _vector_store_instance = None

    # 清理文件系統
    persist_directory = CHROMA_PATH
    if os.path.exists(persist_directory):
        try:
            # 刪除所有文件
            for root, dirs, files in os.walk(persist_directory):
                for f in files:
                    file_path = os.path.join(root, f)
                    try:
                        os.chmod(file_path, 0o777)
                        os.remove(file_path)
                        print(f"已刪除文件: {file_path}")
                    except Exception as e:
                        print(f"刪除文件失敗 {file_path}: {str(e)}")
            
            # 重新創建空目錄
            shutil.rmtree(persist_directory, ignore_errors=True)
            os.makedirs(persist_directory, exist_ok=True)
            os.chmod(persist_directory, 0o777)
            print("向量存儲目錄已重置")
        except Exception as e:
            print(f"清理文件系統時出錯: {str(e)}")

    return None
