import os
from typing import List, Optional, Any

# 修改導入語句以兼容 supabase-py
from supabase import create_client
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from app.utils.openai_client import get_embeddings_model

# Supabase 配置
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
COLLECTION_NAME = "documents"

# 全局 Supabase 客戶端實例
_supabase_client = None
_vector_store_instance = None


def get_supabase_client():
    """獲取 Supabase 客戶端實例"""
    global _supabase_client
    
    if not _supabase_client:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("請在環境變數中設置 SUPABASE_URL 和 SUPABASE_KEY")
        
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print(f"已連接到 Supabase: {SUPABASE_URL}")
    
    return _supabase_client


class SupabaseVectorStore:
    """Supabase 向量存儲實現"""
    
    def __init__(self, client=None, embedding_model=None, table_name="documents"):
        """初始化 Supabase 向量存儲"""
        self.client = client or get_supabase_client()
        self.embedding_model = embedding_model or get_embeddings_model()
        self.table_name = table_name
        
        print(f"初始化 Supabase 向量存儲，表名: {table_name}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """添加文檔到向量存儲"""
        print(f"添加 {len(documents)} 個文檔到 Supabase 向量存儲")
        
        for doc in documents:
            # 獲取嵌入向量
            embeddings = self.embedding_model.embed_query(doc.page_content)
            
            # 準備插入數據
            data = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "embedding": embeddings,
                "source": doc.metadata.get("source", "unknown")
            }
            
            # 插入到 Supabase
            response = self.client.table(self.table_name).insert(data).execute()
            
            if hasattr(response, 'error') and response.error:
                print(f"插入文檔時出錯: {response.error.message}")
    
    def search(self, query: str, limit: int = 5) -> List[Document]:
        """搜索相似文檔"""
        # 獲取查詢嵌入向量
        query_embedding = self.embedding_model.embed_query(query)
        
        # 使用 Supabase 的向量相似度搜索
        response = (
            self.client.rpc(
                "match_documents", 
                {
                    "query_embedding": query_embedding,
                    "match_count": limit
                }
            )
            .execute()
        )
        
        if hasattr(response, 'error') and response.error:
            print(f"搜索文檔時出錯: {response.error.message}")
            return []
        
        # 轉換為 Document 對象
        results = []
        for item in response.data:
            doc = Document(
                page_content=item['content'],
                metadata=item['metadata']
            )
            results.append(doc)
        
        return results
    
    def delete(self, filter=None, **kwargs) -> None:
        """刪除文檔"""
        if not filter:
            print("刪除所有文檔")
            self.client.table(self.table_name).delete().neq("id", 0).execute()
            return
        
        # 根據 source 刪除
        if "source" in filter:
            print(f"刪除來源為 {filter['source']} 的文檔")
            self.client.table(self.table_name).delete().eq("source", filter["source"]).execute()
            return


def get_vector_store(force_new=False):
    """獲取向量存儲實例"""
    global _vector_store_instance
    
    if force_new or not _vector_store_instance:
        embedding_model = get_embeddings_model()
        _vector_store_instance = SupabaseVectorStore(
            embedding_model=embedding_model,
            table_name=COLLECTION_NAME
        )
    
    return _vector_store_instance


def reset_vector_store():
    """重置向量存儲"""
    vector_store = get_vector_store()
    vector_store.delete()
    print("已重置 Supabase 向量存儲")
    
    return None 