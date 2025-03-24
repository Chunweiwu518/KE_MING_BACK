import os
import shutil

import jieba
from chromadb.config import Settings
from langchain_chroma import Chroma

from app.utils.openai_client import get_embeddings_model

# 從環境變數獲取基礎路徑
# 使用環境變量或使用Render平台支持寫入的目錄
BASE_PATH = os.getenv("DATA_PATH", os.path.join(os.getcwd(), ".render", "data"))
CHROMA_PATH = os.path.join(BASE_PATH, "chroma_new")

# 保存全局實例以避免多次創建
_vector_store_instance = None


def limit_k(vs, k):
    """自適應限制 k 值，避免請求太多超過索引的結果"""
    try:
        # 獲取向量存儲中的文檔數量
        if hasattr(vs, "_collection") and vs._collection is not None:
            count = vs._collection.count()
            # 如果請求的 k 大於文檔數量，則返回文檔數量
            if count < k:
                print(
                    f"Number of requested results {k} is greater than number of elements in index {count}, updating n_results = {count}"
                )
                return min(count, max(1, count))  # 至少返回1
        return k
    except Exception as e:
        print(f"限制 k 值時出錯: {str(e)}")
        return k  # 出錯時返回原始值


# 擴展 Chroma 類以增加自適應的 k 限制
class ChromaWithLimit(Chroma):
    """增強版的 Chroma 類，添加了 k 值的自適應限制"""

    def similarity_search(self, query, k=4, **kwargs):
        """執行相似度搜索，自動限制 k 值"""
        # 移除 score_threshold 參數
        if "score_threshold" in kwargs:
            del kwargs["score_threshold"]
        
        k = limit_k(self, k)
        return super().similarity_search(query, k=k, **kwargs)

    def similarity_search_with_score(self, query, k=4, **kwargs):
        """執行帶分數的相似度搜索，自動限制 k 值"""
        # 移除 score_threshold 參數
        if "score_threshold" in kwargs:
            del kwargs["score_threshold"]
        
        k = limit_k(self, k)
        return super().similarity_search_with_score(query, k=k, **kwargs)


class ChromaWithHybridSearch(ChromaWithLimit):
    """增強版的Chroma類，添加了混合搜索功能"""

    def hybrid_search(self, query, k=8, alpha=0.5, **kwargs):
        """
        實現混合搜索：結合向量相似度和關鍵詞匹配
        alpha: 控制向量相似度vs關鍵詞匹配的權重 (0-1)
        """
        try:
            # 向量相似度搜索
            vector_results = self.similarity_search_with_score(query, k=k * 2)
            
            # 關鍵詞匹配 - 使用更多同義詞擴展
            keywords = [w for w in jieba.cut(query) if len(w.strip()) > 1]
            
            # 加入產品型號的特殊處理
            if any(k for k in query.split() if "-" in k):
                # 產品型號可能包含連字符，直接添加完整型號作為關鍵詞
                model_keywords = [k for k in query.split() if "-" in k]
                keywords.extend(model_keywords)
            
            print(f"從查詢中提取的關鍵詞: {keywords}")
            
            # 擴展同義詞
            expanded_keywords = []
            for keyword in keywords:
                expanded_keywords.append(keyword)
                # 添加更多產品相關同義詞
                # 可以根據您的產品目錄添加更多規則
            
            # 降低相似度閾值
            min_score_threshold = 0.3  # 降低閾值以獲取更多結果
            
            # 獲取所有文檔
            collection_data = self._collection.get()
            all_docs = collection_data["documents"]
            all_ids = collection_data["ids"]
            all_metadatas = collection_data["metadatas"]

            # 計算關鍵詞匹配分數
            keyword_scores = {}
            for i, doc in enumerate(all_docs):
                score = 0
                for keyword in keywords:
                    if keyword in doc:
                        score += 1  # 簡單計數
                if score > 0:
                    keyword_scores[all_ids[i]] = (
                        score / len(keywords),
                        doc,
                        all_metadatas[i] if all_metadatas else {},
                    )

            # 合併結果
            hybrid_results = {}

            # 處理向量結果
            for doc, score in vector_results:
                doc_id = doc.metadata.get("id", "unknown")
                if doc_id in hybrid_results:
                    hybrid_results[doc_id]["vector_score"] = 1 - score  # 轉換為相似度
                else:
                    hybrid_results[doc_id] = {
                        "doc": doc,
                        "vector_score": 1 - score,  # 轉換為相似度
                        "keyword_score": 0,
                    }

            # 處理關鍵詞結果
            for doc_id, (kw_score, content, metadata) in keyword_scores.items():
                if doc_id in hybrid_results:
                    hybrid_results[doc_id]["keyword_score"] = kw_score
                else:
                    from langchain_core.documents import Document

                    doc = Document(page_content=content, metadata=metadata)
                    hybrid_results[doc_id] = {
                        "doc": doc,
                        "vector_score": 0,
                        "keyword_score": kw_score,
                    }

            # 計算最終分數
            scored_results = []
            for doc_id, data in hybrid_results.items():
                final_score = (
                    alpha * data["vector_score"] + (1 - alpha) * data["keyword_score"]
                )
                scored_results.append((data["doc"], final_score))

            # 排序並返回前k個結果
            scored_results.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_results[:k]]

        except Exception as e:
            print(f"混合搜索出錯: {str(e)}")
            return []


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
            persist_directory=persist_directory,
        )

        # 使用增強版的 ChromaWithLimit 類
        _vector_store_instance = ChromaWithLimit(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            client_settings=client_settings,
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


def keyword_fallback_search(self, query, max_docs=5):
    """使用更強大的關鍵詞匹配作為向量搜索的回退方法"""
    try:
        # 使用jieba分詞提取關鍵詞
        import jieba
        import jieba.analyse

        # 增加自訂字典以改善分詞
        jieba.load_userdict("user_dict.txt")  # 如果有自訂字典的話

        # 使用 TF-IDF 提取關鍵詞，而不僅是簡單分詞
        keywords = jieba.analyse.extract_tags(query, topK=8)
        print(f"從查詢中提取的關鍵詞: {keywords}")

        # 加入同義詞或相關詞擴展
        expanded_keywords = []
        for keyword in keywords:
            expanded_keywords.append(keyword)
            # 簡單的同義詞處理 (實際項目中可以使用更複雜的同義詞庫)
            if keyword == "圖":
                expanded_keywords.extend(["圖片", "照片", "示意圖", "表格"])
            elif keyword == "燈":
                expanded_keywords.extend(["照明", "燈具", "LED", "燈泡"])
            # 可以根據具體領域添加更多同義詞

        keywords = list(set(expanded_keywords))  # 去重
        print(f"擴展後的關鍵詞: {keywords}")

        # 獲取所有文檔
        collection = self.vector_store._collection
        collection_data = collection.get()

        # 計算關鍵詞匹配分數，採用更複雜的評分機制
        results = []
        for i, content in enumerate(collection_data["documents"]):
            score = 0
            content_lower = content.lower()

            # 基本關鍵詞匹配
            for keyword in keywords:
                if keyword in content:
                    score += 1  # 基本分

                    # 給標題或特殊位置的關鍵詞更高權重
                    if (
                        f"產品名稱: {keyword}" in content
                        or f"型號: {keyword}" in content
                    ):
                        score += 2

                    # 計算關鍵詞頻率
                    keyword_count = content.count(keyword)
                    if keyword_count > 1:
                        score += min(keyword_count * 0.5, 3)  # 最多額外加3分

            if score > 0:
                metadata = (
                    collection_data["metadatas"][i]
                    if collection_data["metadatas"]
                    else {}
                )
                doc = Document(page_content=content, metadata=metadata)
                results.append((doc, score))

        # 排序並返回前N個結果
        results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in results[:max_docs]]
    except Exception as e:
        print(f"關鍵詞回退搜索出錯: {str(e)}")
        return []
