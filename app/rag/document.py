import os

from app.utils.openai_client import get_embeddings_model
from app.utils.vector_store import get_vector_store
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
)


async def process_document(file_path: str) -> bool:
    """處理上傳的文件，切分並存儲到向量數據庫"""
    try:
        print(f"開始處理文件: {file_path}")

        # 根據文件類型選擇加載器
        file_ext = os.path.splitext(file_path)[1].lower()
        print(f"檢測到文件類型: {file_ext}")

        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")  # 確保指定編碼
        elif file_ext == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path)

        # 加載文件
        print(f"使用加載器: {type(loader).__name__}")
        documents = loader.load()
        print(f"成功加載 {len(documents)} 個文檔段落")

        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"文檔已分割為 {len(chunks)} 個塊")

        # 為每個塊添加源文件元數據 - 使用完整路徑作為識別
        for chunk in chunks:
            chunk.metadata["source"] = file_path  # 使用完整路徑
            chunk.metadata["filename"] = os.path.basename(
                file_path
            )  # 保存文件名用於顯示

        # 獲取向量存儲和嵌入模型
        print("初始化向量存儲和嵌入模型...")
        vector_store = get_vector_store()
        embedding_model = get_embeddings_model()

        # 先檢查並刪除相同路徑的舊文檔
        try:
            vector_store.delete(where={"source": file_path})
            print(f"已刪除文件 {file_path} 的現有向量")
        except Exception as del_e:
            print(f"刪除現有向量時出錯（可能是新文件）: {str(del_e)}")

        # 添加到向量數據庫
        print(f"將 {len(chunks)} 個塊添加到向量數據庫...")
        try:
            vector_store.add_documents(chunks)
            print("文檔成功添加到向量數據庫!")
            return True
        except Exception as e:
            print(f"添加文檔時出錯: {str(e)}")
            # 嘗試另一種方法
            try:
                vector_store.add_documents(chunks, embedding=embedding_model)
                print("使用替代方法成功添加文檔!")
                return True
            except Exception as e2:
                print(f"替代方法添加文檔失敗: {str(e2)}")
                raise e2
    except Exception as e:
        print(f"處理文件時出錯: {str(e)}")
        import traceback

        print(traceback.format_exc())
        return False


async def remove_document(file_path: str) -> bool:
    """從向量數據庫中移除文件"""
    try:
        # 獲取向量存儲實例
        vector_store = get_vector_store()

        # 使用文件路徑作為過濾條件刪除文檔
        vector_store.delete(where={"source": file_path})

        return True
    except Exception as e:
        print(f"移除文件時出錯: {str(e)}")
        return False
