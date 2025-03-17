import json
import os

from langchain.schema import Document

from app.utils.openai_client import get_embeddings_model
from app.utils.vector_store import get_vector_store


# 添加新的JSON產品數據加載器
class JSONProductLoader:
    """加載JSON格式的產品數據"""

    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        """加載並處理JSON產品數據"""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            documents = []

            # 處理產品資料
            if "products" in data:
                for product in data["products"]:
                    # 產品基本資訊轉字符串
                    product_info = f"產品ID: {product['id']}\n"
                    product_info += f"產品名稱: {product['name']}\n"
                    product_info += f"產品描述: {product['description']}\n"
                    product_info += f"價格: {product['price']}\n"
                    product_info += f"類別: {product['category']}\n"

                    # 產品規格如果存在
                    if "specifications" in product:
                        product_info += "產品規格:\n"
                        for spec_key, spec_value in product["specifications"].items():
                            product_info += f"- {spec_key}: {spec_value}\n"

                    # 創建Document對象
                    doc = Document(
                        page_content=product_info,
                        metadata={
                            "source": self.file_path,
                            "filename": os.path.basename(self.file_path),
                            "product_id": product["id"],
                            "product_name": product["name"],
                            "product_category": product["category"],
                        },
                    )
                    documents.append(doc)

            return documents
        except Exception as e:
            print(f"處理JSON產品數據時出錯: {str(e)}")
            raise


async def process_document(file_path: str) -> bool:
    """處理上傳的文件，使用 GPT-4o 進行處理，並存儲到向量數據庫

    Args:
        file_path: 文件路徑

    Returns:
        處理是否成功
    """
    try:
        print(f"開始處理文件: {file_path}")

        # 檢查是否為 PDF 文件
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext != ".pdf":
            raise ValueError("只支持 PDF 文件格式")

        # 使用 GPT-4o 處理 PDF
        print("使用 GPT-4o 處理 PDF...")
        documents = process_pdf_with_gpt(file_path)
        print("處理成功，獲取文檔內容")

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
        print("將文檔添加到向量數據庫...")
        try:
            vector_store.add_documents(documents)
            print("文檔成功添加到向量數據庫!")
            return True
        except Exception as e:
            print(f"添加文檔時出錯: {str(e)}")
            try:
                vector_store.add_documents(documents, embedding=embedding_model)
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
        vector_store = get_vector_store()
        vector_store.delete(where={"source": file_path})
        return True
    except Exception as e:
        print(f"移除文件時出錯: {str(e)}")
        return False
