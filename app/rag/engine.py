# ======== 添加 langchain debug 與 verbose 補丁 ========
# 這是一個修復方案，解決"module 'langchain' has no attribute 'debug'"和'verbose'錯誤
import importlib
import json
import logging
import os
import re
import traceback
from typing import Any, Dict, List

# 檢查 langchain 模組
langchain_module = importlib.import_module("langchain")
# 如果 langchain 模組中沒有 debug 屬性，添加一個空的 debug 函數
if not hasattr(langchain_module, "debug"):

    def dummy_debug(*args, **kwargs):
        logging.debug("Called dummy langchain.debug")
        return None

    # 將模擬的 debug 函數添加到 langchain 模組
    setattr(langchain_module, "debug", dummy_debug)
    print("全局：已添加模擬的 langchain.debug 函數以避免錯誤")

# 如果 langchain 模組中沒有 verbose 屬性，添加默認值
if not hasattr(langchain_module, "verbose"):
    # 添加 verbose 屬性並設為 False
    setattr(langchain_module, "verbose", False)
    print("全局：已添加模擬的 langchain.verbose 屬性以避免錯誤")
# ======== 補丁結束 ========

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from openai import OpenAI

from app.utils.vector_store import get_vector_store


class RAGEngine:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = ChatOpenAI(
            model_name=os.getenv("CHAT_MODEL_NAME", "gpt-4-1106-preview"),
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

        # 優化基本提示詞
        self.qa_prompt = PromptTemplate(
            template="""你是一個專業的知識庫助手。請使用以下提供的上下文來回答問題。

上下文信息：
{context}

用戶問題：{question}

回答要求：
1. 請基於上下文提供準確、相關的信息
2. 如果上下文中沒有相關信息，請明確說明"我在知識庫中找不到相關信息"
3. 回答要簡潔明瞭，避免重複信息
4. 適當引用上下文中的具體內容，增加可信度
5. 不要添加上下文之外的信息或個人推測

請以專業、客觀的語氣回答：""",
            input_variables=["context", "question"],
        )

        # 優化產品查詢提示詞
        self.product_qa_prompt = PromptTemplate(
            template="""你是一個專業的產品顧問。請使用以下產品信息來回答問題。

產品信息：
{context}

用戶問題：{question}

回答要求：
1. 優先提供產品的關鍵信息：型號、規格、特點
2. 如果是比較類問題，請列出產品間的主要區別
3. 價格信息要準確，並標註幣種
4. 規格數據要精確，包含單位
5. 如有產品優勢，請具體說明
6. 如果信息不完整，請明確指出缺少哪些信息

請以專業顧問的語氣回答：""",
            input_variables=["context", "question"],
        )

    def setup_retrieval_qa(self, is_product_query=False):
        """
        注意：此方法已棄用，因為在新版的langchain中可能存在兼容性問題。
        請使用process_query方法代替。
        """
        # 向控制台輸出警告
        print("警告: setup_retrieval_qa方法已棄用，請使用process_query")

        # 返回None表示此方法不再有效
        return None

    async def query(
        self, question: str, history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        注意：此方法已棄用，因為在新版的langchain中可能存在兼容性問題。
        請使用process_query方法代替。
        """
        # 向控制台輸出警告
        print("警告: query方法已棄用，請使用process_query")

        # 直接調用process_query處理查詢
        return self.process_query(question, history)

    def is_product_query(self, query: str) -> bool:
        """判斷是否是產品相關查詢"""
        # 檢查是否包含產品ID模式 (如HK-2189, TL-4523等)
        product_id_pattern = r"[A-Z]{2}-\d{4}"
        if re.search(product_id_pattern, query):
            return True

        # 檢查是否包含產品相關關鍵詞
        product_keywords = ["產品", "商品", "規格", "價格", "類別", "工作燈", "頭燈"]
        for keyword in product_keywords:
            if keyword in query:
                return True

        return False

    def get_product_by_id(self, product_id: str):
        """根據產品ID直接檢索相關文檔"""
        # 使用metadata過濾方式優先查詢
        try:
            # 嘗試使用metadata過濾
            docs = self.vector_store.get(where={"product_id": product_id})
            if docs and len(docs) > 0:
                # 確保返回的是Document對象
                if not all(hasattr(doc, "page_content") for doc in docs):
                    # 如果不是Document對象，轉換它們
                    docs = [
                        Document(
                            page_content=doc if isinstance(doc, str) else str(doc),
                            metadata={
                                "source": "product_data",
                                "product_id": product_id,
                            },
                        )
                        for doc in docs
                    ]
                return docs
        except Exception as e:
            print(f"使用metadata過濾查詢產品ID時出錯: {str(e)}")

        # 如果metadata過濾失敗，使用文本搜索
        docs = self.vector_store.similarity_search(product_id, k=3)

        # 確保返回的是Document對象
        if not all(hasattr(doc, "page_content") for doc in docs):
            # 如果不是Document對象，轉換它們
            docs = [
                Document(
                    page_content=doc if isinstance(doc, str) else str(doc),
                    metadata={"source": "similarity_search", "product_id": product_id},
                )
                for doc in docs
            ]

        return docs

    def generate_product_prompt(self, context, query):
        """生成針對產品查詢的提示模板"""
        prompt = f"""
        以下是產品目錄中的資訊：
        {context}
        
        用戶問題: {query}
        
        你是一個專業的產品顧問。根據以上產品目錄資訊，請回應用戶的問題。
        
        如果用戶詢問產品列表或種類，請遵循以下指南：
        1. 將產品按主要類別分組（如工作燈、手電筒、充電器等）
        2. 提供每個類別下大約有多少種產品
        3. 每個類別僅列出2-3個代表性產品作為例子
        4. 告知用戶可以詢問特定類別獲取更多詳情
        
        如果用戶詢問特定產品型號，請提供完整詳細資訊。
        
        保持回答簡潔有用，字數控制在150字以內。避免使用"根據提供的資訊"等引導語。
        """
        return prompt

    async def process_query(self, query, history=None):
        """處理用戶查詢並返回相關資訊和來源"""
        try:
            # 使用查詢變體增強檢索
            query_variations = await self.generate_query_variations(
                query, num_variations=5
            )  # 增加到5個變體
            print(f"生成的查詢變體: {query_variations}")

            # 嘗試搜索，首先使用混合搜索
            if hasattr(self.vector_store, "hybrid_search"):
                # 增加 alpha 值使關鍵詞匹配更重要
                results = self.vector_store.hybrid_search(
                    query=query,
                    k=10,  # 增加檢索數量
                    alpha=0.7,  # 提高關鍵詞匹配的權重 (0.5->0.7)
                )
                if results:
                    docs = [r[0] for r in results]
                else:
                    docs = []
            else:
                docs = []

            # 如果沒有足夠結果，使用變體查詢
            if len(docs) < 3:
                print("首次搜索結果不足，嘗試使用查詢變體")
                all_results = []
                for variant in query_variations:
                    variant_results = self.vector_store.similarity_search(
                        variant,
                        k=5,
                    )
                    all_results.extend(variant_results)

                # 去重
                seen_contents = set()
                unique_docs = []
                for doc in all_results:
                    if doc.page_content not in seen_contents:
                        seen_contents.add(doc.page_content)
                        unique_docs.append(doc)

                docs.extend(
                    [
                        doc
                        for doc in unique_docs
                        if doc.page_content not in seen_contents
                    ]
                )

            # 如果還是沒有足夠結果，進行關鍵詞回退搜索
            if len(docs) < 2:
                print("向量搜索結果不足，使用關鍵詞回退搜索")
                keyword_docs = self.keyword_fallback_search(
                    query, max_docs=8
                )  # 增加到8個文檔
                # 添加不重複的關鍵詞搜索結果
                for doc in keyword_docs:
                    if doc.page_content not in seen_contents:
                        seen_contents.add(doc.page_content)
                        docs.append(doc)

            # 如果有文檔，使用 LLM 生成回應
            if docs:
                # 生成回答與來源
                answer, sources = self.generate_response(
                    query, docs, self.is_product_query(query)
                )

                return {"answer": answer, "sources": sources}

        except Exception as e:
            print(f"處理查詢時出錯: {str(e)}")
            traceback_str = traceback.format_exc()
            print(f"詳細錯誤信息: {traceback_str}")
            return {"answer": f"處理您的問題時發生了技術問題: {str(e)}", "sources": []}

    def is_product_list_query(self, query):
        """判斷是否是產品列表查詢"""
        product_list_keywords = [
            "有哪些產品",
            "產品列表",
            "所有產品",
            "產品種類",
            "產品型號",
            "提供什麼產品",
            "銷售什麼",
            "賣什麼",
            "產品目錄",
        ]

        for keyword in product_list_keywords:
            if keyword in query:
                return True

        return False

    def generate_response(self, query, docs, is_product_query=False):
        """根據查詢和文檔生成回答與來源"""
        try:
            # 如果沒有找到相關文檔
            if not docs or len(docs) == 0:
                return (
                    "我沒有找到與您問題相關的信息。請嘗試上傳更多相關文檔或重新表述您的問題。",
                    [],
                )

            # 準備上下文
            context_parts = []
            for doc in docs:
                # 檢查 doc 是否為 Document 對象或字符串
                if hasattr(doc, "page_content"):
                    # 是 Document 對象
                    context_parts.append(doc.page_content)
                elif isinstance(doc, str):
                    # 是字符串
                    context_parts.append(doc)
                else:
                    # 其他類型
                    context_parts.append(str(doc))

            context = "\n\n".join(context_parts)

            # 使用直接方式生成回答，避免使用可能不兼容的鏈式調用
            if is_product_query:
                # 使用 generate_product_response 方法處理產品查詢
                answer = self.generate_product_response(context, query)
            else:
                # 使用一般提示模板
                prompt = self.qa_prompt.format(context=context, question=query)
                response = self.llm.invoke(prompt)
                answer = (
                    response.content if hasattr(response, "content") else str(response)
                )

            # 處理來源
            sources = []
            for doc in docs:
                if hasattr(doc, "metadata"):
                    source_info = {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "images": {},
                    }

                    # 解析圖片信息
                    images_str = doc.metadata.get("images", "{}")
                    try:
                        images_dict = json.loads(images_str)
                        for key, value in images_dict.items():
                            path, page = value.split("|")
                            source_info["images"][key] = {
                                "path": path,
                                "page": int(page),
                            }
                    except json.JSONDecodeError:
                        print(f"解析圖片信息時出錯: {images_str}")

                    sources.append(source_info)

            return answer, sources

        except Exception as e:
            print(f"生成回答時出錯: {str(e)}")
            traceback_str = traceback.format_exc()
            print(f"詳細錯誤信息: {traceback_str}")
            return ("處理您的問題時發生了技術問題，請稍後再試。", [])

    def generate_product_response(self, context, query):
        """生成產品摘要或推薦的回應"""

        gpt_params = {
            "model": os.getenv("CHAT_MODEL_NAME", "gpt-4o-mini"),
            "temperature": 0.6,
            "top_p": 0.7,
            "frequency_penalty": 0.3,
        }

        prompt = f"""
        以下是產品目錄中的資訊：
        {context}
        
        用戶問題: {query}
        """

        messages = [
            {
                "role": "system",
                "content": """你是專業的產品顧問。請遵循以下規則：
                1. 如果用戶詢問產品列表，按類別分組概述產品，每類僅舉2-3例
                2. 提供每類產品的大致數量
                3. 保持回答簡潔，避免冗長列表
                4. 告知用戶可以詢問特定類別獲取更多詳情
                5. 如果用戶詢問特定產品型號，則提供詳細資訊""",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        gpt_params.update({"messages": messages})

        # 調用 OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(**gpt_params)

        return response.choices[0].message.content

    async def generate_query_variations(self, query, num_variations=3):
        """生成查詢的多個變體以提高召回率"""
        try:
            prompt = f"""
            原始查詢: {query}
            
            請提供{num_variations}個語義相同但措辭不同的查詢變體，以幫助更好地搜索相關信息。
            針對以下幾種情況擴展原始查詢:
            1. 使用同義詞替換關鍵詞
            2. 如果查詢中涉及產品，嘗試使用更專業的產品描述方式
            3. 如果查詢是關於圖片或視覺信息，明確提及「圖片」、「圖表」等詞彙
            4. 如果查詢較短，適當擴展查詢上下文
            5. 重新排列查詢中的詞組順序
            
            僅返回變體列表，每行一個。
            """

            # 使用OpenAI生成變體
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.7,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一個幫助重新表述查詢的助手，需要專注於同義詞替換和重新表述。",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            # 解析回應
            variations_text = response.choices[0].message.content
            variations = [
                line.strip() for line in variations_text.split("\n") if line.strip()
            ]

            # 添加原始查詢
            variations = [query] + variations

            # 針對特定類型查詢的進一步處理
            if "圖" in query or "照片" in query or "顯示" in query:
                variations.append(f"請顯示{query}的相關圖片或示意圖")
                variations.append(f"展示關於{query}的圖形或照片資料")

            return variations
        except Exception as e:
            print(f"生成查詢變體時出錯: {str(e)}")
            return [query]  # 出錯時返回原始查詢

    def keyword_fallback_search(self, query, max_docs=5):
        """使用簡單的關鍵詞匹配作為向量搜索的回退方法"""
        try:
            # 使用jieba分詞提取關鍵詞
            import jieba

            keywords = [w for w in jieba.cut(query) if len(w.strip()) > 1]

            # 獲取所有文檔
            all_docs = []
            collection = self.vector_store._collection
            collection_data = collection.get()

            # 構建結果
            results = []
            for i, content in enumerate(collection_data["documents"]):
                score = 0
                for keyword in keywords:
                    if keyword in content:
                        score += 1

                if score > 0:
                    metadata = (
                        collection_data["metadatas"][i]
                        if collection_data["metadatas"]
                        else {}
                    )
                    results.append(
                        (Document(page_content=content, metadata=metadata), score)
                    )

            # 排序並返回前N個結果
            results.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in results[:max_docs]]
        except Exception as e:
            print(f"關鍵詞回退搜索出錯: {str(e)}")
            return []
