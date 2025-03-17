import os
import re
from typing import Any, Dict, List

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_openai import ChatOpenAI

from app.utils.vector_store import get_vector_store


class RAGEngine:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = ChatOpenAI(
            model_name=os.getenv("CHAT_MODEL_NAME", "gpt-3.5-turbo"),
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.qa_prompt = PromptTemplate(
            template="""你是一個有幫助的AI助手。使用以下上下文來回答問題。
            
            上下文: {context}
            
            問題: {question}
            
            如果你找不到答案，請直接說不知道，不要試圖捏造答案。
            """,
            input_variables=["context", "question"],
        )

        # 新增產品相關問題的專用提示
        self.product_qa_prompt = PromptTemplate(
            template="""你是一個產品信息專家。請使用以下上下文回答關於產品的問題。
            
            上下文: {context}
            
            問題: {question}
            
            請盡可能詳細地回答產品相關問題，包括產品名稱、描述、價格、類別和規格等信息。
            如果找不到特定信息，請明確指出哪些信息是可用的，哪些信息不可用。
            """,
            input_variables=["context", "question"],
        )

        # 修正診斷日誌
        try:
            print("向量存儲初始化成功")
            # 嘗試使用更安全的方式獲取文檔數量
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 1})
            print("檢索器初始化成功")
        except Exception as e:
            print(f"向量存儲診斷時發生錯誤: {str(e)}")

    def setup_retrieval_qa(self, is_product_query=False):
        # 設置檢索問答系統
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 12},  # 增加檢索數量
        )

        # 根據查詢類型選擇不同的提示模板
        prompt = self.product_qa_prompt if is_product_query else self.qa_prompt

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
            verbose=True,  # 添加詳細日誌輸出
        )

    async def query(
        self, question: str, history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        # 判斷是否是產品查詢
        is_product_query = self.is_product_query(question)

        qa = self.setup_retrieval_qa(is_product_query=is_product_query)
        result = await qa.acall({"query": question})

        answer = result["result"]
        sources = []

        # 提取來源文件信息
        if "source_documents" in result:
            for doc in result["source_documents"]:
                sources.append({"content": doc.page_content, "metadata": doc.metadata})

        return {"answer": answer, "sources": sources}

    def is_product_query(self, query: str) -> bool:
        """判斷是否是產品相關查詢"""
        # 檢查是否包含產品ID模式 (多種格式)
        product_id_patterns = [
            r"[A-Z]{2}-\d{4}",  # HK-2189
            r"[A-Z]{3}-[A-Z]\d{3}",  # EDS-G616
            r"[A-Z]{3}-\d{4}[A-Z]",  # 其他可能格式
            r"[A-Z]{2,4}-?[0-9]{3,4}[A-Z]?",  # 通用模式
        ]

        for pattern in product_id_patterns:
            if re.search(pattern, query):
                print(f"檢測到產品ID查詢：{query}")
                return True

        # 檢查是否包含產品相關關鍵詞
        product_keywords = [
            "產品",
            "商品",
            "規格",
            "價格",
            "類別",
            "工作燈",
            "頭燈",
            "詳細資訊",
            "手電筒",
            "軟管",
            "吸磁",
            "LED",
        ]
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

    def process_query(self, query, history=None):
        try:
            # 確保向量存儲已初始化
            if not self.vector_store:
                print("重新初始化向量存儲...")
                self.vector_store = get_vector_store()
                if not self.vector_store:
                    return "我沒有找到任何相關信息，可能是因為尚未上傳任何文件。", []

            print(f"處理查詢: '{query}'")

            # 使用更靈活的搜索策略
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 15,  # 增加檢索數量
                    "score_threshold": 0.5,  # 添加相似度閾值
                },
            )

            # 使用 RetrievalQA 進行查詢
            qa = self.setup_retrieval_qa(is_product_query=self.is_product_query(query))
            result = qa({"query": query})

            # 提取產品ID或關鍵特徵
            product_features = []
            for word in query.split():
                if any(
                    keyword in word.lower()
                    for keyword in ["led", "軟管", "手電筒", "吸磁"]
                ):
                    product_features.append(word)

            # 根據產品特徵過濾和組織答案
            answer = ""
            if product_features:
                answer = "### 相關產品資訊\n\n"
                content_lines = result["result"].split("\n")
                for line in content_lines:
                    if any(
                        feature.lower() in line.lower() for feature in product_features
                    ):
                        answer += line + "\n"
            else:
                answer = result["result"]

            if not answer.strip():
                answer = "抱歉，我找不到完全符合描述的產品。請嘗試使用產品型號或更具體的關鍵字搜尋。"

            # 處理來源文件
            sources = []
            for doc in result["source_documents"]:
                source_info = {"content": doc.page_content, "metadata": doc.metadata}
                sources.append(source_info)

            return answer, sources

        except Exception as e:
            print(f"生成回答時出錯: {str(e)}")
            raise

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

            # 創建提示，根據查詢類型選擇不同的提示模板
            prompt_template = (
                self.product_qa_prompt if is_product_query else self.qa_prompt
            )
            prompt = prompt_template.format(context=context, question=query)

            # 使用LLM生成回答
            messages = [
                {
                    "role": "system",
                    "content": "你是一個有幫助的助手，基於給定的上下文回答問題。"
                    if not is_product_query
                    else "你是一個產品信息專家，詳細解析產品信息並回答問題。",
                },
                {"role": "user", "content": prompt},
            ]

            # 調用 OpenAI API
            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)

            # 處理來源
            sources = []
            for doc in docs:
                if hasattr(doc, "metadata"):
                    source_info = {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                    }
                    sources.append(source_info)

            return answer, sources

        except Exception as e:
            print(f"生成回答時出錯: {str(e)}")
            raise


# 搜索配置
search_kwargs = {
    "k": 15,  # 檢索文檔數量
    "score_threshold": 0.5,  # 相似度閾值
}

# 產品關鍵字
product_keywords = [
    "產品",
    "商品",
    "規格",
    "價格",
    "類別",
    "工作燈",
    "頭燈",
    "詳細資訊",
    "手電筒",
    "軟管",
    "吸磁",
    "LED",
]

# 產品ID模式
product_id_patterns = [
    r"[A-Z]{2}-\d{4}",  # HK-2189
    r"[A-Z]{3}-[A-Z]\d{3}",  # EDS-G616
    r"[A-Z]{3}-\d{4}[A-Z]",  # 其他格式
]
