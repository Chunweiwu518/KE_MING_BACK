import os
from typing import Any, Dict, List

from app.utils.vector_store import get_vector_store
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


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

    def setup_retrieval_qa(self):
        # 設置檢索問答系統
        retriever = self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.qa_prompt},
        )

    async def query(
        self, question: str, history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        qa = self.setup_retrieval_qa()
        result = await qa.acall({"query": question})

        answer = result["result"]
        sources = []

        # 提取來源文件信息
        if "source_documents" in result:
            for doc in result["source_documents"]:
                sources.append({"content": doc.page_content, "metadata": doc.metadata})

        return {"answer": answer, "sources": sources}

    def process_query(self, query, history=None):
        try:
            # 確保向量存儲已初始化
            if not self.vector_store:
                print("重新初始化向量存儲...")
                self.vector_store = get_vector_store()
                if not self.vector_store:
                    print("警告: 無法初始化向量存儲，可能沒有嵌入文件")
                    return {
                        "answer": "我沒有找到任何相關信息，可能是因為尚未上傳任何文件。",
                        "sources": [],
                    }

            # 打印診斷信息
            print(f"使用向量存儲搜索: {query}")

            # 獲取相關文檔
            docs = self.vector_store.similarity_search(query, k=3)

            print(f"找到 {len(docs)} 個相關文檔")

            # 生成回答
            answer, sources = self.generate_response(query, docs)

            print(f"生成回答: {answer[:50]}...")

            return {"answer": answer, "sources": sources}
        except Exception as e:
            import traceback

            print(f"處理查詢時出錯: {str(e)}")
            print(traceback.format_exc())
            raise

    def generate_response(self, query, docs):
        """根據查詢和文檔生成回答與來源"""
        try:
            # 如果沒有找到相關文檔
            if not docs or len(docs) == 0:
                return (
                    "我沒有找到與您問題相關的信息。請嘗試上傳更多相關文檔或重新表述您的問題。",
                    [],
                )

            # 準備上下文
            context = "\n\n".join([doc.page_content for doc in docs])

            # 創建提示
            prompt = self.qa_prompt.format(context=context, question=query)

            # 使用LLM生成回答
            messages = [
                {
                    "role": "system",
                    "content": "你是一個有幫助的助手，基於給定的上下文回答問題。",
                },
                {"role": "user", "content": prompt},
            ]

            # 調用 OpenAI API
            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, "content") else str(response)

            # 格式化來源信息
            sources = []
            for i, doc in enumerate(docs):
                source = {"content": doc.page_content, "metadata": doc.metadata}
                sources.append(source)

            return answer, sources

        except Exception as e:
            import traceback

            print(f"生成回答時出錯: {str(e)}")
            print(traceback.format_exc())
            # 返回一個友好的錯誤消息
            return f"抱歉，在處理您的問題時出現了錯誤。錯誤信息: {str(e)}", []
