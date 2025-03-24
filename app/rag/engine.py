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

    def get_product_by_id(self, product_id):
        """通過產品ID直接檢索產品信息"""
        try:
            # 獲取所有文檔
            collection = self.vector_store._collection
            collection_data = collection.get()
            
            results = []
            for i, content in enumerate(collection_data["documents"]):
                # 檢查產品ID是否在文檔中
                if product_id in content:
                    metadata = collection_data["metadatas"][i] if collection_data["metadatas"] else {}
                    results.append(Document(page_content=content, metadata=metadata))
            
            print(f"通過產品ID '{product_id}' 找到 {len(results)} 個文檔")
            return results
        except Exception as e:
            print(f"通過產品ID檢索出錯: {str(e)}")
            traceback.print_exc()
            return []

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
            )
            print(f"生成的查詢變體: {query_variations}")

            # 檢查是否包含產品型號
            product_model = None
            for part in query.split():
                if "-" in part:
                    product_model = part
                    break
            
            # 如果是產品型號查詢，先嘗試直接通過產品ID檢索
            docs = []
            if product_model:
                print(f"檢測到產品型號查詢: {product_model}")
                product_docs = self.get_product_by_id(product_model)
                if product_docs:
                    print(f"通過產品ID直接找到 {len(product_docs)} 個文檔")
                    docs.extend(product_docs)

            # 如果直接檢索沒有結果，嘗試混合搜索
            if len(docs) < 2:
                if hasattr(self.vector_store, "hybrid_search"):
                    results = self.vector_store.hybrid_search(
                        query=query,
                        k=10,
                        alpha=0.6,
                    )
                    if results:
                        docs.extend([r for r in results if r.page_content not in [d.page_content for d in docs]])
                        print(f"混合搜索找到 {len(docs)} 個文檔")

                # 如果沒有足夠結果，使用變體查詢
                if len(docs) < 3:
                    print("首次搜索結果不足，嘗試使用查詢變體")
                    all_results = []
                    for variant in query_variations:
                        print(f"使用變體查詢: {variant}")
                        variant_results = self.vector_store.similarity_search(
                            variant,
                            k=5,
                        )
                        print(f"變體 '{variant}' 找到 {len(variant_results)} 個文檔")
                        all_results.extend(variant_results)

                    # 去重
                    seen_contents = set([d.page_content for d in docs])
                    for doc in all_results:
                        if doc.page_content not in seen_contents:
                            seen_contents.add(doc.page_content)
                            docs.append(doc)
                    print(f"變體查詢後共有 {len(docs)} 個文檔")

                # 如果還是沒有足夠結果，進行關鍵詞回退搜索
                if len(docs) < 2:
                    print("向量搜索結果不足，使用關鍵詞回退搜索")
                    keyword_docs = self.keyword_fallback_search(
                        query, max_docs=8
                    )
                    print(f"關鍵詞回退搜索找到 {len(keyword_docs)} 個文檔")
                    
                    # 添加不重複的關鍵詞搜索結果
                    seen_contents = set([d.page_content for d in docs])
                    for doc in keyword_docs:
                        if doc.page_content not in seen_contents:
                            seen_contents.add(doc.page_content)
                            docs.append(doc)
                    print(f"關鍵詞回退搜索後共有 {len(docs)} 個文檔")

            # 如果有文檔，使用 LLM 生成回應
            if docs:
                print(f"最終使用 {len(docs)} 個文檔生成回答")
                # 打印找到的文檔內容以便調試
                for i, doc in enumerate(docs):
                    print(f"文檔 {i+1} 內容: {doc.page_content[:100]}...")
                
                # 生成回答與來源
                answer, sources = self.generate_response(
                    query, docs, self.is_product_query(query)
                )
                return {"answer": answer, "sources": sources}
            else:
                # 如果沒有找到相關文檔，返回一個友好的消息
                print("未找到相關文檔，返回默認回應")
                return {
                    "answer": "我沒有找到與您問題相關的資訊。請嘗試重新表述您的問題，或者上傳更多相關文檔。",
                    "sources": []
                }
        except Exception as e:
            print(f"處理查詢時出錯: {str(e)}")
            traceback_str = traceback.format_exc()
            print(f"詳細錯誤信息: {traceback_str}")
            # 確保即使出錯也返回有效的響應
            return {
                "answer": f"處理您的問題時發生了技術問題: {str(e)}",
                "sources": []
            }

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

    async def generate_query_variations(self, query, num_variations=5):
        """生成查詢變體以提高檢索效果"""
        try:
            # 檢查是否包含產品型號
            contains_product_model = any(part for part in query.split() if "-" in part)
            
            if contains_product_model:
                # 提取產品型號
                product_models = [part for part in query.split() if "-" in part]
                model = product_models[0] if product_models else ""
                
                # 為產品型號創建特定的變體
                variations = [
                    query,  # 原始查詢
                    f"{model} 產品資訊",
                    f"{model} 規格",
                    f"{model} 特性",
                    f"{model} 功能",
                    f"{model} 使用說明"
                ]
                
                # 如果變體數量不足，添加更多通用變體
                if len(variations) < num_variations:
                    additional = [
                        f"{model} 技術參數",
                        f"{model} 應用場景",
                        f"{model} 優勢",
                        f"{model} 安裝方式"
                    ]
                    variations.extend(additional[:num_variations-len(variations)])
                    
                return variations[:num_variations]
            else:
                # 原有的查詢變體生成邏輯
                prompt = f"""
                原始查詢: {query}
                
                請提供{num_variations}個語義相同但措辭不同的查詢變體，以幫助更好地搜索相關信息。
                針對以下幾種情況擴展原始查詢:
                1. 使用同義詞替換關鍵詞
                2. 如果查詢中涉及產品，嘗試使用更專業的產品描述方式
                3. 重新排列查詢中的詞組順序
                
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
                
                return variations[:num_variations]
        except Exception as e:
            print(f"生成查詢變體時出錯: {str(e)}")
            # 返回原始查詢作為備用
            return [query]

    def keyword_fallback_search(self, query, max_docs=5):
        """當向量搜索失敗時的關鍵詞回退搜索"""
        try:
            # 使用jieba分詞提取關鍵詞
            import jieba
            keywords = [w for w in jieba.cut(query) if len(w.strip()) > 1]
            
            # 特殊處理產品型號 (通常包含連字符或數字)
            model_parts = []
            for part in query.split():
                if "-" in part:
                    model_parts.append(part)
                    # 也添加不帶連字符的版本
                    model_parts.append(part.replace("-", ""))
            
            if model_parts:
                keywords.extend(model_parts)
                # 將產品型號設為最高優先級
                keywords = model_parts + [k for k in keywords if k not in model_parts]
            
            print(f"關鍵詞回退搜索使用的關鍵詞: {keywords}")
            
            # 獲取所有文檔
            collection = self.vector_store._collection
            collection_data = collection.get()
            
            # 打印文檔數量以便調試
            print(f"知識庫中的文檔總數: {len(collection_data['documents']) if collection_data['documents'] else 0}")
            
            # 計算關鍵詞匹配分數
            results = []
            for i, content in enumerate(collection_data["documents"]):
                score = 0
                
                # 產品型號匹配給予更高權重
                for model in model_parts:
                    if model in content:
                        score += 5  # 產品型號匹配給予高分
                
                # 一般關鍵詞匹配
                for keyword in keywords:
                    if keyword in content:
                        score += 1
                        # 計算關鍵詞頻率
                        keyword_count = content.count(keyword)
                        if keyword_count > 1:
                            score += min(keyword_count * 0.5, 3)
                
                # 只要有任何匹配就返回
                if score > 0:
                    metadata = collection_data["metadatas"][i] if collection_data["metadatas"] else {}
                    results.append((Document(page_content=content, metadata=metadata), score))
                    print(f"找到匹配文檔，分數: {score}, 內容: {content[:50]}...")
            
            # 排序並返回前N個結果
            results.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in results[:max_docs]]
        except Exception as e:
            print(f"關鍵詞回退搜索出錯: {str(e)}")
            traceback.print_exc()
            return []
