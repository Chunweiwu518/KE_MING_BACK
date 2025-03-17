import traceback
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.rag.engine import RAGEngine

router = APIRouter(prefix="/api", tags=["chat"])
rag_engine = RAGEngine()


class ChatRequest(BaseModel):
    query: str
    history: Optional[List[Dict[str, str]]] = []


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Dict[str, Any]):
    try:
        query = request.get("query", "")
        history = request.get("history", [])

        if not query:
            raise HTTPException(status_code=400, detail="查詢不能為空")

        # 增加診斷日誌
        print(f"接收到查詢: {query}")

        # process_query 返回的是 tuple(answer, sources)
        answer, sources = rag_engine.process_query(query, history)

        # 構建正確的響應格式
        response = {"answer": answer, "sources": sources}

        # 調試輸出
        print(f"返回答案: {answer}")
        print(f"返回來源數量: {len(sources)}")

        return response
    except Exception as e:
        # 捕獲並打印詳細錯誤信息
        error_msg = f"處理查詢時出錯: {str(e)}"
        traceback_str = traceback.format_exc()
        print(error_msg)
        print(traceback_str)

        # 返回更詳細的錯誤信息
        raise HTTPException(status_code=500, detail=error_msg)
