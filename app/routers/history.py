import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["history"])


class Message(BaseModel):
    role: str
    content: str
    sources: Optional[List[Dict[str, Any]]] = None


class ChatHistory(BaseModel):
    id: str
    title: str
    messages: List[Message]
    createdAt: str
    lastMessage: Optional[str] = None  # 添加最後一條消息預覽


class CreateHistoryRequest(BaseModel):
    messages: List[Message]
    title: Optional[str] = None


# 初始化數據庫
def init_db():
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_histories
        (id TEXT PRIMARY KEY,
         title TEXT NOT NULL,
         messages TEXT NOT NULL,
         created_at TEXT NOT NULL,
         last_message TEXT)
    """)
    conn.commit()
    conn.close()


init_db()


def get_db():
    return sqlite3.connect("chat_history.db")


@router.get("/history", response_model=List[ChatHistory])
async def get_all_histories():
    """獲取所有對話歷史"""
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT * FROM chat_histories ORDER BY created_at DESC")
        rows = c.fetchall()
        histories = []
        for row in rows:
            messages = json.loads(row[2])
            # 獲取最後一條非用戶消息作為預覽
            last_message = next(
                (
                    msg["content"]
                    for msg in reversed(messages)
                    if msg["role"] == "assistant"
                ),
                "",
            )
            if len(last_message) > 50:
                last_message = last_message[:50] + "..."

            history = ChatHistory(
                id=row[0],
                title=row[1],
                messages=[Message(**msg) for msg in messages],
                createdAt=row[3],
                lastMessage=last_message,
            )
            histories.append(history)
        conn.close()
        return histories
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取對話歷史失敗: {str(e)}")


@router.get("/history/{chat_id}", response_model=ChatHistory)
async def get_chat_history(chat_id: str):
    """獲取特定對話的詳細信息"""
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT * FROM chat_histories WHERE id = ?", (chat_id,))
        row = c.fetchone()

        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="找不到指定的對話記錄")

        messages = json.loads(row[2])
        last_message = next(
            (
                msg["content"]
                for msg in reversed(messages)
                if msg["role"] == "assistant"
            ),
            "",
        )
        if len(last_message) > 50:
            last_message = last_message[:50] + "..."

        chat_history = ChatHistory(
            id=row[0],
            title=row[1],
            messages=[Message(**msg) for msg in messages],
            createdAt=row[3],
            lastMessage=last_message,
        )
        conn.close()
        return chat_history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取對話歷史失敗: {str(e)}")


@router.post("/history", response_model=ChatHistory)
async def create_chat_history(request: CreateHistoryRequest):
    """保存新的對話"""
    try:
        chat_id = str(uuid4())

        # 如果沒有提供標題，使用第一條消息的前20個字符
        title = request.title
        if not title and request.messages:
            first_message = request.messages[0].content.strip()
            if len(first_message) > 30:
                title = first_message[:30] + "..."
            else:
                title = first_message
        elif not title:
            title = f"對話 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        created_at = datetime.now().isoformat()
        messages_json = json.dumps([msg.dict() for msg in request.messages])

        conn = get_db()
        c = conn.cursor()
        c.execute(
            "INSERT INTO chat_histories (id, title, messages, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, title, messages_json, created_at),
        )
        conn.commit()
        conn.close()

        return ChatHistory(
            id=chat_id, title=title, messages=request.messages, createdAt=created_at
        )
    except Exception as e:
        print(f"創建對話記錄失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"創建對話記錄失敗: {str(e)}")


@router.put("/history/{chat_id}", response_model=ChatHistory)
async def update_chat_history(chat_id: str, request: CreateHistoryRequest):
    """更新現有對話"""
    try:
        conn = get_db()
        c = conn.cursor()

        # 檢查對話是否存在
        c.execute("SELECT * FROM chat_histories WHERE id = ?", (chat_id,))
        if not c.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="找不到指定的對話記錄")

        # 更新標題和消息
        title = request.title
        if not title and request.messages:
            first_message = request.messages[0].content.strip()
            if len(first_message) > 30:
                title = first_message[:30] + "..."
            else:
                title = first_message
        elif not title:
            c.execute("SELECT title FROM chat_histories WHERE id = ?", (chat_id,))
            title = c.fetchone()[0]

        messages_json = json.dumps([msg.dict() for msg in request.messages])

        c.execute(
            "UPDATE chat_histories SET title = ?, messages = ? WHERE id = ?",
            (title, messages_json, chat_id),
        )
        conn.commit()

        # 獲取更新後的記錄
        c.execute("SELECT * FROM chat_histories WHERE id = ?", (chat_id,))
        row = c.fetchone()
        conn.close()

        messages = json.loads(row[2])
        return ChatHistory(
            id=row[0],
            title=row[1],
            messages=[Message(**msg) for msg in messages],
            createdAt=row[3],
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"更新對話記錄失敗: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新對話記錄失敗: {str(e)}")


@router.delete("/history/clear_all")
async def clear_all_chats():
    conn = None
    try:
        print("收到清空所有對話請求")
        # 刪除所有對話
        conn = get_db()
        c = conn.cursor()
        c.execute("DELETE FROM chat_histories")
        conn.commit()
        return {"message": "所有對話已清空"}
    except Exception as e:
        print(f"清空所有對話時出錯: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清空所有對話失敗: {str(e)}")
    finally:
        if conn:
            try:
                conn.close()
            except Exception as e:
                print(f"關閉數據庫連接時出錯: {str(e)}")


@router.delete("/history/{chat_id}")
async def delete_chat(chat_id: str):
    try:
        # 刪除指定的對話
        conn = get_db()
        c = conn.cursor()
        c.execute("DELETE FROM chat_histories WHERE id = ?", (chat_id,))
        if c.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="找不到指定的對話記錄")
        conn.commit()
        conn.close()
        return {"message": "對話已刪除"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刪除對話失敗: {str(e)}")
