from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Source(BaseModel):
    content: str
    metadata: Dict[str, Any]
    page_info: Optional[str] = None
    images: Optional[Dict[str, Dict[str, Any]]] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
