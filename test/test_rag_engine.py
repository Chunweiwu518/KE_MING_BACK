from app.rag.engine import RAGEngine


def test_rag_engine():
    print("測試 RAG 引擎")

    engine = RAGEngine()

    # 測試簡單問題
    query = "這是一個測試"

    try:
        print(f"處理查詢: {query}")
        result = engine.process_query(query)
        print(f"返回答案: {result.get('answer')}")
        print(f"來源數量: {len(result.get('sources', []))}")
    except Exception as e:
        print(f"錯誤: {str(e)}")
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    test_rag_engine()
