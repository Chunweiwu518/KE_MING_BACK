import os
import time

import requests

BASE_URL = "http://localhost:8000/api"
TEST_FILE_PATH = "test_document.txt"  # 準備一個測試文件


def create_test_file():
    """創建測試文件"""
    content = """
    這是一個測試文檔。
    用於測試RAG應用程序的功能。
    這個文檔包含一些簡單的信息。
    台灣是一個美麗的島嶼。
    人工智能正在迅速發展。
    """

    with open(TEST_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"測試文件已創建: {TEST_FILE_PATH}")


def test_upload_file():
    """測試文件上傳功能"""
    print("\n===== 測試文件上傳 =====")

    url = f"{BASE_URL}/upload"

    with open(TEST_FILE_PATH, "rb") as f:
        files = {"file": (os.path.basename(TEST_FILE_PATH), f, "text/plain")}
        response = requests.post(url, files=files)

    print(f"狀態碼: {response.status_code}")
    print(f"響應: {response.json()}")

    assert response.status_code == 200, "文件上傳失敗"
    assert response.json()["status"] == "success", "上傳狀態不是success"

    # 等待文件被處理和索引 - 使用更智能的等待機制
    print("正在等待文件處理和索引完成...")

    # 嘗試查詢文件中的內容，直到找到結果或超時
    max_attempts = 10
    attempt = 0
    indexed = False

    while attempt < max_attempts and not indexed:
        # 嘗試查詢文件內容
        test_query = "測試文檔"
        chat_url = f"{BASE_URL}/chat"
        query_response = requests.post(
            chat_url, json={"query": test_query, "history": []}
        )

        if query_response.status_code == 200:
            result = query_response.json()
            if len(result.get("sources", [])) > 0:
                print("文件已成功索引!")
                indexed = True
                break

        # 如果還沒索引，等待一會再試
        attempt += 1
        wait_time = 2 * attempt  # 漸進增加等待時間
        print(f"文件還在處理中... 等待 {wait_time} 秒 (嘗試 {attempt}/{max_attempts})")
        time.sleep(wait_time)

    if not indexed:
        print("警告: 在允許的嘗試次數內未檢測到文件被索引")

    return response.json()


def test_chat():
    """測試聊天功能"""
    print("\n===== 測試聊天查詢 =====")

    url = f"{BASE_URL}/chat"

    # 準備幾個測試問題
    test_questions = ["台灣是什麼?", "文檔中提到了什麼技術?"]

    results = []

    for question in test_questions:
        print(f"\n問題: {question}")

        payload = {"query": question, "history": []}

        response = requests.post(url, json=payload)

        print(f"狀態碼: {response.status_code}")
        result = response.json()
        print(f"回答: {result['answer']}")
        print(f"來源數量: {len(result['sources'])}")

        assert response.status_code == 200, "聊天請求失敗"
        assert "answer" in result, "回答缺失"
        assert "sources" in result, "來源缺失"

        results.append(result)

    return results


def test_history():
    """測試歷史記錄功能"""
    print("\n===== 測試歷史記錄 =====")

    # 添加消息到歷史記錄
    add_url = f"{BASE_URL}/history/add"

    test_message = {"role": "user", "content": "這是一條測試消息"}

    add_response = requests.post(add_url, json=test_message)
    print(f"添加歷史記錄狀態碼: {add_response.status_code}")
    print(f"添加響應: {add_response.json()}")

    assert add_response.status_code == 200, "添加歷史記錄失敗"

    # 獲取歷史記錄
    get_url = f"{BASE_URL}/history"
    get_response = requests.get(get_url)

    print(f"獲取歷史記錄狀態碼: {get_response.status_code}")
    history = get_response.json()
    print(f"歷史記錄數量: {len(history)}")

    assert get_response.status_code == 200, "獲取歷史記錄失敗"
    assert len(history) > 0, "歷史記錄為空"

    # 清空歷史記錄
    clear_url = f"{BASE_URL}/history/clear"
    clear_response = requests.delete(clear_url)

    print(f"清空歷史記錄狀態碼: {clear_response.status_code}")
    print(f"清空響應: {clear_response.json()}")

    assert clear_response.status_code == 200, "清空歷史記錄失敗"

    # 確認歷史記錄已清空
    get_response = requests.get(get_url)
    history = get_response.json()

    print(f"清空後歷史記錄數量: {len(history)}")
    assert len(history) == 0, "歷史記錄未清空"

    return history


def run_all_tests():
    """運行所有測試"""
    print("開始RAG應用功能測試...")

    try:
        create_test_file()
        upload_result = test_upload_file()
        chat_results = test_chat()
        history_result = test_history()

        print("\n===== 測試完成 =====")
        print("所有功能測試通過！")

    except AssertionError as e:
        print(f"\n測試失敗: {str(e)}")
    except Exception as e:
        print(f"\n測試過程中出現錯誤: {str(e)}")
    finally:
        # 清理測試文件
        if os.path.exists(TEST_FILE_PATH):
            os.remove(TEST_FILE_PATH)
            print(f"測試文件已刪除: {TEST_FILE_PATH}")


if __name__ == "__main__":
    run_all_tests()
