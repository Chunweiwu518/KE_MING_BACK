import os
import time
import json
import requests

BASE_URL = "http://localhost:8000/api"
TEST_DATA_FILE_PATH = "test_product_data.json"  # 產品測試資料檔案

def create_test_product_data():
    """創建測試產品數據檔案"""
    test_data = {
        "products": [
            {
                "id": "HK-2189",
                "name": "電池式磁吸軟管工作燈",
                "description": "便攜式LED工作燈，配備磁吸軟管，可靈活調整照明角度。內置充電電池，續航時間長達8小時。",
                "price": 1299,
                "category": "工作燈",
                "specifications": {
                    "續航時間": "8小時",
                    "亮度": "1000流明",
                    "防水等級": "IP65",
                    "材質": "鋁合金+矽膠"
                }
            },
            {
                "id": "TL-4523",
                "name": "專業級LED頭燈",
                "description": "高亮度LED頭燈，適合戶外探險和夜間工作。可調節頭帶，舒適配戴。三種亮度模式，滿足不同場景需求。",
                "price": 899,
                "category": "頭燈",
                "specifications": {
                    "續航時間": "10小時",
                    "亮度": "1500流明",
                    "防水等級": "IP67",
                    "材質": "ABS塑料+尼龍"
                }
            },
            {
                "id": "WL-7732",
                "name": "太陽能戶外照明系統",
                "description": "太陽能供電的戶外照明系統，配備高效太陽能板和大容量電池。三個可調節照明頭，廣泛照明範圍。",
                "price": 2499,
                "category": "戶外照明",
                "specifications": {
                    "續航時間": "24小時",
                    "亮度": "2200流明",
                    "防水等級": "IP68",
                    "材質": "鋁合金+強化玻璃"
                }
            }
        ]
    }

    with open(TEST_DATA_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"測試產品數據文件已創建: {TEST_DATA_FILE_PATH}")
    return TEST_DATA_FILE_PATH

def test_upload_product_data():
    """測試上傳產品數據檔案"""
    print("\n===== 測試上傳產品數據檔案 =====")

    url = f"{BASE_URL}/upload"

    with open(TEST_DATA_FILE_PATH, "rb") as f:
        files = {"file": (os.path.basename(TEST_DATA_FILE_PATH), f, "application/json")}
        response = requests.post(url, files=files)

    print(f"狀態碼: {response.status_code}")
    print(f"響應: {response.json()}")

    assert response.status_code == 200, "檔案上傳失敗"
    assert response.json()["status"] == "success", "上傳狀態不是success"

    # 等待檔案被處理和索引
    print("正在等待檔案處理和索引完成...")

    # 嘗試查詢檔案中的內容，直到找到結果或超時
    max_attempts = 10
    attempt = 0
    indexed = False

    while attempt < max_attempts and not indexed:
        # 嘗試查詢產品資訊
        test_query = "HK-2189"
        chat_url = f"{BASE_URL}/chat"
        query_response = requests.post(
            chat_url, json={"query": test_query, "history": []}
        )

        if query_response.status_code == 200:
            result = query_response.json()
            if "電池式磁吸軟管工作燈" in result.get("answer", ""):
                print("檔案已成功索引!")
                indexed = True
                break

        # 如果還沒索引，等待一會再試
        attempt += 1
        wait_time = 2 * attempt  # 漸進增加等待時間
        print(f"檔案還在處理中... 等待 {wait_time} 秒 (嘗試 {attempt}/{max_attempts})")
        time.sleep(wait_time)

    if not indexed:
        print("警告: 在允許的嘗試次數內未檢測到檔案被索引")
    
    return response.json()

def test_product_query():
    """測試產品資訊查詢"""
    print("\n===== 測試產品資訊查詢 =====")

    url = f"{BASE_URL}/chat"

    # 準備測試問題
    test_questions = ["HK-2189的詳細資訊是什麼?", "TL-4523是什麼產品?", "哪些產品屬於工作燈類別?"]

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

def test_upload_pdf_catalog():
    """測試上傳PDF產品目錄"""
    print("\n===== 測試上傳PDF產品目錄 =====")
    
    pdf_path = "test/盤商目錄202309102-1-4.pdf"  # 根據實際路徑調整
    
    # 確認文件存在
    if not os.path.exists(pdf_path):
        print(f"警告: PDF文件 {pdf_path} 不存在，跳過測試")
        return "PDF文件不存在，測試跳過"
    
    with open(pdf_path, "rb") as f:
        files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
        response = requests.post(f"{BASE_URL}/upload", files=files)
    
    print(f"狀態碼: {response.status_code}")
    print(f"響應: {response.json()}")
    
    assert response.status_code == 200, "PDF檔案上傳失敗"
    assert response.json()["status"] == "success", "上傳狀態不是success"
    
    # 等待索引完成（PDF較大可能需要更長時間）
    print("正在等待PDF文件處理和索引完成...")
    
    # 等待PDF處理完成並被索引
    max_attempts = 15
    attempt = 0
    indexed = False
    
    while attempt < max_attempts and not indexed:
        # 嘗試查詢PDF中可能存在的內容
        test_query = "盤商目錄中的產品"
        chat_url = f"{BASE_URL}/chat"
        query_response = requests.post(
            chat_url, json={"query": test_query, "history": []}
        )
        
        if query_response.status_code == 200:
            result = query_response.json()
            answer = result.get("answer", "")
            print(f"測試查詢回答: {answer[:100]}...")
            
            # 檢查回答是否有實質內容
            if len(answer) > 100 and "沒有找到" not in answer and "無法找到" not in answer:
                print("PDF文件已成功索引!")
                indexed = True
                break
        
        # 如果還沒索引，等待一會再試
        attempt += 1
        wait_time = 5 * attempt  # 漸進增加等待時間
        print(f"PDF還在處理中... 等待 {wait_time} 秒 (嘗試 {attempt}/{max_attempts})")
        time.sleep(wait_time)
    
    if not indexed:
        print("警告: 在允許的嘗試次數內未檢測到PDF被索引")
    
    # 測試PDF內容查詢
    print("\n===== 測試PDF內容查詢 =====")
    
    url = f"{BASE_URL}/chat"
    # 根據PDF內容調整查詢問題
    test_questions = ["盤商目錄中有哪些產品類別?", "盤商目錄中的聯絡方式是什麼?"]
    
    results = []
    for question in test_questions:
        print(f"\n問題: {question}")
        
        payload = {"query": question, "history": []}
        response = requests.post(url, json=payload)
        
        print(f"狀態碼: {response.status_code}")
        result = response.json()
        print(f"回答: {result['answer']}")
        print(f"來源數量: {len(result['sources'])}")
        
        assert response.status_code == 200, "PDF內容查詢失敗"
        results.append(result)
    
    return "PDF測試完成"

def run_product_data_tests():
    """運行產品數據測試"""
    print("開始產品數據RAG應用功能測試...")

    try:
        file_path = create_test_product_data()
        upload_result = test_upload_product_data()
        query_results = test_product_query()
        
        # 添加PDF測試
        print("\n開始測試PDF目錄文件...")
        pdf_result = test_upload_pdf_catalog()

        print("\n===== 測試完成 =====")
        print("產品數據測試通過！")

        # 返回測試結果
        return {
            "upload_result": upload_result,
            "query_results": query_results,
            "pdf_result": pdf_result
        }

    except AssertionError as e:
        print(f"\n測試失敗: {str(e)}")
        raise
    except Exception as e:
        print(f"\n測試過程中出現錯誤: {str(e)}")
        raise
    finally:
        # 清理測試文件
        if os.path.exists(TEST_DATA_FILE_PATH):
            os.remove(TEST_DATA_FILE_PATH)
            print(f"測試文件已刪除: {TEST_DATA_FILE_PATH}")

if __name__ == "__main__":
    run_product_data_tests() 