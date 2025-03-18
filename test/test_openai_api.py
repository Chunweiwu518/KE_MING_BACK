import os
import sys
import time
import traceback

import openai
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()


def test_openai_api():
    """測試 OpenAI API 連接和功能"""
    print("========== OpenAI API 測試 ==========")

    # 取得 API 金鑰
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("錯誤: 找不到 OPENAI_API_KEY 環境變數")
        return False

    # 設置 API 金鑰
    print(f"使用 API 金鑰: {api_key[:5]}...{api_key[-4:]}")
    openai.api_key = api_key

    # 獲取模型名稱
    embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    chat_model = os.getenv("CHAT_MODEL_NAME", "gpt-4o-mini")

    print(f"嵌入模型: {embedding_model}")
    print(f"聊天模型: {chat_model}")

    try:
        # 測試 1: 文本嵌入
        print("\n--- 測試文本嵌入 ---")
        embedding_text = "這是一個測試文本，用於測試 OpenAI API 連接。"
        print(f"測試文本: {embedding_text}")

        start_time = time.time()
        embedding_response = openai.embeddings.create(
            model=embedding_model, input=embedding_text
        )

        embedding_time = time.time() - start_time

        # 檢查嵌入響應
        embedding_vector = embedding_response.data[0].embedding
        print(f"嵌入維度: {len(embedding_vector)}")
        print(f"嵌入生成時間: {embedding_time:.2f}秒")
        print(f"嵌入示例 (前5個值): {embedding_vector[:5]}")
        print("嵌入測試成功! ✅")

        # 測試 2: 聊天完成
        print("\n--- 測試聊天完成 ---")
        chat_prompt = "你好，請用一句話告訴我今天的天氣如何?"
        print(f"提示: {chat_prompt}")

        start_time = time.time()
        chat_response = openai.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "你是一個有用的助手。"},
                {"role": "user", "content": chat_prompt},
            ],
        )

        chat_time = time.time() - start_time

        # 檢查聊天響應
        chat_text = chat_response.choices[0].message.content
        print(f"回應: {chat_text}")
        print(f"生成時間: {chat_time:.2f}秒")
        print("聊天測試成功! ✅")

        print("\n所有 OpenAI API 測試通過! 🎉")
        return True

    except Exception as e:
        print(f"\n錯誤: {str(e)}")
        print("\n詳細錯誤信息:")
        traceback.print_exc()
        print("\nAPI 測試失敗 ❌")
        return False


if __name__ == "__main__":
    success = test_openai_api()
    sys.exit(0 if success else 1)
