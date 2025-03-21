import os
import base64
import requests
import json
from pathlib import Path
import argparse
from pdf2image import convert_from_path
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API設定
API_URL = "https://api.openai.com/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

def convert_pdf_to_images(pdf_path):
    """將PDF轉換為圖像列表"""
    print(f"轉換PDF: {pdf_path}")
    try:
        images = convert_from_path(pdf_path)
        print(f"成功轉換 {len(images)} 頁")
        return images
    except Exception as e:
        print(f"PDF轉換錯誤: {e}")
        return []

def encode_image_to_base64(image):
    """將圖像編碼為base64格式"""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def analyze_pdf_with_openai(pdf_path, max_pages=3):
    """使用OpenAI分析PDF內容"""
    images = convert_pdf_to_images(pdf_path)
    
    if not images:
        return "無法處理PDF檔案"
    
    # 限制頁數以避免超出API限制
    images = images[:max_pages]
    
    # 準備消息內容
    messages = [
        {
            "role": "system",
            "content": "您是一個專業的PDF文件分析助手。請分析以下PDF頁面的內容，提取所有可見文字，表格和重要資訊。"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"這是一個{len(images)}頁的PDF文件。請幫我提取所有可見的文字內容，並且組織成結構化的格式。如果有表格，請嘗試保留其結構。"
                }
            ]
        }
    ]
    
    # 加入圖像
    for i, image in enumerate(images):
        base64_image = encode_image_to_base64(image)
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": "high"
            }
        })
        
        # 加入分隔文字（如果不是最後一頁）
        if i < len(images) - 1:
            messages[1]["content"].append({
                "type": "text",
                "text": f"==== 第 {i+1} 頁結束，第 {i+2} 頁開始 ===="
            })
    
    # 準備API請求
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 4000,
        "temperature": 0.3
    }
    
    try:
        print("正在發送請求到OpenAI API...")
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        
        result = response.json()
        text_content = result["choices"][0]["message"]["content"]
        
        # 計算使用的tokens
        total_tokens = result["usage"]["total_tokens"]
        print(f"處理完成！使用了 {total_tokens} tokens")
        
        return text_content
    
    except Exception as e:
        return f"API請求錯誤: {str(e)}"

def save_result(content, output_path):
    """將結果保存到文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"結果已保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="使用OpenAI API進行PDF OCR分析")
    parser.add_argument("pdf_path", help="PDF檔案路徑")
    parser.add_argument("--output", help="輸出文件路徑", default="output.txt")
    parser.add_argument("--max_pages", type=int, help="最大處理頁數", default=3)
    args = parser.parse_args()
    
    if not Path(args.pdf_path).exists():
        print(f"錯誤: 找不到PDF檔案 '{args.pdf_path}'")
        return
        
    print(f"開始處理PDF: {args.pdf_path}")
    result = analyze_pdf_with_openai(args.pdf_path, args.max_pages)
    save_result(result, args.output)
    print("處理完成！")

if __name__ == "__main__":
    main() 