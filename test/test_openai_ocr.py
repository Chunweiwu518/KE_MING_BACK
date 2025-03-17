import os
import fitz  # PyMuPDF
import io
from PIL import Image
import base64
from openai import OpenAI
import json
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 初始化 OpenAI 客戶端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_page_as_image(pdf_path, page_num=0, zoom=2):
    """
    從PDF中提取指定頁面並轉換為圖像
    
    Args:
        pdf_path (str): PDF文件路徑
        page_num (int): 頁碼 (從0開始)
        zoom (int): 放大倍數以提高圖像質量
        
    Returns:
        PIL.Image: 提取的頁面圖像
    """
    try:
        # 打開PDF文件
        pdf_document = fitz.open(pdf_path)
        
        # 檢查頁碼是否有效
        if page_num >= len(pdf_document):
            raise ValueError(f"頁碼 {page_num} 超出範圍，PDF 總共有 {len(pdf_document)} 頁")
        
        # 獲取指定頁面
        page = pdf_document[page_num]
        
        # 將頁面渲染為圖像
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # 將像素數據轉換為PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        return img
    
    except Exception as e:
        print(f"提取PDF頁面時出錯: {str(e)}")
        raise

def encode_image_to_base64(image):
    """
    將PIL圖像轉換為base64編碼
    
    Args:
        image (PIL.Image): 輸入圖像
        
    Returns:
        str: base64編碼的圖像字符串
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def ocr_with_openai(image, prompt="請讀取圖像中的所有文字"):
    """
    使用OpenAI Vision API進行OCR
    
    Args:
        image (PIL.Image): 要進行OCR的圖像
        prompt (str): 提示OpenAI模型的文本
        
    Returns:
        str: OpenAI返回的文本
    """
    # 編碼圖像
    base64_image = encode_image_to_base64(image)
    
    # 設置請求
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    
    # 返回結果
    return response.choices[0].message.content

def main():
    # 設置PDF路徑
    pdf_path = r"D:\KE_MING_RAG\backend\test\盤商目錄202309102-1-4.pdf"
    
    # 提取第一頁圖像
    print(f"正在從 {pdf_path} 提取第1頁...")
    image = extract_page_as_image(pdf_path, page_num=0)
    
    # 儲存圖像以供檢查
    output_path = r"D:\KE_MING_RAG\backend\test\output_page.png"
    image.save(output_path)
    print(f"頁面已保存為圖像: {output_path}")
    
    # 使用OpenAI進行OCR
    print("正在使用OpenAI進行OCR...")
    extracted_text = ocr_with_openai(image, "請詳細讀取並列出圖像中的所有文字內容，保持原有格式和布局。如有表格，請盡量保持表格結構。")
    
    # 保存OCR結果
    output_text_path = r"D:\KE_MING_RAG\backend\test\ocr_result.txt"
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    print(f"OCR結果已保存到: {output_text_path}")
    print("\nOCR結果預覽:")
    print("-" * 50)
    print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
    print("-" * 50)

if __name__ == "__main__":
    main() 