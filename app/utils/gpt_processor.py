import json
import os
from typing import List

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain.schema import Document
from openai import OpenAI

load_dotenv()


class GPTDocumentProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # 確保靜態文件目錄存在
        self.static_dir = os.path.join(os.getcwd(), "static", "images", "products")
        os.makedirs(self.static_dir, exist_ok=True)

    def extract_images(self):
        """從 PDF 提取圖片"""
        images = {}
        pdf = fitz.open(self.pdf_path)

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]

                # 將圖片保存到指定目錄
                image_filename = f"product_{page_num + 1}_{img_index + 1}.png"
                image_path = os.path.join(self.static_dir, image_filename)

                # 保存圖片
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)

                # 記錄圖片路徑和頁碼
                images[f"page_{page_num + 1}_{img_index + 1}"] = {
                    "path": f"/images/products/{image_filename}",
                    "page": page_num + 1,
                }

                print(f"已提取圖片: {image_filename}, 儲存路徑: {image_path}")

        return images

    def process(self) -> list[Document]:
        """處理 PDF 文件"""
        try:
            # 先上傳文件
            with open(self.pdf_path, "rb") as file:
                response = self.client.files.create(file=file, purpose="user_data")
                file_id = response.id

            # 使用文件 ID 進行處理
            text = """請詳細描述這個產品目錄中的所有產品資訊，並標註每個產品在PDF中的頁碼，格式如下：

### [產品型號] (第X頁)
- **產品名稱**: [名稱]
- **產品描述**: [描述，包含配件和電池資訊]
- **尺寸規格**: [尺寸]
- **裝箱數量**: [數量]
- **建議售價**: [價格]

請確保每個產品資訊都標註所在頁碼。"""
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "file", "file": {"file_id": file_id}},
                            {"type": "text", "text": text},
                        ],
                    }
                ],
            )

            # 處理完成後刪除上傳的文件
            try:
                self.client.files.delete(file_id)
            except Exception as e:
                print(f"刪除文件時出錯: {str(e)}")

            # 提取圖片
            images = self.extract_images()

            # 將圖片信息轉換為字符串格式
            images_str = {}
            for key, value in images.items():
                images_str[key] = f"{value['path']}|{value['page']}"

            # 將圖片資訊加入 metadata
            doc = Document(
                page_content=response.choices[0].message.content,
                metadata={
                    "source": self.pdf_path,
                    "filename": os.path.basename(self.pdf_path),
                    "extraction_method": "gpt4o",
                    "images": json.dumps(images_str),  # 將字典轉換為 JSON 字符串
                },
            )

            return [doc]

        except Exception as e:
            print(f"GPT-4o 處理時出錯: {str(e)}")
            raise


def process_pdf_with_gpt(pdf_path: str) -> List[Document]:
    """使用 GPT-4o 處理 PDF 文件的便捷函數"""
    processor = GPTDocumentProcessor(pdf_path)
    return processor.process()
