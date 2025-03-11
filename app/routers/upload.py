import glob
import os
import shutil
import time
import uuid
from typing import Dict

from app.rag.document import process_document, remove_document
from app.utils.vector_store import get_vector_store, reset_vector_store
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/api", tags=["upload"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 保存上傳文件的映射關係
file_mappings: Dict[str, str] = {}  # 顯示名稱 -> 實際文件名


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 創建上傳目錄
        upload_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        # 生成唯一文件名
        file_extension = os.path.splitext(file.filename)[1]
        new_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(upload_dir, new_filename)

        # 保存文件
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        # 保存文件映射關係
        file_mappings[file.filename] = new_filename

        # 處理文件並添加到向量數據庫
        print(f"開始處理文件: {file_path}")
        success = await process_document(file_path)

        if not success:
            # 如果處理失敗，清理文件
            os.remove(file_path)
            del file_mappings[file.filename]
            raise HTTPException(status_code=500, detail="文件處理失敗")

        print(f"文件處理完成: {file_path}")

        return {
            "status": "success",
            "filename": file.filename,  # 返回原始文件名
            "message": "文件已上傳並處理完成",
        }
    except Exception as e:
        print(f"上傳文件時出錯: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上傳失敗: {str(e)}")


@router.delete("/files/{filename}")
async def delete_file(filename: str):
    """刪除文件"""
    try:
        if filename not in file_mappings:
            raise HTTPException(status_code=404, detail="找不到指定的文件")

        actual_filename = file_mappings[filename]
        file_path = os.path.join(os.getcwd(), "uploads", actual_filename)

        # 從向量數據庫中移除文件
        await remove_document(file_path)

        # 刪除實際文件
        if os.path.exists(file_path):
            os.remove(file_path)

        # 移除映射關係
        del file_mappings[filename]

        return {"status": "success", "message": "文件已刪除"}
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"刪除文件時出錯: {str(e)}")
        raise HTTPException(status_code=500, detail=f"刪除文件失敗: {str(e)}")


@router.delete("/files/clear")
async def clear_all_files():
    """清空所有文件"""
    try:
        # 清空向量數據庫
        vector_store = get_vector_store()
        vector_store.delete(where={})  # 刪除所有文檔

        # 刪除所有實際文件
        upload_dir = os.path.join(os.getcwd(), "uploads")
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # 清空文件映射
        file_mappings.clear()

        return {"status": "success", "message": "所有文件已清空"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空文件失敗: {str(e)}")


@router.delete("/vector-store/clear")
async def clear_vector_store():
    """徹底清空向量知識庫"""
    try:
        # 1. 關閉現有連接
        reset_vector_store()

        # 2. 清除文件
        vector_dir = os.path.join(os.getcwd(), "chroma_new")
        if os.path.exists(vector_dir):
            # 確保所有文件句柄都已釋放
            time.sleep(1)

            try:
                # 直接使用 shutil.rmtree 清除整個目錄
                shutil.rmtree(vector_dir, ignore_errors=True)
                os.makedirs(vector_dir, exist_ok=True)
                print("已清空向量庫目錄")
            except Exception as e:
                print(f"清除目錄時出錯: {str(e)}")
                # 如果直接刪除失敗，嘗試逐個刪除
                for root, dirs, files in os.walk(vector_dir, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except Exception as e:
                            print(f"刪除文件失敗: {str(e)}")
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except Exception as e:
                            print(f"刪除目錄失敗: {str(e)}")

        # 3. 清空文件映射
        file_mappings.clear()

        # 4. 強制重新初始化向量庫
        get_vector_store(force_new=True)

        return {
            "status": "success",
            "message": "向量知識庫已清空",
            "timestamp": time.time(),  # 添加時間戳以便前端判斷更新
        }
    except Exception as e:
        print(f"清空向量知識庫時出錯: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清空向量知識庫失敗: {str(e)}")


@router.post("/upload/folder")
async def upload_folder(folder_path: str = None):
    """上傳本地資料夾中的所有支持的文件"""
    try:
        if not folder_path or not os.path.exists(folder_path):
            raise HTTPException(status_code=400, detail="請提供有效的資料夾路徑")

        # 支持的文件類型
        supported_extensions = [".txt", ".pdf", ".docx"]

        # 遞歸搜索所有支持的文件
        all_files = []
        for ext in supported_extensions:
            files = glob.glob(os.path.join(folder_path, f"**/*{ext}"), recursive=True)
            all_files.extend(files)

        if not all_files:
            raise HTTPException(status_code=400, detail="資料夾中沒有找到支持的文件")

        # 處理每個文件
        processed_files = []
        failed_files = []

        for file_path in all_files:
            try:
                # 複製文件到上傳目錄
                file_name = os.path.basename(file_path)
                new_filename = f"{uuid.uuid4()}{os.path.splitext(file_name)[1]}"
                dest_path = os.path.join(os.getcwd(), "uploads", new_filename)

                # 複製文件
                import shutil

                shutil.copy2(file_path, dest_path)

                # 保存文件映射
                file_mappings[file_name] = new_filename

                # 處理文件
                success = await process_document(dest_path)

                if success:
                    processed_files.append(file_name)
                else:
                    failed_files.append(file_name)
                    # 清理失敗的文件
                    os.remove(dest_path)
                    del file_mappings[file_name]

            except Exception as e:
                print(f"處理文件 {file_path} 時出錯: {str(e)}")
                failed_files.append(file_name)

        return {
            "status": "success",
            "processed_files": processed_files,
            "failed_files": failed_files,
            "message": f"成功處理 {len(processed_files)} 個文件，失敗 {len(failed_files)} 個",
        }

    except Exception as e:
        print(f"處理資料夾時出錯: {str(e)}")
        raise HTTPException(status_code=500, detail=f"處理資料夾失敗: {str(e)}")


@router.get("/vector-store/stats")
async def get_vector_store_stats():
    """獲取向量知識庫統計信息"""
    try:
        # 檢查文件系統狀態
        vector_dir = os.path.join(os.getcwd(), "chroma_new")
        dir_empty = True

        if os.path.exists(vector_dir):
            for root, _, files in os.walk(vector_dir):
                if files and any(not f.startswith(".") for f in files):
                    dir_empty = False
                    break

        # 如果目錄為空，直接返回空狀態
        if dir_empty:
            return {
                "status": "success",
                "message": "向量庫是空的",
                "total_chunks": 0,
                "unique_files": 0,
                "files": [],
                "is_empty": True,
            }

        # 否則獲取詳細統計
        vector_store = get_vector_store(force_new=True)
        all_docs = vector_store.get()
        metadatas = all_docs.get("metadatas", [])

        unique_files = {
            meta["filename"] for meta in metadatas if meta and "filename" in meta
        }

        return {
            "status": "success",
            "total_chunks": len(metadatas),
            "unique_files": len(unique_files),
            "files": list(unique_files),
            "is_empty": len(metadatas) == 0,
        }
    except Exception as e:
        print(f"獲取統計信息時出錯: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "total_chunks": 0,
            "unique_files": 0,
            "files": [],
            "is_empty": True,
        }


@router.post("/vector-store/hard-reset")
async def hard_reset_vector_store():
    """徹底重置向量庫（需要重啟服務）"""
    try:
        # 關閉當前所有連接
        reset_vector_store()

        # 寫入一個信號文件
        with open("RESET_DB", "w") as f:
            f.write(str(time.time()))

        # 返回需要重啟的提示
        return {
            "status": "success",
            "message": "已標記知識庫需要重置，請重啟服務以完成重置",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置標記失敗: {str(e)}")
