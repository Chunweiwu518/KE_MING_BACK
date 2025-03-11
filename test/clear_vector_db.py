import os
import shutil
import time


def clear_vector_db():
    """清除向量資料庫目錄"""
    # 獲取當前工作目錄
    current_dir = os.getcwd()

    # 需要清除的所有可能的向量庫目錄
    vector_db_dirs = [
        os.path.join(current_dir, "vector_db"),
        os.path.join(current_dir, "chroma_db"),
        os.path.join(current_dir, "chroma_new"),
    ]

    for db_path in vector_db_dirs:
        if os.path.exists(db_path):
            print(f"正在刪除向量資料庫目錄: {db_path}")
            try:
                # 嘗試直接刪除目錄
                shutil.rmtree(db_path, ignore_errors=True)
                print(f"向量資料庫已刪除: {db_path}")
            except Exception as e:
                print(f"無法直接刪除目錄 {db_path}: {str(e)}")

                # 如果直接刪除失敗，嘗試重命名後刪除
                try:
                    temp_dir = db_path + "_old_" + str(int(time.time()))
                    print(f"嘗試重命名目錄 {db_path} -> {temp_dir}")
                    os.rename(db_path, temp_dir)

                    # 重命名成功後嘗試刪除
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        print(f"成功刪除重命名後的目錄: {temp_dir}")
                    except Exception as e2:
                        print(
                            f"無法刪除重命名後的目錄 {temp_dir}，可能需要稍後手動刪除: {str(e2)}"
                        )
                except Exception as rename_err:
                    print(f"重命名目錄失敗: {str(rename_err)}")

                    # 嘗試逐個刪除文件
                    try:
                        print("嘗試逐個刪除文件...")
                        for root, dirs, files in os.walk(db_path):
                            for f in files:
                                try:
                                    file_path = os.path.join(root, f)
                                    os.remove(file_path)
                                    print(f"已刪除: {file_path}")
                                except Exception as file_err:
                                    print(f"無法刪除文件: {file_path}: {str(file_err)}")
                    except Exception as walk_err:
                        print(f"遍歷目錄失敗: {str(walk_err)}")

    # 清空上傳目錄
    upload_dir = os.path.join(current_dir, "uploads")
    if os.path.exists(upload_dir):
        print(f"正在清空上傳目錄: {upload_dir}")
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print(f"已刪除上傳文件: {filename}")
                except Exception as e:
                    print(f"無法刪除文件 {filename}: {str(e)}")

    # 創建新的空目錄
    for db_path in vector_db_dirs:
        os.makedirs(db_path, exist_ok=True)
        print(f"已創建新的空向量庫目錄: {db_path}")

    print("已清除所有向量資料庫。重新運行應用程序將使用新的資料庫。")


if __name__ == "__main__":
    clear_vector_db()
