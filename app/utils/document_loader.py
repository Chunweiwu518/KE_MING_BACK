def get_text_splitter(chunk_size=500, chunk_overlap=100):
    """獲取更智能的文本分割器"""
    # 使用遞歸字符文本分割器，能更智能地處理文本結構
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )


def preprocess_text(text):
    """文本預處理，增強語義內容"""
    # 處理常見的產品型號格式，使其標準化
    processed = re.sub(r"(LED|EDS|HK)-(\d+)", r"\1-\2", text)

    # 處理尺寸、規格等關鍵信息的格式
    processed = re.sub(r"(\d+)x(\d+)", r"\1×\2", processed)
    processed = re.sub(r"(\d+)mAh", r"\1 mAh", processed)

    # 確保常見詞匯的一致性
    processed = processed.replace("照明燈", "LED燈")
    processed = processed.replace("電池容量", "電池")

    return processed
