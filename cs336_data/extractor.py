from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

def html2text(raw_html: bytes)-> str:
    """extracts text from a byte string containing raw html
    """
    try:
        encoding = detect_encoding(raw_html)
        decoded = raw_html.decode(encoding, errors="replace")
        return extract_plain_text(decoded)
    except Exception:
        return None

def extract_texts_from_warc(warc_file_path: str):
    """Reads WARC file and yields plain text extracted from HTML"""
    with open(warc_file_path, "rb") as stream:
        # 使用ArchiveIterator遍历WARC文件中的记录
        for record in ArchiveIterator(stream):
            # 只处理response类型的记录（网页内容）
            if record.record_type == WarcRecordType.response:
                # 检查内容类型是否包含HTML
                content_type = record.http_headers.get('content-type', '')
                if 'text/html' in content_type.lower():
                    # 读取记录内容
                    content = record.reader.read()
                    # 使用html2text函数转换为纯文本
                    plain_text = html2text(content)
                    # 如果转换成功且文本不为空，则yield结果
                    if plain_text and plain_text.strip():
                        yield plain_text

def extract_wet_texts_from_warc_file(warc_wet_file_path: str):
    """Reads WET file and yields plain text extracted from HTML"""
    with open(warc_wet_file_path, "rb") as stream:
        # 使用ArchiveIterator遍历WET文件中的记录
        for record in ArchiveIterator(stream):
            # WET文件中包含conversion类型的记录（已提取的文本）
            if record.record_type == WarcRecordType.conversion:
                # 直接读取记录内容，因为WET文件中已经是纯文本
                content = record.reader.read()
                try:
                    # 解码为字符串
                    text = content.decode('utf-8', errors='replace')
                    # 过滤掉空白内容
                    if text and text.strip():
                        yield text.strip()
                except Exception:
                    # 如果解码失败，跳过这条记录
                    continue


if __name__ == "__main__":
    first_k = 1
    i = 0
    for item in extract_texts_from_warc("../data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.gz"):
        if i < first_k:
            # 写入文件
            with open("warc_content.txt", "a") as f:
                f.write(item)
        else:
            break
        i += 1
    print("warc_content.txt done")
    i = 0
    for item in extract_wet_texts_from_warc_file("../data/CC/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz"):
        if i < first_k:
            # 写入文件
            with open("warc_wet_content.txt", "a") as f:
                f.write(item)
        else:
            break
        i += 1
    
    print("warc_wet_content.txt done")

        


