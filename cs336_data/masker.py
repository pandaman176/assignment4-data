import re
from cs336_data.extractor import extract_texts_from_warc
from cs336_data.common import DATA_DIR

email_pattern = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
phone_pattern = re.compile(r'''
    (                           # 整体捕获
        \b\d{10}\b              # 纯数字：2831823829
        |
        \(\d{3}\)[\s-]?\d{3}[\s-]?\d{4}    # 带括号 (283) 182-3829 或 (283) 182 3829
        |
        \d{3}[-.\s]\d{3}[-.\s]\d{4}        # 中间有分隔符：283-182-3829 / 283.182.3829 / 283 182 3829
    )
''', re.VERBOSE)
ip_pattern = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')

def mask_emails(text: str) -> str:
    """
    take a unicode string and return a new string with all email addresses masked
    """
    return re.subn(email_pattern, "|||EMAIL_ADDRESS|||", text)

def mask_phone_numbers(text: str) -> str:
    """
    take a unicode string and return a new string with all phone numbers masked
    """
    return re.subn(phone_pattern, "|||PHONE_NUMBER|||", text)

def mask_ips(text: str) -> str:
    """
    take a unicode string and return a new string with all IP addresses masked
    """
    return re.subn(ip_pattern, "|||IP_ADDRESS|||", text)

def mask_all(text: str) -> str:
    """
    take a unicode string and return a new string with all sensitive information masked
    """
    text, num_emails = mask_emails(text)
    text, num_phone_numbers = mask_phone_numbers(text)
    text, num_ips = mask_ips(text)
    return {
        "text": text,
        "num_emails": num_emails,
        "num_phone_numbers": num_phone_numbers,
        "num_ips": num_ips,
    }

if __name__ == "__main__":
    i = 0
    for item in extract_texts_from_warc(DATA_DIR / "CC" / "CC-MAIN-20250417135010-20250417165010-00065.warc.gz"):
        result = mask_all(item)
        if result["num_emails"] > 0 and result["num_phone_numbers"] > 0 and result["num_ips"] > 0:
            print(result["text"])
            print("-"*100)
            print("num_emails:", result["num_emails"])
            print("num_phone_numbers:", result["num_phone_numbers"])
            print("num_ips:", result["num_ips"])
            print("="*100)
            i += 1
            if i > 5:
                break