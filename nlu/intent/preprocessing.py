import re
import unicodedata


def normalize(text: str | None) -> str | None:
    """Normalize text by removing accents and converting to lowercase."""
    if text is None:
        return None
    text = text.lower()
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r"[^0-9a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
                  r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữ"
                  r"ỳýỵỷỹđ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
