import re

def clean_text(text: str) -> str:

    text = text.lower()
    
    text = re.sub(r'@\w+', '', text)
    
    text = re.sub(r'http\S+', '', text)
    
    text = re.sub(r'[^\w\s\']', ' ', text)

    text = re.sub(r'\.\s*\.\s*\.+', ' ', text)
    text = re.sub(r'(\.\s){2,}\.?', ' ', text)
    text = re.sub(r'\.{2,}', ' ', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = text.strip()
    
    return text


            

print(f"Очистка завершена. Результат записан в файл")