import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 256
MIN_LEN = 4
MAX_LEN = 80


TOKENIZER = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(line: str, num_targets: int = 1) -> tuple[list[int], int]:
    """
    Функция для токенизирования предложения

    Аргументы:
        line (str): предложение для токенизации
        num_targets (int, optional): количество токенов, которые будут использоваться как цель предсказания.

    Возвращает:
        tuple[list[int], int]: кортеж, первый элемент - токенизированный контекст, второй - токенизированная цель.
    """
    assert (num_targets >= 0)

    token_ids = TOKENIZER.encode(
        line, add_special_tokens=False, max_length=MAX_LEN, truncation=True)
    tok_len = len(token_ids)

    if (tok_len < MIN_LEN) or (tok_len > MAX_LEN) or (num_targets >= tok_len):
        return None

    tail = min(len(token_ids), MAX_LEN) - num_targets
    context = token_ids[:tail]
    target = token_ids[tail:tail+num_targets] if num_targets > 0 else [-1]
    
    # Возвращаем первый токен из target если num_targets = 1
    if num_targets == 1:
        target = target[0] if target else -1
    
    return context, target

def collate(batch) -> dict:
    """
    Пользовательская функция для объединения в батч

    Аргументы:
        batch: список из датасета

    Возвращает:
        dict: словарь с объединенными тензорами для батча
    """
    contexts = [item['context'] for item in batch]
    tokens = [item['token'] for item in batch]
    lengths = [len(ctx) for ctx in contexts]
    padded_contexts = pad_sequence(contexts, batch_first=True, padding_value=0)

    return {
        'contexts': padded_contexts.to(DEVICE),
        'lengths': lengths,
        'tokens': torch.tensor(tokens, dtype=torch.long).to(DEVICE)
    }

class TextDataset(Dataset):
    """
    Датасет PyTorch для задачи автодополнения текста
    """

    def __init__(self, texts, num_targets: int = 1):
        self.samples = []

        if isinstance(texts, str):
            with open(texts, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]

        for line in tqdm(texts, desc="Токенизация текстов"):
            ret = tokenize(line, num_targets)
            if ret:
                context, target = ret
                self.samples.append({
                    'context': torch.tensor(context, dtype=torch.long),
                    'token': target
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


