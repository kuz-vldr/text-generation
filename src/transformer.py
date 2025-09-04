import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm
import numpy as np

class TransformerGenerator:
    def __init__(self, model_name="distilgpt2"):
        """
        Инициализация предобученной модели трансформера для генерации текста
        """
        print(f"Загрузка модели {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            truncation=True, 
            padding_side="left",
            padding="max_length", 
            max_length=80
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = 0 if torch.cuda.is_available() else -1
        
        self.generator = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=32
        )
        
        print(f"Модель {model_name} загружена успешно")

    def generate(self, contexts, max_new_tokens=20):
        
        if not contexts:
            return []
            
        results = []
        
        data = [{"context": context} for context in contexts]
        dataset = Dataset.from_list(data)
        
        print(f"Генерация {len(contexts)} текстов...")
        for result in tqdm(self.generator(KeyDataset(dataset, "context"), max_new_tokens=max_new_tokens)):
            generated_text = result[0]["generated_text"]
            results.append(generated_text)
        
        return results
    
    

    
