import torch
import torch.nn as nn
from tqdm import tqdm
import random
from torch.nn.utils.rnn import pack_padded_sequence
from src.next_token_dataset import TOKENIZER
import evaluate

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
HIDDEN_DIM = 128

class LstmModel(nn.Module):
    """
    Однонаправленная LSTM модель для предсказания следующего токена
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.rnn = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.to(DEVICE)

    def forward(self, contexts, lengths):

        emb = self.embedding(contexts)
        
        packed_emb = pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.rnn(packed_emb)
        
        last_hidden = hidden[-1]
        
        logits = self.fc(last_hidden)
        
        return logits

    def evaluate_model(self, loader, criterion, rouge_metric=None):

        self.eval()
        correct, total = 0, 0
        sum_loss = 0
        
        with torch.no_grad():
            for batch in loader:
                contexts = batch['contexts']
                lengths = batch['lengths']
                target_tokens = batch['tokens']
                
                logits = self(contexts, lengths)
                loss = criterion(logits, target_tokens)
                
                preds = torch.argmax(logits, dim=1)
                correct += (preds == target_tokens).sum().item()
                total += target_tokens.size(0)
                sum_loss += loss.item()
        
        avg_loss = sum_loss / len(loader)
        accuracy = correct / total if total > 0 else 0
        
        rouge_score = None
        if rouge_metric is not None:
            predictions = []
            references = []
            
            with torch.no_grad():
                for batch in loader:
                    contexts = batch['contexts']
                    lengths = batch['lengths']
                    target_tokens = batch['tokens']
                    
                    logits = self(contexts, lengths)
                    preds = torch.argmax(logits, dim=1)
                    
                    for pred, target in zip(preds, target_tokens):
                        try:
                            pred_text = TOKENIZER.decode([pred.item()]) if pred.item() != -1 else ""
                            target_text = TOKENIZER.decode([target.item()]) if target.item() != -1 else ""
                            predictions.append(pred_text)
                            references.append(target_text)
                        except:
                            predictions.append("")
                            references.append("")
            
            try:
                rouge_score = rouge_metric.compute(
                    predictions=predictions[:100],
                    references=references[:100],
                    use_stemmer=True
                )
            except Exception as e:
                print(f"Ошибка при расчете ROUGE: {e}")
                rouge_score = {"rouge1": 0.0}
                
        return avg_loss, accuracy, rouge_score

    def train_model(self, n_epochs, learning_rate, train_loader, val_loader, rouge_metric=None):

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_rouge_scores = []
        
        for epoch in range(n_epochs):
            self.train()
            train_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
            for batch in progress_bar:
                contexts = batch['contexts']
                lengths = batch['lengths']
                target_tokens = batch['tokens']
                
                optimizer.zero_grad()
                
                logits = self(contexts, lengths)
                loss = criterion(logits, target_tokens)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            val_loss, val_acc, val_rouge = self.evaluate_model(
                val_loader, criterion, rouge_metric
            )
            
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            val_rouge_scores.append(val_rouge)
            
            print(f"Epoch {epoch+1}/{n_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")
            
            if val_rouge:
                print("  ROUGE Metrics:")
                for metric, score in val_rouge.items():
                    print(f"    {metric}: {score:.4f}")
            
            print()
        
        return train_losses, val_losses, val_accuracies, val_rouge_scores

    def generate_tokens(self, context_ids, max_length=20, temperature=1.0):
        
        self.eval()
        generated_tokens = []
        current_context = context_ids.clone().to(DEVICE)
        
        with torch.no_grad():
            for _ in range(max_length):
                context_length = current_context.size(0)
                
                logits = self(current_context.unsqueeze(0), [context_length])
                
                logits = logits / temperature
                
                probs = torch.softmax(logits, dim=-1)
                
                next_token = torch.multinomial(probs, 1).squeeze().item()
                
                generated_tokens.append(next_token)
                
                current_context = torch.cat([current_context, torch.tensor([next_token], device=DEVICE)])
                
                if next_token == 0:
                    break
        
        return generated_tokens

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=DEVICE))