"""
Data loading utilities for Atlas training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from typing import List, Dict, Optional, Union
from pathlib import Path
import random


class LongContextDataset(Dataset):
    """Dataset for long-context language modeling tasks"""
    
    def __init__(self, 
                 texts: List[str], 
                 tokenizer, 
                 max_length: int = 2048,
                 min_length: int = 512,
                 overlap_ratio: float = 0.1):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            overlap_ratio: Ratio of overlap between consecutive sequences
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        self.overlap_ratio = overlap_ratio
        
        self.examples = self._prepare_examples(texts)
    
    def _prepare_examples(self, texts: List[str]) -> List[Dict[str, torch.Tensor]]:
        """Prepare training examples from texts"""
        examples = []
        
        for text in texts:
            # Tokenize the text
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            
            # Handle empty tokens case
            if len(tokens) < 2:
                continue
            
            if len(tokens) < self.min_length:
                # Repeat text to meet minimum length
                repetitions = (self.min_length // len(tokens)) + 1
                extended_text = text + " " + " ".join([text] * repetitions)
                tokens = self.tokenizer.encode(extended_text, add_special_tokens=True)
                if len(tokens) < self.min_length:
                    continue
            
            # Create sliding window examples with proper indexing
            overlap_size = int(self.max_length * self.overlap_ratio)
            step_size = max(1, self.max_length - overlap_size)  # Ensure step_size >= 1
            
            # Fix off-by-one error: ensure we don't go beyond token boundaries
            for i in range(0, len(tokens), step_size):
                end_pos = min(i + self.max_length, len(tokens))
                
                # Ensure we have at least min_length tokens
                if end_pos - i < self.min_length:
                    if i == 0:
                        # If this is the first chunk and it's too small, pad it
                        sequence = tokens[i:end_pos]
                        if len(sequence) >= 2:
                            examples.append({
                                'input_ids': torch.tensor(sequence[:-1], dtype=torch.long),
                                'labels': torch.tensor(sequence[1:], dtype=torch.long)
                            })
                    break
                    
                sequence = tokens[i:end_pos]
                
                # Create input/label pairs, ensuring we have at least 2 tokens
                if len(sequence) >= 2:
                    examples.append({
                        'input_ids': torch.tensor(sequence[:-1], dtype=torch.long),
                        'labels': torch.tensor(sequence[1:], dtype=torch.long)
                    })
                
                # Stop if we've reached the end
                if end_pos >= len(tokens):
                    break
        
        # Ensure we have at least one example
        if len(examples) == 0:
            # Create a fallback example from first text
            if texts:
                fallback_text = "This is a test document. " + texts[0][:100]
                tokens = self.tokenizer.encode(fallback_text, add_special_tokens=True)
                if len(tokens) >= 2:
                    examples.append({
                        'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
                        'labels': torch.tensor(tokens[1:], dtype=torch.long)
                    })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class RecallDataset(Dataset):
    """Dataset for testing recall abilities (inspired by BABIL and needle-in-haystack)"""
    
    def __init__(self, 
                 tokenizer,
                 num_samples: int = 1000,
                 context_length: int = 2048,
                 num_facts: int = 5):
        """
        Args:
            tokenizer: Tokenizer to use
            num_samples: Number of samples to generate
            context_length: Length of context
            num_facts: Number of facts to embed in context
        """
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.context_length = context_length
        self.num_facts = num_facts
        
        self.examples = self._generate_examples()
    
    def _generate_examples(self) -> List[Dict[str, torch.Tensor]]:
        """Generate synthetic recall examples"""
        examples = []
        
        # Template sentences for context
        context_templates = [
            "The weather today is quite pleasant with sunshine.",
            "Many people enjoy reading books in their free time.",
            "Technology continues to advance at a rapid pace.",
            "The city has many beautiful parks and gardens.",
            "Students often study in the library during exams.",
            "Cooking is an essential skill for daily life.",
            "Music has the power to influence emotions greatly.",
            "Exercise is important for maintaining good health.",
            "Travel broadens one's perspective and understanding.",
            "Art galleries showcase creativity and imagination."
        ]
        
        for i in range(self.num_samples):
            # Generate facts to remember
            facts = []
            for j in range(self.num_facts):
                fact_id = random.randint(1000, 9999)
                fact_value = random.randint(100, 999)
                facts.append(f"Fact {fact_id}: The answer is {fact_value}.")
            
            # Create context with embedded facts
            context_sentences = []
            fact_positions = sorted(random.sample(range(20), min(self.num_facts, 20)))
            
            fact_idx = 0
            for pos in range(20):
                if fact_idx < len(fact_positions) and pos == fact_positions[fact_idx]:
                    context_sentences.append(facts[fact_idx])
                    fact_idx += 1
                else:
                    context_sentences.append(random.choice(context_templates))
            
            # Add question at the end
            query_fact = random.choice(facts)
            fact_parts = query_fact.split(": The answer is ")
            question = f"Question: What is the answer for {fact_parts[0]}?"
            answer = fact_parts[1].rstrip('.')
            
            # Combine into full text
            full_text = " ".join(context_sentences) + " " + question + " Answer: " + answer
            
            # Tokenize
            tokens = self.tokenizer.encode(full_text)
            
            # Ensure minimum length
            if len(tokens) < 10:
                # Pad with repeated text if too short
                full_text = full_text + " " + " ".join(context_templates) + " " + question + " Answer: " + answer
                tokens = self.tokenizer.encode(full_text)
            
            # Truncate if too long
            if len(tokens) > self.context_length:
                # Keep the question and answer at the end
                question_tokens = self.tokenizer.encode(" " + question + " Answer: " + answer)
                context_tokens = tokens[:self.context_length - len(question_tokens)]
                tokens = context_tokens + question_tokens
            
            if len(tokens) > 1:
                examples.append({
                    'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
                    'labels': torch.tensor(tokens[1:], dtype=torch.long)
                })
        
        # Ensure we have at least one example
        if len(examples) == 0:
            # Create a simple fallback example
            simple_text = "This is a test. The answer is 42."
            tokens = self.tokenizer.encode(simple_text)
            if len(tokens) > 1:
                examples.append({
                    'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
                    'labels': torch.tensor(tokens[1:], dtype=torch.long)
                })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class BookDataset(Dataset):
    """Dataset for loading book/document data"""
    
    def __init__(self,
                 data_dir: str,
                 tokenizer,
                 max_length: int = 2048,
                 file_extensions: List[str] = ['.txt', '.json']):
        """
        Args:
            data_dir: Directory containing text files
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            file_extensions: File extensions to load
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load all text files
        texts = self._load_texts(file_extensions)
        self.examples = self._prepare_examples(texts)
    
    def _load_texts(self, file_extensions: List[str]) -> List[str]:
        """Load texts from files"""
        texts = []
        
        for ext in file_extensions:
            for file_path in self.data_dir.glob(f"**/*{ext}"):
                try:
                    if ext == '.json':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, dict) and 'text' in data:
                                texts.append(data['text'])
                            elif isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict) and 'text' in item:
                                        texts.append(item['text'])
                                    elif isinstance(item, str):
                                        texts.append(item)
                    else:  # .txt
                        with open(file_path, 'r', encoding='utf-8') as f:
                            texts.append(f.read())
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
        
        return texts
    
    def _prepare_examples(self, texts: List[str]) -> List[Dict[str, torch.Tensor]]:
        """Prepare training examples"""
        examples = []
        
        for text in texts:
            # Split into sentences for better boundary handling
            sentences = text.split('.')
            current_text = ""
            
            for sentence in sentences:
                current_text += sentence + "."
                tokens = self.tokenizer.encode(current_text)
                
                if len(tokens) >= self.max_length:
                    # Create example
                    tokens = tokens[:self.max_length]
                    examples.append({
                        'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
                        'labels': torch.tensor(tokens[1:], dtype=torch.long)
                    })
                    
                    # Reset with some overlap
                    overlap_sentences = sentences[-2:] if len(sentences) >= 2 else sentences
                    current_text = ".".join(overlap_sentences) + "."
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable length sequences"""
    
    if len(batch) == 0:
        return {'input_ids': torch.empty(0), 'labels': torch.empty(0), 'attention_mask': torch.empty(0)}
    
    # Get max length in batch
    max_length = max(len(item['input_ids']) for item in batch)
    
    # Handle empty batch
    if max_length == 0:
        return {'input_ids': torch.empty(0), 'labels': torch.empty(0), 'attention_mask': torch.empty(0)}
    
    # Pad sequences
    input_ids = []
    labels = []
    attention_mask = []
    
    for item in batch:
        input_seq = item['input_ids']
        label_seq = item['labels']
        
        # Skip empty sequences
        if len(input_seq) == 0:
            continue
        
        # Pad sequences
        pad_length = max_length - len(input_seq)
        
        # Pad input_ids and labels with appropriate values
        padded_input = torch.cat([input_seq, torch.zeros(pad_length, dtype=torch.long)])
        padded_labels = torch.cat([label_seq, torch.full((pad_length,), -100, dtype=torch.long)])  # -100 for ignore in loss
        
        # Create attention mask (1 for real tokens, 0 for padding)
        mask = torch.cat([torch.ones(len(input_seq)), torch.zeros(pad_length)])
        
        input_ids.append(padded_input)
        labels.append(padded_labels)
        attention_mask.append(mask)
    
    # Handle case where all sequences were empty
    if len(input_ids) == 0:
        return {'input_ids': torch.empty(0), 'labels': torch.empty(0), 'attention_mask': torch.empty(0)}
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_mask)
    }


def create_dataloaders(config, tokenizer) -> Dict[str, DataLoader]:
    """Create train and validation dataloaders"""
    
    # Get configuration values with safe defaults
    dataset_path = getattr(config.data, 'dataset_path', None) if hasattr(config, 'data') else None
    train_split = getattr(config.data, 'train_split', 0.9) if hasattr(config, 'data') else 0.9
    max_length = getattr(config.data, 'max_length', 512) if hasattr(config, 'data') else 512
    num_workers = getattr(config.data, 'num_workers', 4) if hasattr(config, 'data') else 4
    batch_size = getattr(config.training, 'batch_size', 8) if hasattr(config, 'training') else 8
    
    # Example: Create sample data if no data directory specified
    if not dataset_path or not os.path.exists(dataset_path):
        print("Creating sample dataset...")
        
        # Generate sample texts
        sample_texts = []
        for i in range(100):  # Reduced for faster testing
            text = f"""
            Document {i}: This is a sample document for training the Atlas model. 
            The Atlas architecture introduces several key innovations including deep memory modules,
            polynomial feature mappings, and the Omega rule for context memorization.
            
            The key insight is that traditional recurrent models update memory based only on the
            current token, which leads to suboptimal memorization. Atlas instead optimizes
            memory based on an entire context window, allowing it to memorize context rather
            than individual tokens.
            
            This approach leads to significant improvements in long-context understanding,
            recall-intensive tasks, and needle-in-haystack evaluations.
            """ * (i % 2 + 1)  # Vary length but keep smaller
            
            sample_texts.append(text)
        
        # Split into train/val
        split_idx = int(len(sample_texts) * train_split)
        train_texts = sample_texts[:split_idx]
        val_texts = sample_texts[split_idx:]
        
        # Create datasets
        train_dataset = LongContextDataset(
            train_texts, 
            tokenizer, 
            max_length=max_length
        )
        
        val_dataset = LongContextDataset(
            val_texts,
            tokenizer,
            max_length=max_length
        )
        
        # Ensure datasets have examples
        if len(train_dataset) == 0:
            print("Warning: Empty train dataset, creating fallback examples")
            fallback_texts = ["This is a test sentence for training."] * 10
            train_dataset = LongContextDataset(fallback_texts, tokenizer, max_length=max_length)
        
        if len(val_dataset) == 0:
            print("Warning: Empty val dataset, creating fallback examples")
            fallback_texts = ["This is a test sentence for validation."] * 5
            val_dataset = LongContextDataset(fallback_texts, tokenizer, max_length=max_length)
        
        # Also create a recall dataset for evaluation
        recall_dataset = RecallDataset(
            tokenizer,
            num_samples=20,  # Reduced for testing
            context_length=max_length
        )
        
    else:
        # Load from specified directory
        train_dataset = BookDataset(
            dataset_path,
            tokenizer,
            max_length=max_length
        )
        
        # Create validation split
        val_size = int(len(train_dataset) * (1 - train_split))
        train_size = len(train_dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        recall_dataset = RecallDataset(
            tokenizer,
            num_samples=20,  # Reduced for testing
            context_length=max_length
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 to avoid multiprocessing issues
        collate_fn=collate_fn,
        pin_memory=False  # Disable for CPU testing
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    recall_loader = DataLoader(
        recall_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'recall': recall_loader
    }


def main():
    """Test data loading"""
    from transformers import GPT2Tokenizer
    from config import Config
    
    # Initialize
    config = Config()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataloaders
    dataloaders = create_dataloaders(config, tokenizer)
    
    print(f"Train batches: {len(dataloaders['train'])}")
    print(f"Val batches: {len(dataloaders['val'])}")
    print(f"Recall batches: {len(dataloaders['recall'])}")
    
    # Test a batch
    batch = next(iter(dataloaders['train']))
    print(f"Batch input shape: {batch['input_ids'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    print(f"Batch attention mask shape: {batch['attention_mask'].shape}")


if __name__ == "__main__":
    main()
