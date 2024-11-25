from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
import json


class FakeNewsDataset(Dataset):
    def __init__(self, directory_path, data_path):
        self.image_paths = []
        self.texts = []
        self.labels = []
        
        # Load JSONL file
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data = json.loads(line)
                    self.image_paths.append(directory_path + data['image_path'])
                    self.texts.append(data['tweet_text'])
                    # Convert 'Yes'/'No' to 1/0
                    self.labels.append(1 if data['class_label'] == 'Yes' else 0)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Load and preprocess image to 240x240 (ArabicCLIP's expected size)
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image = image.resize((240, 240))
            image = np.array(image)
            image = image.astype('float32') / 255.0  # Normalize to [0,1]
            
            # Convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1)  # Convert to CxHxW format
            
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            # Return a blank image in case of error
            image = torch.zeros((3, 240, 240))
            
        return (
            image,
            self.texts[idx],  # Raw text - will be processed by the model
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

def get_datasets():
    """Helper function to get all datasets"""
    
    train_dataset = FakeNewsDataset('/kaggle/input/clef23arabicfnddataset/CT23_1A_checkworthy_multimodal_arabic/',
        '/kaggle/input/clef23arabicfnddataset/CT23_1A_checkworthy_multimodal_arabic/CT23_1A_checkworthy_multimodal_arabic_train.jsonl'
    )
    
    val_dataset = FakeNewsDataset('/kaggle/input/clef23arabicfnddataset/CT23_1A_checkworthy_multimodal_arabic/',
        '/kaggle/input/clef23arabicfnddataset/CT23_1A_checkworthy_multimodal_arabic/CT23_1A_checkworthy_multimodal_arabic_dev.jsonl'
    )
    
    test_dataset = FakeNewsDataset('/kaggle/input/clef23arabicfnddataset/CT23_1A_checkworthy_multimodal_arabic_test/',
        '/kaggle/input/clef23arabicfnddataset/CT23_1A_checkworthy_multimodal_arabic_test_gold/CT23_1A_checkworthy_multimodal_arabic_test_gold.jsonl'
    )
    
    return train_dataset, val_dataset, test_dataset