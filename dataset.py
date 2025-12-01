"""
Dataset personalizado para carregar e processar dados de sotaque brasileiro
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class BrazilianAccentDataset(Dataset):
    """Dataset para classificação de sotaques brasileiros"""
    
    def __init__(
        self,
        csv_path: str,
        audio_base_path: str,
        sample_rate: int = 16000,
        duration: float = 5.0,
        n_mfcc: int = 40,
        n_mels: int = 128,
        augment: bool = False,
        label_column: str = 'birth_state'
    ):
        """
        Args:
            csv_path: Caminho para o arquivo CSV com metadados
            audio_base_path: Caminho base onde os áudios estão armazenados
            sample_rate: Taxa de amostragem desejada
            duration: Duração esperada dos áudios em segundos
            n_mfcc: Número de coeficientes MFCC
            n_mels: Número de bandas mel-frequency
            augment: Se True, aplica data augmentation
            label_column: Coluna a ser usada como label (birth_state ou current_state)
        """
        self.csv_path = csv_path
        self.audio_base_path = audio_base_path
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.augment = augment
        self.max_length = int(sample_rate * duration)
        
        # Carregar dataset
        self.df = pd.read_csv(csv_path)
        
        # Filtrar apenas linhas com áudio válido
        self.df = self.df[self.df['audio_file_path'].notna()].reset_index(drop=True)
        
        # Preparar labels
        self.label_column = label_column
        self.df['label'] = self.df[label_column]
        
        # Criar mapeamento de labels
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.df['label'].unique()))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
        # Adicionar índices numéricos
        self.df['label_idx'] = self.df['label'].map(self.label_to_idx)
        
        print(f"Dataset carregado: {len(self.df)} amostras")
        print(f"Número de classes: {self.num_classes}")
        print(f"Classes: {list(self.label_to_idx.keys())}")
        print(f"Distribuição de classes:")
        print(self.df['label'].value_counts())
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Carrega e pré-processa o áudio"""
        try:
            # Carregar áudio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Normalizar
            audio = librosa.util.normalize(audio)
            
            # Ajustar comprimento
            if len(audio) < self.max_length:
                # Pad com zeros
                audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
            else:
                # Truncar
                audio = audio[:self.max_length]
            
            return audio
        except Exception as e:
            print(f"Erro ao carregar {audio_path}: {e}")
            # Retornar áudio silencioso
            return np.zeros(self.max_length)
    
    def _apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """Aplica augmentação de dados no áudio"""
        if not self.augment:
            return audio
        
        # Time stretching (aleatório)
        if np.random.random() > 0.5:
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # Pitch shifting (aleatório)
        if np.random.random() > 0.5:
            n_steps = np.random.randint(-2, 3)
            audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
        
        # Adicionar ruído (aleatório)
        if np.random.random() > 0.5:
            noise_amp = 0.005 * np.random.random()
            audio = audio + noise_amp * np.random.randn(len(audio))
        
        # Ajustar comprimento novamente após augmentações
        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
        else:
            audio = audio[:self.max_length]
        
        return audio
    
    def _extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extrai features do áudio"""
        features = {}
        
        # MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc
        )
        features['mfcc'] = mfcc
        
        # Delta MFCC (primeira derivada)
        mfcc_delta = librosa.feature.delta(mfcc)
        features['mfcc_delta'] = mfcc_delta
        
        # Delta-Delta MFCC (segunda derivada)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features['mfcc_delta2'] = mfcc_delta2
        
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels
        )
        # Converter para dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel_spec'] = mel_spec_db
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate
        )
        features['chroma'] = chroma
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate
        )
        features['contrast'] = contrast
        
        return features
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """Retorna uma amostra do dataset"""
        row = self.df.iloc[idx]
        
        # Caminho completo do áudio
        audio_path = os.path.join(self.audio_base_path, row['audio_file_path'])
        
        # Carregar áudio
        audio = self._load_audio(audio_path)
        
        # Aplicar augmentação
        audio = self._apply_augmentation(audio)
        
        # Extrair features
        features = self._extract_features(audio)
        
        # Concatenar todas as features
        # Usando mel-spectrogram como feature principal
        feature_tensor = torch.FloatTensor(features['mel_spec'])
        
        # Adicionar canal (para compatibilidade com CNNs)
        feature_tensor = feature_tensor.unsqueeze(0)
        
        # Label
        label = int(row['label_idx'])
        
        # Metadados adicionais
        metadata = {
            'audio_path': audio_path,
            'birth_state': row['birth_state'],
            'current_state': row['current_state'],
            'gender': row['gender'],
            'age': row['age']
        }
        
        return feature_tensor, label, metadata


def custom_collate_fn(batch):
    """
    Custom collate function to handle metadata dictionaries with string values
    """
    features = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    # Keep metadata as a list of dictionaries (don't try to collate it)
    metadata = [item[2] for item in batch]
    
    return features, labels, metadata


def create_data_loaders(
    csv_path: str,
    audio_base_path: str,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    num_workers: int = 4,
    sample_rate: int = 16000,
    augment_train: bool = True,
    label_column: str = 'birth_state',
    min_samples_per_class: int = 6
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Cria DataLoaders para treino, validação e teste
    
    Returns:
        train_loader, val_loader, test_loader, info_dict
    """
    # Carregar dataset completo
    full_dataset = BrazilianAccentDataset(
        csv_path=csv_path,
        audio_base_path=audio_base_path,
        sample_rate=sample_rate,
        augment=False,
        label_column=label_column
    )
    
    # Filtrar classes com poucas amostras
    df = full_dataset.df
    class_counts = df['label'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index.tolist()
    
    if len(valid_classes) < len(class_counts):
        removed_classes = class_counts[class_counts < min_samples_per_class]
        print(f"\n⚠️  Removendo classes com menos de {min_samples_per_class} amostras:")
        for cls, count in removed_classes.items():
            print(f"   - {cls}: {count} amostra(s)")
        
        df = df[df['label'].isin(valid_classes)].reset_index(drop=True)
        full_dataset.df = df
        
        # Recriar mapeamento de labels
        full_dataset.label_to_idx = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
        full_dataset.idx_to_label = {idx: label for label, idx in full_dataset.label_to_idx.items()}
        full_dataset.num_classes = len(full_dataset.label_to_idx)
        
        # Atualizar índices numéricos
        full_dataset.df['label_idx'] = full_dataset.df['label'].map(full_dataset.label_to_idx)
        
        print(f"\nDataset filtrado: {len(df)} amostras, {full_dataset.num_classes} classes")
        print(f"Classes mantidas: {list(full_dataset.label_to_idx.keys())}")
    
    # Calcular tamanhos dos splits
    total_size = len(df)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"\nSplit do dataset:")
    print(f"Treino: {train_size} amostras")
    print(f"Validação: {val_size} amostras")
    print(f"Teste: {test_size} amostras")
    
    # Split estratificado por label
    from sklearn.model_selection import train_test_split
    
    # Primeiro split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_split),
        stratify=df['label'],
        random_state=42
    )
    
    # Segundo split: val vs test
    val_ratio = val_split / (val_split + (1 - train_split - val_split))
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio),
        stratify=temp_df['label'],
        random_state=42
    )
    
    # Salvar índices para cada split
    train_indices = train_df.index.tolist()
    val_indices = val_df.index.tolist()
    test_indices = test_df.index.tolist()
    
    # Criar datasets específicos
    train_dataset = BrazilianAccentDataset(
        csv_path=csv_path,
        audio_base_path=audio_base_path,
        sample_rate=sample_rate,
        augment=augment_train,
        label_column=label_column
    )
    train_dataset.df = train_df.reset_index(drop=True)
    
    val_dataset = BrazilianAccentDataset(
        csv_path=csv_path,
        audio_base_path=audio_base_path,
        sample_rate=sample_rate,
        augment=False,
        label_column=label_column
    )
    val_dataset.df = val_df.reset_index(drop=True)
    
    test_dataset = BrazilianAccentDataset(
        csv_path=csv_path,
        audio_base_path=audio_base_path,
        sample_rate=sample_rate,
        augment=False,
        label_column=label_column
    )
    test_dataset.df = test_df.reset_index(drop=True)
    
    # Criar DataLoaders com multithreading e custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=custom_collate_fn
    )
    
    # Informações do dataset
    info = {
        'num_classes': full_dataset.num_classes,
        'label_to_idx': full_dataset.label_to_idx,
        'idx_to_label': full_dataset.idx_to_label,
        'classes': list(full_dataset.label_to_idx.keys()),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset)
    }
    
    return train_loader, val_loader, test_loader, info


if __name__ == "__main__":
    # Teste do dataset
    csv_path = "sotaque-brasileiro-data/sotaque-brasileiro.csv"
    audio_base_path = "sotaque-brasileiro-data"
    
    train_loader, val_loader, test_loader, info = create_data_loaders(
        csv_path=csv_path,
        audio_base_path=audio_base_path,
        batch_size=16,
        num_workers=4
    )
    
    print("\nTeste do DataLoader:")
    for batch_idx, (features, labels, metadata) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Unique labels: {torch.unique(labels)}")
        break


