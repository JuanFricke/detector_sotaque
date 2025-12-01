"""
Script de treinamento otimizado com multithreading para detector de sotaque
"""
import os
import time
import json
from datetime import datetime
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from dataset import create_data_loaders
from models import get_model


class AccentDetectorTrainer:
    """Classe para treinar o detector de sotaques com otimizações"""
    
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        csv_path: str,
        audio_base_path: str,
        save_dir: str = "experiments",
        batch_size: int = 32,
        num_workers: int = 4,
        learning_rate: float = 0.001,
        device: str = None,
        mixed_precision: bool = True,
        label_column: str = 'birth_state'
    ):
        """
        Args:
            model_name: Nome do modelo a ser usado
            num_classes: Número de classes
            csv_path: Caminho para o CSV com metadados
            audio_base_path: Caminho base dos áudios
            save_dir: Diretório para salvar checkpoints e logs
            batch_size: Tamanho do batch
            num_workers: Número de workers para DataLoader
            learning_rate: Taxa de aprendizado inicial
            device: Dispositivo (cuda ou cpu)
            mixed_precision: Usar mixed precision training
            label_column: Coluna de label (birth_state ou current_state)
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.csv_path = csv_path
        self.audio_base_path = audio_base_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.mixed_precision = mixed_precision
        self.label_column = label_column
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Usando device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memória disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Create save directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{model_name}_{timestamp}"
        self.save_dir = os.path.join(save_dir, self.experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"Experimento: {self.experiment_name}")
        print(f"Salvando em: {self.save_dir}")
        
        # Initialize data loaders
        print("\nCarregando dados...")
        self.train_loader, self.val_loader, self.test_loader, self.data_info = create_data_loaders(
            csv_path=csv_path,
            audio_base_path=audio_base_path,
            batch_size=batch_size,
            num_workers=num_workers,
            augment_train=True,
            label_column=label_column
        )
        
        self.num_classes = self.data_info['num_classes']
        self.label_to_idx = self.data_info['label_to_idx']
        self.idx_to_label = self.data_info['idx_to_label']
        
        # Initialize model
        print(f"\nCriando modelo: {model_name}")
        self.model = get_model(model_name, self.num_classes)
        self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total de parâmetros: {total_params:,}")
        print(f"Parâmetros treináveis: {trainable_params:,}")
        
        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision and self.device.type == 'cuda' else None
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
    
    def train_epoch(self) -> Tuple[float, float]:
        """Treina por uma época"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Treinando")
        for batch_idx, (features, labels, metadata) in enumerate(pbar):
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision training
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, float]:
        """Valida o modelo"""
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc="Validando")
        for batch_idx, (features, labels, metadata) in enumerate(pbar):
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1)
            })
        
        epoch_loss = running_loss / len(self.val_loader)
        accuracy = 100. * np.mean(np.array(all_predictions) == np.array(all_labels))
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return epoch_loss, accuracy, f1
    
    def train(
        self,
        num_epochs: int = 50,
        early_stopping_patience: int = 15,
        save_best_only: bool = True
    ):
        """
        Treina o modelo
        
        Args:
            num_epochs: Número de épocas
            early_stopping_patience: Paciência para early stopping
            save_best_only: Salvar apenas o melhor modelo
        """
        print(f"\nIniciando treinamento por {num_epochs} épocas...")
        print(f"Early stopping patience: {early_stopping_patience}")
        
        best_epoch = 0
        epochs_without_improvement = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"Época {epoch+1}/{num_epochs}")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1 = self.validate()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"\nResumo da Época {epoch+1}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.4f}")
            print(f"  Tempo: {epoch_time:.2f}s")
            
            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ Novo melhor modelo! Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")
            else:
                epochs_without_improvement += 1
                if not save_best_only:
                    self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠ Early stopping ativado após {epochs_without_improvement} épocas sem melhoria")
                break
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Treinamento concluído!")
        print(f"Tempo total: {total_time/60:.2f} minutos")
        print(f"Melhor época: {best_epoch}")
        print(f"Melhor Val Acc: {self.best_val_acc:.2f}%")
        print(f"Melhor Val F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}")
        
        # Plot training history
        self.plot_training_history()
        
        # Save training info
        self.save_training_info(total_time, best_epoch)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Salva checkpoint do modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_name': self.model_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'num_classes': self.num_classes
        }
        
        if is_best:
            path = os.path.join(self.save_dir, 'best_model.pth')
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        
        torch.save(checkpoint, path)
    
    def plot_training_history(self):
        """Plota histórico de treinamento"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Loss por Época', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.val_accuracies, 'g-', linewidth=2)
        axes[0, 1].set_title('Acurácia de Validação', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Acurácia (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 0].plot(epochs, self.val_f1_scores, 'm-', linewidth=2)
        axes[1, 0].set_title('F1 Score de Validação', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].text(0.5, 0.5, f'Melhor Acurácia: {self.best_val_acc:.2f}%\n'
                                  f'Melhor F1 Score: {self.best_val_f1:.4f}\n'
                                  f'Total de Épocas: {len(self.train_losses)}',
                       ha='center', va='center', fontsize=14,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico de treinamento salvo em: {self.save_dir}/training_history.png")
    
    def save_training_info(self, total_time: float, best_epoch: int):
        """Salva informações do treinamento"""
        info = {
            'experiment_name': self.experiment_name,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'classes': list(self.label_to_idx.keys()),
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': len(self.train_losses),
            'best_epoch': best_epoch,
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'total_training_time_minutes': total_time / 60,
            'device': str(self.device),
            'mixed_precision': self.mixed_precision,
            'label_column': self.label_column,
            'train_size': self.data_info['train_size'],
            'val_size': self.data_info['val_size'],
            'test_size': self.data_info['test_size']
        }
        
        with open(os.path.join(self.save_dir, 'training_info.json'), 'w') as f:
            json.dump(info, f, indent=4)
        
        print(f"Informações de treinamento salvas em: {self.save_dir}/training_info.json")
    
    @torch.no_grad()
    def evaluate(self, use_test_set: bool = True) -> Dict:
        """
        Avalia o modelo no conjunto de teste
        
        Args:
            use_test_set: Se True, usa test set, senão usa validation set
        
        Returns:
            Dicionário com métricas de avaliação
        """
        print("\n" + "="*60)
        print("Avaliando modelo...")
        print("="*60)
        
        # Load best model
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Modelo carregado: {best_model_path}")
        
        self.model.eval()
        
        loader = self.test_loader if use_test_set else self.val_loader
        dataset_name = "Teste" if use_test_set else "Validação"
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(loader, desc=f"Avaliando em {dataset_name}")
        for features, labels, metadata in pbar:
            features = features.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            outputs = self.model(features)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100. * np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Classification report
        class_names = [self.idx_to_label[i] for i in range(self.num_classes)]
        report = classification_report(
            all_labels,
            all_predictions,
            target_names=class_names,
            digits=4
        )
        
        print(f"\n{dataset_name} Set - Acurácia: {accuracy:.2f}%")
        print(f"\nRelatório de Classificação:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title(f'Matriz de Confusão - {dataset_name} Set', fontsize=16, fontweight='bold')
        plt.ylabel('Verdadeiro', fontsize=12)
        plt.xlabel('Predito', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'confusion_matrix_{dataset_name.lower()}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nMatriz de confusão salva em: {self.save_dir}/confusion_matrix_{dataset_name.lower()}.png")
        
        # Save detailed results
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'true_labels': all_labels,
            'probabilities': np.array(all_probs).tolist()
        }
        
        with open(os.path.join(self.save_dir, f'evaluation_{dataset_name.lower()}.json'), 'w') as f:
            json.dump({k: v for k, v in results.items() if k not in ['probabilities']}, f, indent=4, default=str)
        
        return results


if __name__ == "__main__":
    # Configurações
    CSV_PATH = "sotaque-brasileiro-data/sotaque-brasileiro.csv"
    AUDIO_BASE_PATH = "sotaque-brasileiro-data"
    MODEL_NAME = "attention_cnn"  # Opções: 'cnn', 'resnet', 'attention_cnn', 'lstm'
    BATCH_SIZE = 16
    NUM_WORKERS = 4  # Ajuste conforme o número de CPUs disponíveis
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Criar trainer
    trainer = AccentDetectorTrainer(
        model_name=MODEL_NAME,
        num_classes=None,  # Será determinado automaticamente
        csv_path=CSV_PATH,
        audio_base_path=AUDIO_BASE_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        learning_rate=LEARNING_RATE,
        mixed_precision=True,  # Usar mixed precision se disponível GPU
        label_column='birth_state'  # Classificar por estado de nascimento
    )
    
    # Treinar
    trainer.train(
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=15
    )
    
    # Avaliar
    trainer.evaluate(use_test_set=True)


