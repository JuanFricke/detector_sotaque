"""
Script para fazer predições com o modelo treinado
"""
import os
import json
import torch
import numpy as np
import librosa
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from models import get_model


class AccentPredictor:
    """Classe para fazer predições de sotaque"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = None
    ):
        """
        Args:
            checkpoint_path: Caminho para o checkpoint do modelo
            device: Dispositivo (cuda ou cpu)
        """
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Carregando modelo de: {checkpoint_path}")
        print(f"Usando device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model_name = checkpoint['model_name']
        self.num_classes = checkpoint['num_classes']
        self.label_to_idx = checkpoint['label_to_idx']
        # Ensure idx_to_label has integer keys
        self.idx_to_label = {int(k): v for k, v in checkpoint['idx_to_label'].items()}
        
        # Create and load model
        self.model = get_model(self.model_name, self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Audio parameters
        self.sample_rate = 16000
        self.duration = 5.0
        self.max_length = int(self.sample_rate * self.duration)
        self.n_mels = 128
        
        print(f"Modelo carregado: {self.model_name}")
        print(f"Classes: {list(self.label_to_idx.keys())}")
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Carrega e pré-processa o áudio"""
        # Carregar áudio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Normalizar
        audio = librosa.util.normalize(audio)
        
        # Ajustar comprimento
        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)), mode='constant')
        else:
            audio = audio[:self.max_length]
        
        return audio
    
    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extrai features do áudio"""
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels
        )
        
        # Converter para dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    @torch.no_grad()
    def predict(
        self,
        audio_path: str,
        return_probs: bool = True
    ) -> Dict:
        """
        Faz predição para um áudio
        
        Args:
            audio_path: Caminho para o arquivo de áudio
            return_probs: Se True, retorna probabilidades para todas as classes
        
        Returns:
            Dicionário com predição e probabilidades
        """
        # Load and process audio
        audio = self._load_audio(audio_path)
        features = self._extract_features(audio)
        
        # Convert to tensor
        feature_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
        feature_tensor = feature_tensor.to(self.device)
        
        # Predict
        output = self.model(feature_tensor)
        probs = torch.softmax(output, dim=1)
        
        # Get prediction
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_label = self.idx_to_label[pred_idx]
        pred_confidence = probs[0, pred_idx].item()
        
        result = {
            'audio_path': audio_path,
            'predicted_accent': pred_label,
            'confidence': pred_confidence
        }
        
        if return_probs:
            all_probs = {
                self.idx_to_label[i]: probs[0, i].item()
                for i in range(self.num_classes)
            }
            result['all_probabilities'] = all_probs
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        audio_paths: List[str],
        return_probs: bool = True
    ) -> List[Dict]:
        """
        Faz predições para múltiplos áudios
        
        Args:
            audio_paths: Lista de caminhos para arquivos de áudio
            return_probs: Se True, retorna probabilidades para todas as classes
        
        Returns:
            Lista de dicionários com predições
        """
        results = []
        
        for audio_path in audio_paths:
            result = self.predict(audio_path, return_probs)
            results.append(result)
        
        return results
    
    def print_prediction(self, result: Dict):
        """Imprime resultado de predição de forma formatada"""
        print("\n" + "="*60)
        print(f"Áudio: {os.path.basename(result['audio_path'])}")
        print(f"Sotaque Predito: {result['predicted_accent']}")
        print(f"Confiança: {result['confidence']*100:.2f}%")
        
        if 'all_probabilities' in result:
            print("\nProbabilidades por classe:")
            probs = result['all_probabilities']
            # Ordenar por probabilidade
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            for label, prob in sorted_probs:
                bar = "█" * int(prob * 50)
                print(f"  {label:>5}: {prob*100:5.2f}% {bar}")
        
        print("="*60)


def main():
    """Exemplo de uso"""
    import sys
    
    if len(sys.argv) < 3:
        print("Uso: python predict.py <checkpoint_path> <audio_path>")
        print("Exemplo: python predict.py experiments/attention_cnn_20231201_120000/best_model.pth audio.wav")
        return
    
    checkpoint_path = sys.argv[1]
    audio_path = sys.argv[2]
    
    # Verificar se arquivos existem
    if not os.path.exists(checkpoint_path):
        print(f"Erro: Checkpoint não encontrado: {checkpoint_path}")
        return
    
    if not os.path.exists(audio_path):
        print(f"Erro: Áudio não encontrado: {audio_path}")
        return
    
    # Criar preditor
    predictor = AccentPredictor(checkpoint_path)
    
    # Fazer predição
    result = predictor.predict(audio_path, return_probs=True)
    
    # Mostrar resultado
    predictor.print_prediction(result)
    
    # Salvar resultado
    output_path = audio_path.replace('.wav', '_prediction.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"\nResultado salvo em: {output_path}")


if __name__ == "__main__":
    main()


