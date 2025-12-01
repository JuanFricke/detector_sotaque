"""
Funções utilitárias para o detector de sotaque
"""
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import librosa
import librosa.display


def load_checkpoint(checkpoint_path: str) -> Dict:
    """
    Carrega checkpoint do modelo
    
    Args:
        checkpoint_path: Caminho para o checkpoint
    
    Returns:
        Dicionário com informações do checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def get_model_summary(checkpoint_path: str) -> Dict:
    """
    Obtém resumo do modelo a partir do checkpoint
    
    Args:
        checkpoint_path: Caminho para o checkpoint
    
    Returns:
        Dicionário com resumo do modelo
    """
    checkpoint = load_checkpoint(checkpoint_path)
    
    summary = {
        'model_name': checkpoint.get('model_name', 'N/A'),
        'num_classes': checkpoint.get('num_classes', 'N/A'),
        'best_val_acc': checkpoint.get('best_val_acc', 'N/A'),
        'best_val_f1': checkpoint.get('best_val_f1', 'N/A'),
        'epoch': checkpoint.get('epoch', 'N/A'),
        'classes': list(checkpoint.get('label_to_idx', {}).keys())
    }
    
    return summary


def visualize_audio(audio_path: str, sample_rate: int = 16000, save_path: Optional[str] = None):
    """
    Visualiza áudio com diferentes representações
    
    Args:
        audio_path: Caminho para o arquivo de áudio
        sample_rate: Taxa de amostragem
        save_path: Caminho para salvar figura (opcional)
    """
    # Carregar áudio
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # 1. Forma de onda
    axes[0].plot(np.arange(len(audio)) / sr, audio)
    axes[0].set_title('Forma de Onda', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Tempo (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Espectrograma
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img1 = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=axes[1])
    axes[1].set_title('Espectrograma', fontsize=14, fontweight='bold')
    fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')
    
    # 3. Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img2 = librosa.display.specshow(mel_spec_db, y_axis='mel', x_axis='time', sr=sr, ax=axes[2])
    axes[2].set_title('Mel-Spectrogram', fontsize=14, fontweight='bold')
    fig.colorbar(img2, ax=axes[2], format='%+2.0f dB')
    
    # 4. MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    img3 = librosa.display.specshow(mfcc, x_axis='time', sr=sr, ax=axes[3])
    axes[3].set_title('MFCC', fontsize=14, fontweight='bold')
    fig.colorbar(img3, ax=axes[3])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualização salva em: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_audios(audio_paths: List[str], labels: List[str], sample_rate: int = 16000,
                  save_path: Optional[str] = None):
    """
    Compara múltiplos áudios lado a lado
    
    Args:
        audio_paths: Lista de caminhos para áudios
        labels: Lista de labels para cada áudio
        sample_rate: Taxa de amostragem
        save_path: Caminho para salvar figura (opcional)
    """
    n_audios = len(audio_paths)
    fig, axes = plt.subplots(n_audios, 2, figsize=(15, 4*n_audios))
    
    if n_audios == 1:
        axes = axes.reshape(1, -1)
    
    for i, (audio_path, label) in enumerate(zip(audio_paths, labels)):
        # Carregar áudio
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Forma de onda
        axes[i, 0].plot(np.arange(len(audio)) / sr, audio)
        axes[i, 0].set_title(f'{label} - Forma de Onda', fontsize=12, fontweight='bold')
        axes[i, 0].set_xlabel('Tempo (s)')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(mel_spec_db, y_axis='mel', x_axis='time', 
                                       sr=sr, ax=axes[i, 1])
        axes[i, 1].set_title(f'{label} - Mel-Spectrogram', fontsize=12, fontweight='bold')
        fig.colorbar(img, ax=axes[i, 1], format='%+2.0f dB')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparação salva em: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], 
                         normalize: bool = False, save_path: Optional[str] = None):
    """
    Plota matriz de confusão
    
    Args:
        cm: Matriz de confusão
        classes: Lista de nomes das classes
        normalize: Se True, normaliza a matriz
        save_path: Caminho para salvar figura (opcional)
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
               cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão' + (' (Normalizada)' if normalize else ''),
             fontsize=16, fontweight='bold')
    plt.ylabel('Verdadeiro', fontsize=12)
    plt.xlabel('Predito', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matriz de confusão salva em: {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Calcula pesos das classes para balanceamento
    
    Args:
        labels: Lista de labels
        num_classes: Número de classes
    
    Returns:
        Tensor com pesos das classes
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return torch.FloatTensor(class_weights)


def print_model_info(model: torch.nn.Module):
    """
    Imprime informações do modelo
    
    Args:
        model: Modelo PyTorch
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print("INFORMAÇÕES DO MODELO")
    print("="*60)
    print(f"Total de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")
    print(f"Parâmetros não-treináveis: {total_params - trainable_params:,}")
    print(f"Tamanho estimado: {total_params * 4 / (1024**2):.2f} MB")
    print("="*60)


def save_predictions(predictions: List[Dict], output_path: str):
    """
    Salva predições em arquivo JSON
    
    Args:
        predictions: Lista de predições
        output_path: Caminho para salvar
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)
    
    print(f"Predições salvas em: {output_path}")


def load_predictions(input_path: str) -> List[Dict]:
    """
    Carrega predições de arquivo JSON
    
    Args:
        input_path: Caminho do arquivo
    
    Returns:
        Lista de predições
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    return predictions


def get_audio_duration(audio_path: str) -> float:
    """
    Obtém duração do áudio em segundos
    
    Args:
        audio_path: Caminho para o áudio
    
    Returns:
        Duração em segundos
    """
    audio, sr = librosa.load(audio_path, sr=None)
    duration = len(audio) / sr
    return duration


def batch_process_predictions(model_path: str, audio_dir: str, output_dir: str,
                             batch_size: int = 32):
    """
    Processa predições em lote de forma eficiente
    
    Args:
        model_path: Caminho para o modelo
        audio_dir: Diretório com áudios
        output_dir: Diretório de saída
        batch_size: Tamanho do batch
    """
    import glob
    from predict import AccentPredictor
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar modelo
    predictor = AccentPredictor(model_path)
    
    # Listar áudios
    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    
    if not audio_files:
        print(f"Nenhum arquivo .wav encontrado em: {audio_dir}")
        return
    
    print(f"Processando {len(audio_files)} áudios...")
    
    # Processar em lote
    all_results = []
    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i+batch_size]
        batch_results = predictor.predict_batch(batch_files, return_probs=True)
        all_results.extend(batch_results)
        
        print(f"Processados {min(i+batch_size, len(audio_files))}/{len(audio_files)}")
    
    # Salvar resultados
    output_path = os.path.join(output_dir, 'predictions.json')
    save_predictions(all_results, output_path)
    
    # Estatísticas
    predictions = [r['predicted_accent'] for r in all_results]
    confidences = [r['confidence'] for r in all_results]
    
    print(f"\n{'='*60}")
    print("ESTATÍSTICAS DAS PREDIÇÕES")
    print(f"{'='*60}")
    print(f"Total de predições: {len(all_results)}")
    print(f"Confiança média: {np.mean(confidences)*100:.2f}%")
    print(f"Confiança mínima: {np.min(confidences)*100:.2f}%")
    print(f"Confiança máxima: {np.max(confidences)*100:.2f}%")
    print(f"\nDistribuição de sotaques preditos:")
    
    from collections import Counter
    accent_counts = Counter(predictions)
    for accent, count in accent_counts.most_common():
        pct = count / len(predictions) * 100
        print(f"  {accent}: {count} ({pct:.1f}%)")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    # Exemplos de uso
    print("Funções utilitárias carregadas com sucesso!")
    
    # Exemplo: visualizar áudio
    # visualize_audio("sotaque-brasileiro-data/accent/audio.wav", save_path="audio_viz.png")
    
    # Exemplo: obter resumo do modelo
    # summary = get_model_summary("experiments/modelo/best_model.pth")
    # print(summary)


