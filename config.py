"""
Configurações centralizadas para o detector de sotaque
"""

# Caminhos
DATA_CONFIG = {
    'csv_path': 'sotaque-brasileiro-data/sotaque-brasileiro.csv',
    'audio_base_path': 'sotaque-brasileiro-data',
    'save_dir': 'experiments'
}

# Parâmetros de áudio
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'duration': 5.0,  # segundos
    'n_mfcc': 40,
    'n_mels': 128
}

# Parâmetros de treinamento
TRAINING_CONFIG = {
    'batch_size': 16,
    'num_workers': 4,  # Ajuste conforme número de CPUs
    'num_epochs': 50,
    'learning_rate': 0.001,
    'early_stopping_patience': 15,
    'weight_decay': 1e-5
}

# Parâmetros de data split
DATA_SPLIT_CONFIG = {
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15
}

# Modelos disponíveis
AVAILABLE_MODELS = {
    'cnn': {
        'name': 'CNN',
        'description': 'Rede Convolucional básica',
        'params': 'Médio',
        'speed': 'Rápido',
        'accuracy': 'Boa'
    },
    'resnet': {
        'name': 'ResNet',
        'description': 'Rede Residual profunda',
        'params': 'Alto',
        'speed': 'Médio',
        'accuracy': 'Muito Boa'
    },
    'attention_cnn': {
        'name': 'Attention CNN',
        'description': 'CNN com mecanismos de atenção',
        'params': 'Alto',
        'speed': 'Médio',
        'accuracy': 'Excelente'
    },
    'lstm': {
        'name': 'LSTM',
        'description': 'Rede recorrente',
        'params': 'Médio',
        'speed': 'Lento',
        'accuracy': 'Boa'
    }
}

# Configuração de labels
LABEL_CONFIG = {
    'label_column': 'birth_state',  # Pode ser 'birth_state' ou 'current_state'
    'min_samples_per_class': 10  # Filtrar classes com poucas amostras
}

# Otimizações
OPTIMIZATION_CONFIG = {
    'mixed_precision': True,  # Mixed precision training (requer GPU)
    'pin_memory': True,  # Pin memory para transferências GPU
    'persistent_workers': True,  # Manter workers vivos
    'compile_model': False  # PyTorch 2.0 compile (experimental)
}

# Data augmentation
AUGMENTATION_CONFIG = {
    'augment_train': True,
    'time_stretch_range': (0.9, 1.1),
    'pitch_shift_range': (-2, 3),
    'noise_amplitude': 0.005,
    'augment_probability': 0.5
}

# Visualização
VISUALIZATION_CONFIG = {
    'plot_style': 'seaborn',
    'figure_dpi': 300,
    'save_format': 'png'
}

# Logging
LOGGING_CONFIG = {
    'log_interval': 10,  # Log a cada N batches
    'save_best_only': True,
    'save_checkpoint_interval': 5  # Salvar checkpoint a cada N épocas
}


def get_config():
    """Retorna todas as configurações"""
    return {
        'data': DATA_CONFIG,
        'audio': AUDIO_CONFIG,
        'training': TRAINING_CONFIG,
        'data_split': DATA_SPLIT_CONFIG,
        'models': AVAILABLE_MODELS,
        'label': LABEL_CONFIG,
        'optimization': OPTIMIZATION_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'logging': LOGGING_CONFIG
    }


def print_config():
    """Imprime todas as configurações"""
    config = get_config()
    
    print("="*60)
    print("CONFIGURAÇÕES DO SISTEMA")
    print("="*60)
    
    for section, settings in config.items():
        print(f"\n[{section.upper()}]")
        if section == 'models':
            for model_id, model_info in settings.items():
                print(f"\n  {model_id}:")
                for key, value in model_info.items():
                    print(f"    {key}: {value}")
        else:
            for key, value in settings.items():
                print(f"  {key}: {value}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print_config()


