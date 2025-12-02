"""
Script simplificado para testar o treinamento
"""
import sys
import os

print("🚀 Iniciando detector de sotaque...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

try:
    print("\n📦 Importando bibliotecas...")
    import torch
    print(f"  ✅ PyTorch {torch.__version__}")
    
    import numpy as np
    print(f"  ✅ NumPy {np.__version__}")
    
    import pandas as pd
    print(f"  ✅ Pandas {pd.__version__}")
    
    import librosa
    print(f"  ✅ Librosa {librosa.__version__}")
    
    print("\n🔍 Verificando dataset...")
    csv_path = "sotaque-brasileiro-data/sotaque-brasileiro.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"  ✅ Dataset carregado: {len(df)} amostras")
    else:
        print(f"  ❌ Dataset não encontrado: {csv_path}")
        sys.exit(1)
    
    print("\n🧠 Importando módulos do projeto...")
    from dataset import BrazilianAccentDataset
    print("  ✅ dataset.py")
    
    from models import get_model
    print("  ✅ models.py")
    
    print("\n✅ Todos os módulos carregados com sucesso!")
    print("\n🏋️ Iniciando treinamento simplificado...")
    
    # Criar dataset pequeno para teste
    dataset = BrazilianAccentDataset(
        csv_path=csv_path,
        audio_base_path="sotaque-brasileiro-data",
        sample_rate=16000,
        augment=False
    )
    
    print(f"\n📊 Dataset criado:")
    print(f"  - Amostras: {len(dataset)}")
    print(f"  - Classes: {dataset.num_classes}")
    print(f"  - Estados: {list(dataset.label_to_idx.keys())}")
    
    # Criar modelo
    print("\n🤖 Criando modelo CNN...")
    model = get_model('cnn', num_classes=dataset.num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ Modelo criado com {total_params:,} parâmetros")
    
    # Testar carregamento de uma amostra
    print("\n🎵 Testando carregamento de áudio...")
    features, label, metadata = dataset[0]
    print(f"  ✅ Features shape: {features.shape}")
    print(f"  ✅ Label: {label} ({dataset.idx_to_label[str(label)]})")
    
    print("\n" + "="*60)
    print("✅ SISTEMA FUNCIONANDO CORRETAMENTE!")
    print("="*60)
    print("\nPara treinar o modelo completo, execute:")
    print("  python train.py")
    print("\nOu use o comando simplificado:")
    print("  python main.py train --model cnn --epochs 30")
    print("="*60)
    
except ImportError as e:
    print(f"\n❌ Erro ao importar biblioteca: {e}")
    print("\n💡 Solução: Instale as dependências com:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ Erro: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)



