"""
Script para testar a instalação e configuração do sistema
"""
import sys
import os


def test_python_version():
    """Testa versão do Python"""
    print("🐍 Testando versão do Python...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Requer >= 3.8)")
        return False


def test_imports():
    """Testa importação de bibliotecas"""
    print("\n📦 Testando bibliotecas...")
    
    libraries = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'librosa': 'Librosa',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'tqdm'
    }
    
    all_ok = True
    for lib, name in libraries.items():
        try:
            __import__(lib)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - NÃO INSTALADO")
            all_ok = False
    
    return all_ok


def test_torch_cuda():
    """Testa disponibilidade de CUDA"""
    print("\n🚀 Testando GPU/CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA disponível")
            print(f"   ℹ️  GPU: {torch.cuda.get_device_name(0)}")
            print(f"   ℹ️  CUDA Version: {torch.version.cuda}")
            print(f"   ℹ️  Memória: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print(f"   ⚠️  CUDA não disponível (treinamento será em CPU)")
            return False
    except Exception as e:
        print(f"   ❌ Erro ao verificar CUDA: {e}")
        return False


def test_dataset():
    """Testa disponibilidade do dataset"""
    print("\n📂 Testando dataset...")
    
    csv_path = "sotaque-brasileiro-data/sotaque-brasileiro.csv"
    audio_dir = "sotaque-brasileiro-data/accent"
    
    if os.path.exists(csv_path):
        print(f"   ✅ CSV encontrado: {csv_path}")
        
        # Contar linhas
        with open(csv_path, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for _ in f) - 1  # -1 para header
        print(f"   ℹ️  {num_lines} amostras no dataset")
    else:
        print(f"   ❌ CSV não encontrado: {csv_path}")
        return False
    
    if os.path.exists(audio_dir):
        # Contar arquivos wav
        import glob
        wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
        print(f"   ✅ Diretório de áudios encontrado")
        print(f"   ℹ️  {len(wav_files)} arquivos .wav")
    else:
        print(f"   ❌ Diretório de áudios não encontrado: {audio_dir}")
        return False
    
    return True


def test_audio_loading():
    """Testa carregamento de áudio"""
    print("\n🎵 Testando carregamento de áudio...")
    
    audio_dir = "sotaque-brasileiro-data/accent"
    
    if not os.path.exists(audio_dir):
        print("   ⚠️  Pulando teste (dataset não encontrado)")
        return False
    
    try:
        import glob
        import librosa
        
        wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
        
        if not wav_files:
            print("   ⚠️  Nenhum arquivo .wav encontrado")
            return False
        
        # Testar carregar primeiro áudio
        test_audio = wav_files[0]
        audio, sr = librosa.load(test_audio, sr=16000, duration=1.0)
        
        print(f"   ✅ Áudio carregado com sucesso")
        print(f"   ℹ️  Arquivo de teste: {os.path.basename(test_audio)}")
        print(f"   ℹ️  Sample rate: {sr} Hz")
        print(f"   ℹ️  Duração: {len(audio)/sr:.2f}s")
        print(f"   ℹ️  Shape: {audio.shape}")
        
        return True
    except Exception as e:
        print(f"   ❌ Erro ao carregar áudio: {e}")
        return False


def test_model_creation():
    """Testa criação de modelo"""
    print("\n🧠 Testando criação de modelo...")
    
    try:
        import torch
        from models import get_model
        
        # Testar criar um modelo simples
        model = get_model('cnn', num_classes=10)
        
        # Testar forward pass
        dummy_input = torch.randn(2, 1, 128, 313)
        output = model(dummy_input)
        
        print(f"   ✅ Modelo criado com sucesso")
        print(f"   ℹ️  Input shape: {dummy_input.shape}")
        print(f"   ℹ️  Output shape: {output.shape}")
        print(f"   ℹ️  Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"   ❌ Erro ao criar modelo: {e}")
        return False


def test_multiprocessing():
    """Testa multiprocessing"""
    print("\n⚡ Testando multiprocessing...")
    
    try:
        import multiprocessing as mp
        
        cpu_count = mp.cpu_count()
        print(f"   ✅ Multiprocessing disponível")
        print(f"   ℹ️  CPUs disponíveis: {cpu_count}")
        
        if cpu_count >= 4:
            print(f"   ✅ Recomendado: num_workers=4")
        elif cpu_count >= 2:
            print(f"   ⚠️  Poucos CPUs. Recomendado: num_workers=2")
        else:
            print(f"   ⚠️  CPU único. Recomendado: num_workers=0")
        
        return True
    except Exception as e:
        print(f"   ❌ Erro ao testar multiprocessing: {e}")
        return False


def print_summary(results):
    """Imprime resumo dos testes"""
    print("\n" + "="*60)
    print("📊 RESUMO DOS TESTES")
    print("="*60)
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal de testes: {total}")
    print(f"Passou: {passed}")
    print(f"Falhou: {total - passed}")
    
    if passed == total:
        print("\n✅ TODOS OS TESTES PASSARAM! Sistema pronto para uso.")
        print("\nPróximos passos:")
        print("  1. python main.py analyze")
        print("  2. python main.py train")
    else:
        print("\n⚠️  ALGUNS TESTES FALHARAM")
        print("\nProblemas encontrados:")
        for test, passed in results.items():
            if not passed:
                print(f"  ❌ {test}")
        
        print("\nSoluções:")
        print("  - Instale as dependências: pip install -r requirements.txt")
        print("  - Verifique se o dataset está no lugar correto")
        print("  - Consulte README.md para mais informações")
    
    print("="*60)


def main():
    """Executa todos os testes"""
    print("="*60)
    print("🧪 TESTE DE INSTALAÇÃO - Detector de Sotaque")
    print("="*60)
    
    results = {}
    
    # Executar testes
    results['Python Version'] = test_python_version()
    results['Libraries'] = test_imports()
    results['GPU/CUDA'] = test_torch_cuda()
    results['Dataset'] = test_dataset()
    results['Audio Loading'] = test_audio_loading()
    results['Model Creation'] = test_model_creation()
    results['Multiprocessing'] = test_multiprocessing()
    
    # Resumo
    print_summary(results)
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


