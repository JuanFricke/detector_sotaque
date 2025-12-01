"""
Script para iniciar o treinamento com output forçado
"""
import sys
import os

# Forçar output não bufferizado
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("=" * 60, flush=True)
print("🚀 INICIANDO TREINAMENTO - Detector de Sotaque", flush=True)
print("=" * 60, flush=True)

# Importar e executar treinamento
from train import AccentDetectorTrainer

CSV_PATH = "sotaque-brasileiro-data/sotaque-brasileiro.csv"
AUDIO_BASE_PATH = "sotaque-brasileiro-data"
MODEL_NAME = "cnn"  # Modelo mais rápido para teste
BATCH_SIZE = 8
NUM_WORKERS = 0  # 0 para Windows evitar problemas
NUM_EPOCHS = 30  # Menos épocas para teste
LEARNING_RATE = 0.001

print(f"\n📋 Configurações:", flush=True)
print(f"  Modelo: {MODEL_NAME}", flush=True)
print(f"  Épocas: {NUM_EPOCHS}", flush=True)
print(f"  Batch Size: {BATCH_SIZE}", flush=True)
print(f"  Workers: {NUM_WORKERS}", flush=True)
print(f"  Learning Rate: {LEARNING_RATE}", flush=True)
print("", flush=True)

try:
    # Criar trainer
    trainer = AccentDetectorTrainer(
        model_name=MODEL_NAME,
        num_classes=None,
        csv_path=CSV_PATH,
        audio_base_path=AUDIO_BASE_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        learning_rate=LEARNING_RATE,
        mixed_precision=False,  # Desabilitar para CPU
        label_column='birth_state'
    )
    
    print("\n✅ Trainer criado com sucesso!", flush=True)
    print(f"Número de classes: {trainer.num_classes}", flush=True)
    print(f"Classes: {list(trainer.label_to_idx.keys())}", flush=True)
    
    # Treinar
    print(f"\n{'='*60}", flush=True)
    print("🏋️ INICIANDO TREINAMENTO...", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    trainer.train(
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=10
    )
    
    # Avaliar
    print(f"\n{'='*60}", flush=True)
    print("📊 AVALIANDO MODELO...", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    trainer.evaluate(use_test_set=True)
    
    print(f"\n{'='*60}", flush=True)
    print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\nModelo salvo em: {trainer.save_dir}", flush=True)
    print(f"Melhor acurácia: {trainer.best_val_acc:.2f}%", flush=True)
    print(f"Melhor F1-Score: {trainer.best_val_f1:.4f}", flush=True)
    
except KeyboardInterrupt:
    print("\n\n⚠️ Treinamento interrompido pelo usuário", flush=True)
    sys.exit(1)
    
except Exception as e:
    print(f"\n\n❌ ERRO: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)


