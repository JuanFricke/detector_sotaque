"""
Demo final da funcionalidade de segmentação
"""
from predict import AccentPredictor
import os

print("\n" + "="*70)
print("DEMONSTRAÇÃO: SEGMENTAÇÃO AUTOMÁTICA DE ÁUDIO")
print("="*70)

# Configuração
checkpoint_path = "experiments/attention_cnn_20251201_194410/best_model.pth"

# Testar com diferentes áudios
test_audios = [
    "real_data/pe_povo_vai_comer_abbora_melancia_panna_lula_meme.wav",  # ~5.25s
    "real_data/rj_davy_jones_kinnect.wav",  # Provavelmente curto
]

print(f"\n📁 Carregando modelo: {os.path.basename(checkpoint_path)}")
predictor = AccentPredictor(checkpoint_path)

for i, audio_path in enumerate(test_audios, 1):
    print(f"\n{'='*70}")
    print(f"TESTE {i}/2: {os.path.basename(audio_path)}")
    print(f"{'='*70}")
    
    try:
        result = predictor.predict(audio_path, return_probs=True)
        predictor.print_prediction(result)
        
        # Destacar se foi segmentado
        if 'num_segments' in result and result['num_segments'] > 1:
            print(f"\n✅ FUNCIONALIDADE ATIVA: Áudio dividido em {result['num_segments']} segmentos")
            print(f"   Resultado é a MÉDIA das predições de cada segmento")
        else:
            print(f"\n✓ Áudio curto (≤5s) - predição única")
    
    except Exception as e:
        print(f"❌ Erro ao processar: {e}")

print("\n" + "="*70)
print("✅ DEMONSTRAÇÃO CONCLUÍDA")
print("="*70)
print("\nCOMO FUNCIONA:")
print("• Áudio ≤ 5s: Predição única normal")
print("• Áudio > 5s: Dividido em segmentos de 5s")
print("•  Cada segmento é analisado individualmente")
print("• Resultado final = MÉDIA das probabilidades de todos os segmentos")
print("="*70)

