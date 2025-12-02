---
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section {
    font-size: 32px;
  }
  h1 {
    color: #0066cc;
    font-size: 48px;
  }
  h2 {
    color: #0066cc;
  }
  code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 24px;
  }
  table {
    font-size: 26px;
  }
---

# 🎙️ Detector de Sotaque Brasileiro
## Deep Learning + Otimizações

**Classificação de 11 estados brasileiros por áudio**

Ciência da Computação - IA

---

## 🎯 O Problema

**Entrada**: Áudio de 5 segundos de fala em português
**Saída**: Estado do falante (BA, CE, MG, RJ, RS, SP, etc.)

### Dataset
- 819 amostras de áudio
- 11 classes (estados)
- Features: Mel-Spectrogram + MFCC

### Desafio
⚠️ Dataset pequeno para Deep Learning → **Precisamos otimizar!**

---

## 🧠 Modelo: Attention CNN

```python
Input (Mel-Spectrogram 128x130)
    ↓
4 Blocos Convolucionais
    ├── Conv2D → BatchNorm → ReLU → MaxPool
    └── Extrai padrões no espectrograma
    ↓
Self-Attention Module
    ├── Spatial: onde focar?
    └── Channel: quais features?
    ↓
FC Layers + Softmax → 11 classes
```

**Por quê Attention?** Foca nas partes discriminativas do áudio

---

## ⚡ Técnica 1: Data Augmentation

### Problema: Dataset pequeno (819 samples)

### Solução: Augmentation no treino
```python
# Aplicado apenas no conjunto de treino
Time Stretching:  velocidade ±10%
Pitch Shifting:   tom ±2 semitons  
Gaussian Noise:   ruído de fundo
```

### Resultado
- ✅ Aumenta dataset efetivo de 819 → ~3000 variações
- ✅ Modelo mais robusto
- ✅ Previne overfitting

---

## ⚡ Técnica 2: Mixed Precision Training

### Problema: GPU com memória limitada

### Solução: Float16 + Float32 híbrido
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():  # Usa FP16 automaticamente
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()  # Escala gradientes
scaler.step(optimizer)
```

### Resultado
- ✅ **~2x mais rápido** em GPUs modernas
- ✅ **50% menos memória**
- ✅ Batch size maior → melhor convergência

---

## ⚡ Técnica 3: Multi-threaded DataLoader

### Problema: CPU ociosa enquanto GPU processa

### Solução: Carregamento paralelo
```python
DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,           # 4 threads paralelas
    pin_memory=True,         # Memória pinada para GPU
    persistent_workers=True  # Workers vivos entre épocas
)
```

### Resultado
- ✅ **3-4x mais rápido** no carregamento
- ✅ GPU sempre ocupada
- ✅ Reduz tempo de treino de 2h → 45min

---

## ⚡ Técnica 4: Learning Rate Schedule + Early Stop

### LR Scheduling
```python
ReduceLROnPlateau(
    optimizer,
    patience=5,    # Espera 5 épocas sem melhoria
    factor=0.5     # Reduz LR pela metade
)
```
**Efeito**: LR 0.001 → 0.0005 → 0.00025 (ajuste fino automático)

### Early Stopping
```python
if val_f1 não melhora por 15 épocas:
    para_treinamento()
```
**Efeito**: Para em ~30 épocas ao invés de 50 (economiza 40% do tempo)

---

## 📊 Resultados

### Experimento: Attention CNN

![width:1100px](experiments/attention_cnn_20251201_194410/training_history.png)

---

## 📊 Matriz de Confusão

![width:800px](experiments/attention_cnn_20251201_194410/confusion_matrix_teste.png)

**Acurácia: ~73%** (Random baseline = 9%)

---

## ⚡ Técnica 5: Segmentação de Áudio

### Problema: Áudios reais têm duração variável

### Solução: Segmentação automática
```python
if audio_duration > 5s:
    # Divide em segmentos de 5s
    segments = split_audio(audio, segment_length=5s)
    
    # Prediz cada segmento
    predictions = [predict(seg) for seg in segments]
    
    # Resultado = média ponderada
    final = weighted_average(predictions)
```

### Resultado
- ✅ **Funciona com qualquer duração** de áudio
- ✅ **Aumenta robustez** ao considerar múltiplos trechos
- ✅ **Reduz impacto de ruído** localizado

---

## 💻 DEMONSTRAÇÃO AO VIVO

### Teste com Segmentação

```bash
# Testa áudios reais (incluindo > 5s)
python demo_segmentation.py
```

### O que o modelo faz:
1. Detecta duração do áudio
2. **Se > 5s**: divide em segmentos
3. Analisa cada segmento
4. Combina resultados (média ponderada)
5. Retorna predição final

**Vamos ver a segmentação funcionando!** 🎤

---

## ✅ Resumo: Técnicas Aplicadas

| Técnica | Benefício | Ganho |
|---------|-----------|-------|
| **Data Augmentation** | Aumenta dataset | Previne overfitting |
| **Mixed Precision** | Menos memória | 2x mais rápido |
| **Multi-threading** | Paralelismo | 3-4x carregamento |
| **LR Schedule** | Ajuste fino | Melhor convergência |
| **Early Stopping** | Para no momento certo | Economiza 40% tempo |
| **Attention** | Foca no importante | +5% acurácia vs CNN |
| **Segmentação** | Áudio > 5s | Funciona qualquer duração |

### Resultado Final
✅ 73% acurácia · ✅ 45min treino · ✅ Qualquer duração · ✅ Código modular

**GitHub**: [seu-usuario]/detector_sotaque

---
