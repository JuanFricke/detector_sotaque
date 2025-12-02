# Nova Funcionalidade: Segmentação Automática de Áudio

## O que mudou?

Agora o detector de sotaque processa automaticamente áudios longos (> 5 segundos) dividindo-os em segmentos e calculando a média das predições.

## Como funciona

### Áudio ≤ 5 segundos
- Comportamento normal: uma única predição

### Áudio > 5 segundos
1. O áudio é dividido em segmentos de 5 segundos
2. Cada segmento é analisado independentemente
3. As probabilidades de todos os segmentos são calculadas
4. O resultado final é a **MÉDIA** das probabilidades de todos os segmentos

## Exemplo

Se um áudio de 10 segundos é analisado:

```
Segmento 1 (0-5s):   RJ: 100%,  SP: 0%
Segmento 2 (5-10s):  RJ: 0%,    SP: 100%

Resultado Final:     RJ: 50%,   SP: 50%
```

## Uso

### Modo Interativo
```bash
python main.py interactive
```

### Linha de Comando
```bash
python main.py predict --checkpoint modelo.pth --input audio_longo.wav
```

### Via API Python
```python
from predict import AccentPredictor

predictor = AccentPredictor("modelo.pth")
result = predictor.predict("audio_longo.wav", return_probs=True)

# Verificar se foi segmentado
if 'num_segments' in result:
    print(f"Áudio dividido em {result['num_segments']} segmentos")
    print(f"Predições por segmento: {result['segment_predictions']}")
    print(f"Resultado médio: {result['predicted_accent']}")
```

## Formato da Resposta

### Áudio Curto (≤5s)
```json
{
  "audio_path": "audio.wav",
  "audio_duration": 4.5,
  "predicted_accent": "RJ",
  "confidence": 0.95,
  "all_probabilities": {...}
}
```

### Áudio Longo (>5s)
```json
{
  "audio_path": "audio_longo.wav",
  "audio_duration": 10.5,
  "num_segments": 3,
  "segment_predictions": [
    {"segment": 1, "predicted_accent": "RJ", "confidence": 95.0},
    {"segment": 2, "predicted_accent": "SP", "confidence": 80.0},
    {"segment": 3, "predicted_accent": "RJ", "confidence": 90.0}
  ],
  "predicted_accent": "RJ",
  "confidence": 0.88,
  "all_probabilities": {...}
}
```

## Benefícios

✅ **Maior precisão**: Análise de múltiplas partes do áudio  
✅ **Robustez**: Menos sensível a trechos isolados  
✅ **Transparência**: Você vê a predição de cada segmento  
✅ **Automático**: Funciona sem mudanças no código existente

## Demonstração

Execute o script de demonstração:
```bash
python demo_segmentation.py
```

## Arquivos Modificados

- `predict.py`: Adicionada lógica de segmentação no método `predict()`
  - Novo método: `_split_audio_into_segments()`
  - Atualizado: `print_prediction()` para mostrar info de segmentos

