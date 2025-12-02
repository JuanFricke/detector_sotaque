# Diagrama: Funcionamento da Segmentação de Áudio

## Fluxo de Processamento

```
┌─────────────────────────────────────────────────────────────┐
│                    ENTRADA: Arquivo de Áudio                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Carregar Áudio │
                    │  e Normalizar   │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Verificar       │
                    │ Duração         │
                    └─────────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
          Duração ≤ 5s              Duração > 5s
                 │                         │
                 ▼                         ▼
        ┌─────────────────┐    ┌─────────────────────┐
        │ PREDIÇÃO ÚNICA  │    │ DIVIDIR EM SEGMENTOS│
        │                 │    │ de 5 segundos       │
        │  ┌──────────┐   │    └─────────────────────┘
        │  │ Áudio    │   │               │
        │  │ Original │   │               ▼
        │  └──────────┘   │    ┌─────────────────────┐
        │       │         │    │ Segmento 1 (0-5s)   │
        │       ▼         │    │ Segmento 2 (5-10s)  │
        │  ┌──────────┐   │    │ Segmento 3 (10-15s) │
        │  │ Features │   │    │ ...                 │
        │  │ (Mel)    │   │    └─────────────────────┘
        │  └──────────┘   │               │
        │       │         │               ▼
        │       ▼         │    ┌─────────────────────┐
        │  ┌──────────┐   │    │ Para cada segmento: │
        │  │  Modelo  │   │    │                     │
        │  │  CNN     │   │    │  ┌───────────┐     │
        │  └──────────┘   │    │  │ Features  │     │
        │       │         │    │  │ (Mel)     │     │
        │       ▼         │    │  └───────────┘     │
        │  ┌──────────┐   │    │       │            │
        │  │ Softmax  │   │    │       ▼            │
        │  │ (Probs)  │   │    │  ┌───────────┐     │
        │  └──────────┘   │    │  │  Modelo   │     │
        │       │         │    │  │  CNN      │     │
        │       ▼         │    │  └───────────┘     │
        │  ┌──────────┐   │    │       │            │
        │  │ RESULTADO│   │    │       ▼            │
        │  │ RJ: 95%  │   │    │  ┌───────────┐     │
        │  └──────────┘   │    │  │ Softmax   │     │
        └─────────────────┘    │  │ (Probs)   │     │
                               │  └───────────┘     │
                               └─────────────────────┘
                                        │
                                        ▼
                               ┌─────────────────────┐
                               │ COLETAR TODAS AS    │
                               │ PROBABILIDADES      │
                               │                     │
                               │ Seg1: [.95, .03, ..]│
                               │ Seg2: [.80, .15, ..]│
                               │ Seg3: [.90, .05, ..]│
                               └─────────────────────┘
                                        │
                                        ▼
                               ┌─────────────────────┐
                               │ CALCULAR MÉDIA      │
                               │                     │
                               │ Média: [.88, .08, ..]│
                               └─────────────────────┘
                                        │
                                        ▼
                               ┌─────────────────────┐
                               │ RESULTADO FINAL     │
                               │ RJ: 88%             │
                               │ (média de N segm.)  │
                               └─────────────────────┘
```

## Exemplo Numérico

### Entrada: Áudio de 10 segundos

```
Áudio Original (10s)
├── Segmento 1 (0-5s)
│   Análise: RJ=100%, SP=0%, MG=0%
│
└── Segmento 2 (5-10s)
    Análise: RJ=0%, SP=100%, MG=0%

Cálculo da Média:
├── RJ: (100% + 0%) / 2 = 50%
├── SP: (0% + 100%) / 2 = 50%
└── MG: (0% + 0%) / 2 = 0%

Resultado Final:
├── Predição: RJ (50%) ou SP (50%) - empate, escolhe maior índice
└── Confiança: 50%
```

### Outro Exemplo: Áudio de 15 segundos

```
Áudio Original (15s)
├── Segmento 1 (0-5s):   RJ=90%, SP=8%, MG=2%
├── Segmento 2 (5-10s):  RJ=85%, SP=10%, MG=5%
└── Segmento 3 (10-15s): RJ=95%, SP=3%, MG=2%

Cálculo da Média:
├── RJ: (90 + 85 + 95) / 3 = 90%
├── SP: (8 + 10 + 3) / 3 = 7%
└── MG: (2 + 5 + 2) / 3 = 3%

Resultado Final:
├── Predição: RJ
└── Confiança: 90%
```

## Benefícios Visuais

```
ANTES (processava só 5s):
[========5s========][-----------------ignorado-----------------]
      Analisa               ❌ Perdido

DEPOIS (processa tudo):
[====Seg1====][====Seg2====][====Seg3====][====Seg4====]
     ✓             ✓              ✓             ✓
     
Resultado = Média(Seg1, Seg2, Seg3, Seg4)
```

## Código Simplificado

```python
# Pseudocódigo

function predict(audio):
    duration = get_duration(audio)
    
    if duration <= 5s:
        # Caminho antigo (inalterado)
        features = extract_features(audio)
        probs = model(features)
        return {"accent": argmax(probs), "confidence": max(probs)}
    
    else:
        # NOVO: Caminho com segmentação
        segments = split_into_5s_segments(audio)
        all_probs = []
        
        for segment in segments:
            features = extract_features(segment)
            probs = model(features)
            all_probs.append(probs)
        
        avg_probs = mean(all_probs)
        return {
            "accent": argmax(avg_probs), 
            "confidence": max(avg_probs),
            "segments": segments_info
        }
```

## Status da Implementação

✅ Detecção automática de áudio longo (>5s)  
✅ Divisão em segmentos de 5s  
✅ Processamento de cada segmento  
✅ Cálculo de média das probabilidades  
✅ Exibição de resultados por segmento  
✅ Compatibilidade com código existente  

