# 🚀 Guia Rápido - Detector de Sotaque

## Instalação em 3 Passos

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Analisar Dados (Opcional mas Recomendado)
```bash
python main.py analyze
```

### 3. Treinar Modelo
```bash
python main.py train --model attention_cnn --epochs 50
```

---

## Comandos Principais

### 📊 Análise de Dados
```bash
python main.py analyze
```

### 🏋️ Treinamento

**Básico:**
```bash
python main.py train
```

**Customizado:**
```bash
python main.py train --model resnet --epochs 100 --batch-size 32 --workers 8
```

**Opções de Modelo:**
- `cnn` - CNN básica (rápido)
- `resnet` - ResNet (preciso)
- `attention_cnn` - CNN com atenção (recomendado)
- `lstm` - LSTM (para sequências)

### 🎯 Predição

**Arquivo único:**
```bash
python main.py predict --checkpoint experiments/modelo/best_model.pth --input audio.wav
```

**Diretório:**
```bash
python main.py predict --checkpoint experiments/modelo/best_model.pth --input audios/ --output results.json
```

### 📋 Listar

**Modelos disponíveis:**
```bash
python main.py list-models
```

**Experimentos salvos:**
```bash
python main.py list-experiments
```

---

## Ajuste de Performance

### 💻 Para CPU
```bash
python main.py train --workers 0 --batch-size 8
```

### 🚀 Para GPU
```bash
python main.py train --batch-size 32 --workers 4
```

### ⚡ Para GPU Potente
```bash
python main.py train --model resnet --batch-size 64 --workers 8 --epochs 100
```

---

## Troubleshooting Rápido

### ❌ Erro "Out of Memory"
```bash
# Reduza batch size
python main.py train --batch-size 8
```

### ❌ Erro "DataLoader Workers"
```bash
# Use workers=0 no Windows
python main.py train --workers 0
```

### ❌ Modelo não aprende
```bash
# Reduza learning rate
python main.py train --lr 0.0001
```

---

## Estrutura de Outputs

```
experiments/
└── attention_cnn_20231201_120000/
    ├── best_model.pth              # Modelo treinado
    ├── training_history.png        # Gráficos
    ├── confusion_matrix_teste.png  # Matriz de confusão
    ├── training_info.json          # Info do treinamento
    └── evaluation_teste.json       # Métricas
```

---

## Exemplos Práticos

### Exemplo 1: Treinamento Rápido
```bash
# 1. Analisar dados
python main.py analyze

# 2. Treinar modelo rápido (CNN)
python main.py train --model cnn --epochs 30 --batch-size 16

# 3. Ver experimentos
python main.py list-experiments
```

### Exemplo 2: Treinamento de Alta Qualidade
```bash
# 1. Treinar com melhor modelo
python main.py train --model attention_cnn --epochs 100 --batch-size 32 --lr 0.0005

# 2. Fazer predições
python main.py predict --checkpoint experiments/[EXPERIMENTO]/best_model.pth --input teste.wav
```

### Exemplo 3: Processar Lote de Áudios
```bash
# Predizer todos os áudios de uma pasta
python main.py predict \
    --checkpoint experiments/[EXPERIMENTO]/best_model.pth \
    --input audios_para_classificar/ \
    --output resultados.json
```

---

## Parâmetros Importantes

| Parâmetro | Padrão | Descrição | Quando Ajustar |
|-----------|--------|-----------|----------------|
| `--model` | attention_cnn | Arquitetura | Sempre testar diferentes |
| `--epochs` | 50 | Número de épocas | Aumentar para melhor qualidade |
| `--batch-size` | 16 | Tamanho do batch | Aumentar se tiver RAM/VRAM |
| `--workers` | 4 | Workers paralelos | Ajustar conforme CPU |
| `--lr` | 0.001 | Learning rate | Reduzir se não convergir |
| `--patience` | 15 | Early stopping | Aumentar para treinar mais |

---

## Dicas de Uso

✅ **Use GPU se disponível** - 5-10x mais rápido
✅ **Comece com análise de dados** - Entenda o dataset
✅ **Teste diferentes modelos** - Cada um tem seus pontos fortes
✅ **Use early stopping** - Evita overfitting
✅ **Monitore as métricas** - Acompanhe gráficos de treinamento
✅ **Salve seus resultados** - Documente experimentos

---

## Próximos Passos

1. ✅ Analisar dados
2. ✅ Treinar primeiro modelo
3. ✅ Avaliar resultados
4. ✅ Ajustar hiperparâmetros
5. ✅ Testar diferentes modelos
6. ✅ Fazer predições em dados reais

---

**Dúvidas?** Consulte o [README.md](README.md) completo ou abra uma issue!


