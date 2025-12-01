# 🎙️ Detector de Sotaque Brasileiro - IA

Sistema completo de detecção de sotaques brasileiros usando Deep Learning, com otimizações de multithreading e as melhores práticas de Machine Learning.

## 📋 Índice

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Uso](#uso)
- [Modelos Disponíveis](#modelos-disponíveis)
- [Otimizações](#otimizações)
- [Resultados](#resultados)
- [Arquitetura](#arquitetura)

## 🎯 Características

- ✅ **Múltiplos Modelos**: CNN, ResNet, Attention CNN, LSTM
- ✅ **Data Augmentation**: Time stretching, pitch shifting, ruído
- ✅ **Multithreading Otimizado**: DataLoader com workers paralelos
- ✅ **Mixed Precision Training**: Treinamento mais rápido com menor uso de memória
- ✅ **Early Stopping**: Prevenção de overfitting
- ✅ **Learning Rate Scheduling**: Ajuste automático da taxa de aprendizado
- ✅ **Visualizações Completas**: Gráficos de treinamento, matriz de confusão
- ✅ **Métricas Detalhadas**: Acurácia, F1-Score, Precision, Recall
- ✅ **Análise Exploratória**: Script completo de EDA

## 🔧 Requisitos

- Python 3.8+
- PyTorch 2.0+
- CUDA (opcional, para GPU)
- 8GB+ RAM recomendado
- GPU com 4GB+ VRAM (opcional)

## 📦 Instalação

### 1. Clone o repositório
```bash
cd detector_sotaque
```

### 2. Crie um ambiente virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

## 📁 Estrutura do Projeto

```
detector_sotaque/
├── sotaque-brasileiro-data/          # Dataset
│   ├── accent/                       # Arquivos de áudio (.wav)
│   └── sotaque-brasileiro.csv        # Metadados
├── dataset.py                        # Dataset customizado e DataLoaders
├── models.py                         # Arquiteturas de modelos
├── train.py                          # Script de treinamento
├── predict.py                        # Script de inferência
├── analyze_data.py                   # Análise exploratória
├── requirements.txt                  # Dependências
├── README.md                         # Documentação
└── experiments/                      # Checkpoints e logs (criado automaticamente)
```

## 🚀 Uso

### 1. Análise Exploratória dos Dados

Antes de treinar, explore o dataset:

```bash
python analyze_data.py
```

Isso gerará:
- Visualizações da distribuição dos dados
- Estatísticas detalhadas
- Matriz de migração entre estados
- Relatório JSON com métricas

### 2. Treinamento

#### Treinamento Básico
```bash
python train.py
```

#### Personalizar Configurações

Edite as configurações no arquivo `train.py`:

```python
MODEL_NAME = "attention_cnn"  # 'cnn', 'resnet', 'attention_cnn', 'lstm'
BATCH_SIZE = 16               # Ajuste conforme sua GPU
NUM_WORKERS = 4               # Número de CPUs para carregamento de dados
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
```

#### Treinar com GPU
```bash
# Automático - detecta GPU se disponível
python train.py
```

#### Treinar apenas com CPU
```python
# Em train.py, altere:
trainer = AccentDetectorTrainer(
    ...
    device='cpu'
)
```

### 3. Fazer Predições

Após treinar, use o modelo para fazer predições:

```bash
python predict.py experiments/attention_cnn_TIMESTAMP/best_model.pth audio.wav
```

Exemplo de saída:
```
============================================================
Áudio: audio.wav
Sotaque Predito: SP
Confiança: 87.45%

Probabilidades por classe:
     SP: 87.45% ████████████████████████████████████████████
     RJ: 8.32%  ████
     MG: 2.15%  █
     RS: 1.08%  
============================================================
```

### 4. Avaliar Modelo

O script de treinamento já avalia automaticamente o modelo no conjunto de teste ao final do treinamento.

## 🧠 Modelos Disponíveis

### 1. **CNN (Convolutional Neural Network)**
- Modelo base com 4 blocos convolucionais
- Rápido e eficiente
- Bom para datasets menores

### 2. **ResNet (Residual Network)**
- Conexões residuais para treinar redes mais profundas
- Melhor generalização
- Recomendado para datasets maiores

### 3. **Attention CNN** (Recomendado)
- CNN com mecanismos de atenção
- Foca nas partes mais importantes do áudio
- Melhor performance geral

### 4. **LSTM (Long Short-Term Memory)**
- Modelo recorrente para sequências temporais
- Captura dependências de longo prazo
- Mais lento, mas muito eficaz

## ⚡ Otimizações Implementadas

### Multithreading
- **DataLoader Workers**: Carregamento paralelo de dados
- **Persistent Workers**: Workers mantidos vivos entre épocas
- **Pin Memory**: Transferência mais rápida para GPU

```python
DataLoader(
    dataset,
    num_workers=4,          # 4 threads paralelas
    pin_memory=True,        # Otimização para GPU
    persistent_workers=True # Workers persistentes
)
```

### Mixed Precision Training
- Usa float16 onde possível para economizar memória
- Mantém float32 onde necessário para estabilidade
- ~2x mais rápido em GPUs modernas

```python
# Automático com GradScaler
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

### Data Augmentation
- **Time Stretching**: Varia velocidade do áudio
- **Pitch Shifting**: Altera tom
- **Ruído Gaussiano**: Aumenta robustez
- Aplicado apenas no conjunto de treino

## 📊 Resultados

O sistema gera automaticamente:

### Durante o Treinamento
- Loss de treino e validação por época
- Acurácia de validação
- F1-Score
- Checkpoints do melhor modelo

### Após o Treinamento
- **Gráficos de treinamento**: `training_history.png`
- **Matriz de confusão**: `confusion_matrix_teste.png`
- **Relatório de classificação**: Precision, Recall, F1 por classe
- **Arquivo JSON**: `training_info.json` com todos os detalhes

### Estrutura de Saída
```
experiments/
└── attention_cnn_20231201_120000/
    ├── best_model.pth                    # Melhor modelo
    ├── training_history.png              # Gráficos de treinamento
    ├── confusion_matrix_teste.png        # Matriz de confusão
    ├── training_info.json                # Informações do treinamento
    └── evaluation_teste.json             # Métricas detalhadas
```

## 🏗️ Arquitetura

### Pipeline de Processamento

```
Áudio WAV
    ↓
Carregamento (librosa)
    ↓
Normalização
    ↓
Ajuste de Comprimento (5 segundos)
    ↓
Data Augmentation (treino apenas)
    ↓
Extração de Features
    ├── MFCC
    ├── Mel-Spectrogram (principal)
    ├── Chroma
    └── Spectral Contrast
    ↓
Modelo Deep Learning
    ↓
Classificação de Sotaque
```

### Features Extraídas

1. **Mel-Spectrogram**: Representação tempo-frequência
2. **MFCC**: Coeficientes cepstrais de mel
3. **Delta MFCC**: Primeira e segunda derivadas
4. **Chroma**: Características harmônicas
5. **Spectral Contrast**: Contraste espectral

## 🎓 Boas Práticas Implementadas

- ✅ **Stratified Split**: Divisão estratificada por classe
- ✅ **Cross-Validation Ready**: Fácil adaptação para k-fold
- ✅ **Reproducibilidade**: Seeds fixadas
- ✅ **Logging Completo**: Todas as métricas salvas
- ✅ **Checkpoint System**: Salva melhor modelo automaticamente
- ✅ **Early Stopping**: Para quando não há melhoria
- ✅ **Learning Rate Scheduling**: Ajuste automático
- ✅ **Gradient Scaling**: Para mixed precision
- ✅ **Memory Optimization**: Pin memory e non-blocking transfers

## 📈 Dicas de Performance

### Para Melhorar Acurácia
1. Aumente o número de épocas
2. Use o modelo `attention_cnn` ou `resnet`
3. Ajuste o learning rate (tente 0.0001 ou 0.0005)
4. Aumente data augmentation

### Para Treinar Mais Rápido
1. Use GPU se disponível
2. Aumente `batch_size` (se memória permitir)
3. Aumente `num_workers` (4-8 geralmente ideal)
4. Use mixed precision training
5. Reduza resolução de features se necessário

### Para Economizar Memória
1. Reduza `batch_size`
2. Use modelo `cnn` ao invés de `resnet`
3. Reduza número de mel bands
4. Desative mixed precision se causar problemas

## 🐛 Troubleshooting

### Erro: "Out of Memory"
- Reduza `batch_size`
- Reduza `num_workers`
- Use CPU ao invés de GPU

### Erro: "DataLoader Workers"
- No Windows, defina `num_workers=0`
- Ou use: `persistent_workers=False`

### Modelo não aprende (loss não diminui)
- Reduza learning rate
- Verifique balanceamento de classes
- Aumente número de épocas
- Verifique data augmentation (pode estar muito agressivo)

## 📝 Licença

Este projeto é open source e está disponível sob a licença MIT.

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## 📧 Contato

Para dúvidas ou sugestões, abra uma issue no repositório.

---

**Desenvolvido com ❤️ usando PyTorch e as melhores práticas de Deep Learning**


