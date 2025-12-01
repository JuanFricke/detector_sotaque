# 🎯 INSTRUÇÕES DE USO - Detector de Sotaque Brasileiro

## ✅ Sistema Completo Criado!

Seu detector de sotaque brasileiro está pronto! Aqui está tudo que foi criado:

## 📁 Arquivos Criados

### Arquivos Principais
- **`dataset.py`** - Dataset customizado com DataLoader otimizado
- **`models.py`** - 4 arquiteturas de modelos (CNN, ResNet, Attention CNN, LSTM)
- **`train.py`** - Script de treinamento com otimizações
- **`predict.py`** - Script de inferência/predição
- **`main.py`** - Interface CLI para todos os comandos
- **`config.py`** - Configurações centralizadas
- **`utils.py`** - Funções utilitárias
- **`analyze_data.py`** - Análise exploratória dos dados

### Documentação
- **`README.md`** - Documentação completa
- **`QUICKSTART.md`** - Guia rápido de uso
- **`requirements.txt`** - Dependências do projeto

### Outros
- **`.gitignore`** - Arquivos a ignorar no Git
- **`test_installation.py`** - Script de teste
- **`experiments/`** - Diretório para salvar experimentos

---

## 🚀 COMO USAR

### Passo 1: Instalar Dependências

Abra o terminal na pasta do projeto e execute:

```bash
pip install -r requirements.txt
```

**Nota**: Isso pode levar alguns minutos. Se estiver em um ambiente virtual, ative-o primeiro.

### Passo 2: Testar Instalação (Opcional)

```bash
python test_installation.py
```

Este comando verificará se tudo está instalado corretamente.

### Passo 3: Analisar os Dados

```bash
python main.py analyze
```

Isso criará visualizações e estatísticas do dataset na pasta `data_analysis/`.

### Passo 4: Treinar o Modelo

**Opção 1: Treinamento Básico**
```bash
python main.py train
```

**Opção 2: Treinamento Customizado**
```bash
python main.py train --model attention_cnn --epochs 50 --batch-size 16
```

**Nota**: O treinamento pode levar de 30 minutos a algumas horas dependendo do seu hardware.

### Passo 5: Fazer Predições

Após o treinamento, use o modelo para classificar novos áudios:

```bash
python main.py predict --checkpoint experiments/[NOME_DO_EXPERIMENTO]/best_model.pth --input audio.wav
```

Para ver seus experimentos:
```bash
python main.py list-experiments
```

---

## 🎨 Características Implementadas

### ✅ Processamento de Áudio
- Carregamento e normalização automática
- Extração de múltiplas features (MFCC, Mel-Spectrogram, Chroma, etc.)
- Ajuste automático de comprimento

### ✅ Data Augmentation
- Time stretching (variação de velocidade)
- Pitch shifting (mudança de tom)
- Adição de ruído gaussiano
- Aplicado apenas no conjunto de treino

### ✅ Modelos de Deep Learning
1. **CNN** - Rede convolucional básica (rápida)
2. **ResNet** - Rede residual profunda (precisa)
3. **Attention CNN** - CNN com atenção (recomendada)
4. **LSTM** - Rede recorrente (sequências temporais)

### ✅ Otimizações de Performance
- **Multithreading**: DataLoader com workers paralelos
- **Mixed Precision Training**: Treinamento 2x mais rápido em GPUs
- **Pin Memory**: Transferências otimizadas para GPU
- **Persistent Workers**: Workers mantidos vivos entre épocas
- **Gradient Scaling**: Para estabilidade em mixed precision

### ✅ Treinamento Inteligente
- Early stopping automático
- Learning rate scheduling
- Checkpoint do melhor modelo
- Validação a cada época
- Métricas detalhadas (Accuracy, F1, Precision, Recall)

### ✅ Visualizações
- Gráficos de loss e accuracy
- Matriz de confusão
- Análise exploratória dos dados
- Comparação de modelos

### ✅ Interface Amigável
- CLI com comandos intuitivos
- Documentação completa
- Mensagens de progresso
- Tratamento de erros

---

## 📊 Estrutura do Dataset

O dataset contém:
- **819 arquivos de áudio** (.wav)
- **Metadados** com informações de:
  - Estado de nascimento
  - Estado atual
  - Idade
  - Gênero
  - Profissão
  - Texto falado

O modelo pode ser treinado para classificar sotaque por:
- Estado de nascimento (`--label-column birth_state`)
- Estado atual (`--label-column current_state`)

---

## 🎓 Exemplos de Comandos

### Ver modelos disponíveis
```bash
python main.py list-models
```

### Treinar com GPU (se disponível)
```bash
python main.py train --model attention_cnn --batch-size 32 --workers 4
```

### Treinar apenas com CPU
```bash
python main.py train --model cnn --batch-size 8 --workers 0
```

### Predizer múltiplos áudios
```bash
python main.py predict --checkpoint experiments/modelo/best_model.pth --input pasta_com_audios/ --output resultados.json
```

### Treinar por mais tempo (melhor qualidade)
```bash
python main.py train --model resnet --epochs 100 --patience 20
```

---

## 💡 Dicas Importantes

### Para Melhor Performance
1. Use GPU se disponível (CUDA)
2. Aumente `num_workers` conforme suas CPUs
3. Use o modelo `attention_cnn` ou `resnet`
4. Aumente o número de épocas
5. Experimente diferentes learning rates

### Para Economizar Recursos
1. Use modelo `cnn` (menor)
2. Reduza `batch_size`
3. Use `num_workers=0` ou `1`
4. Desative mixed precision

### Para Melhor Acurácia
1. Aumente data augmentation
2. Use early stopping com paciência maior
3. Experimente diferentes modelos
4. Ajuste learning rate (tente 0.0001 ou 0.0005)
5. Balance as classes se necessário

---

## 🐛 Soluções para Problemas Comuns

### Erro: "Out of Memory"
```bash
# Solução: Reduza batch size
python main.py train --batch-size 8
```

### Erro: "CUDA out of memory"
```bash
# Solução 1: Reduza batch size
python main.py train --batch-size 4

# Solução 2: Use CPU
python main.py train --device cpu
```

### Erro com DataLoader workers no Windows
```bash
# Solução: Use workers=0
python main.py train --workers 0
```

### Modelo não aprende (loss não diminui)
```bash
# Solução: Reduza learning rate
python main.py train --lr 0.0001
```

### Treinamento muito lento
```bash
# Verifique se está usando GPU
python test_installation.py

# Se não tiver GPU, use modelo menor
python main.py train --model cnn
```

---

## 📈 Resultados Esperados

Dependendo do modelo e configurações:
- **Acurácia de Validação**: 60-85%
- **F1-Score**: 0.6-0.8
- **Tempo de Treinamento**: 30 min - 3 horas

**Nota**: Resultados variam conforme hardware e hiperparâmetros.

---

## 🔄 Próximos Passos

1. ✅ Instale as dependências
2. ✅ Analise os dados
3. ✅ Treine seu primeiro modelo
4. ✅ Avalie os resultados
5. ✅ Experimente diferentes configurações
6. ✅ Use o modelo para classificar áudios reais

---

## 📚 Documentação Completa

- **README.md** - Documentação técnica completa
- **QUICKSTART.md** - Guia rápido e objetivo
- **Este arquivo** - Instruções de uso

---

## 🤝 Suporte

Se encontrar problemas:
1. Verifique a documentação
2. Execute `python test_installation.py`
3. Consulte a seção de troubleshooting
4. Verifique se as dependências estão instaladas

---

## 🎉 Pronto para Começar!

Seu sistema está completo e pronto para uso! Comece com:

```bash
# 1. Instalar
pip install -r requirements.txt

# 2. Analisar
python main.py analyze

# 3. Treinar
python main.py train

# 4. Usar
python main.py predict --checkpoint [MODELO] --input [AUDIO]
```

**Boa sorte com seu detector de sotaque! 🎙️🇧🇷**


