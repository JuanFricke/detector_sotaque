# Experimentos - Detector de Sotaque

Este diretório contém os experimentos e modelos treinados.

## Estrutura

Cada experimento é salvo em um diretório com o formato:
```
{model_name}_{timestamp}/
├── best_model.pth              # Melhor modelo (checkpoint)
├── training_history.png        # Gráficos de treinamento
├── confusion_matrix_teste.png  # Matriz de confusão
├── training_info.json          # Informações do treinamento
└── evaluation_teste.json       # Métricas de avaliação
```

## Listar Experimentos

```bash
python main.py list-experiments
```

## Carregar Modelo

```python
from predict import AccentPredictor

predictor = AccentPredictor('experiments/attention_cnn_20231201/best_model.pth')
result = predictor.predict('audio.wav')
```

## Comparar Experimentos

Use o notebook `notebooks/compare_experiments.ipynb` para comparar múltiplos experimentos.


