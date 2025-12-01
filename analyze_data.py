"""
Script para análise exploratória dos dados de sotaque brasileiro
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def analyze_dataset(csv_path: str, save_dir: str = "data_analysis"):
    """
    Realiza análise exploratória do dataset
    
    Args:
        csv_path: Caminho para o CSV com metadados
        save_dir: Diretório para salvar visualizações
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Carregar dados
    print("Carregando dados...")
    df = pd.read_csv(csv_path)
    
    print(f"\n{'='*60}")
    print("INFORMAÇÕES GERAIS DO DATASET")
    print(f"{'='*60}")
    print(f"Total de amostras: {len(df)}")
    print(f"Número de colunas: {len(df.columns)}")
    print(f"\nColunas disponíveis:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Estatísticas básicas
    print(f"\n{'='*60}")
    print("ESTATÍSTICAS BÁSICAS")
    print(f"{'='*60}")
    
    # Estados
    print("\nEstados de Nascimento:")
    birth_states = df['birth_state'].value_counts()
    print(birth_states)
    print(f"\nNúmero de estados diferentes: {len(birth_states)}")
    
    print("\nEstados Atuais:")
    current_states = df['current_state'].value_counts()
    print(current_states)
    
    # Gênero
    print("\nDistribuição por Gênero:")
    gender_dist = df['gender'].value_counts()
    print(gender_dist)
    
    # Idade
    print("\nEstatísticas de Idade:")
    print(df['age'].describe())
    
    # Criar visualizações
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Distribuição de Estados de Nascimento
    plt.subplot(3, 3, 1)
    birth_states_top = birth_states.head(15)
    plt.barh(range(len(birth_states_top)), birth_states_top.values, color='steelblue')
    plt.yticks(range(len(birth_states_top)), birth_states_top.index)
    plt.xlabel('Número de Amostras')
    plt.title('Top 15 Estados de Nascimento', fontweight='bold', fontsize=12)
    plt.gca().invert_yaxis()
    
    # 2. Distribuição de Estados Atuais
    plt.subplot(3, 3, 2)
    current_states_top = current_states.head(15)
    plt.barh(range(len(current_states_top)), current_states_top.values, color='coral')
    plt.yticks(range(len(current_states_top)), current_states_top.index)
    plt.xlabel('Número de Amostras')
    plt.title('Top 15 Estados Atuais', fontweight='bold', fontsize=12)
    plt.gca().invert_yaxis()
    
    # 3. Distribuição por Gênero
    plt.subplot(3, 3, 3)
    colors = ['#ff9999', '#66b3ff']
    plt.pie(gender_dist.values, labels=gender_dist.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    plt.title('Distribuição por Gênero', fontweight='bold', fontsize=12)
    
    # 4. Distribuição de Idade
    plt.subplot(3, 3, 4)
    plt.hist(df['age'].dropna(), bins=30, color='green', alpha=0.7, edgecolor='black')
    plt.xlabel('Idade')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Idade', fontweight='bold', fontsize=12)
    plt.axvline(df['age'].mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {df["age"].mean():.1f}')
    plt.legend()
    
    # 5. Idade por Gênero
    plt.subplot(3, 3, 5)
    gender_ages = df.groupby('gender')['age'].apply(list)
    plt.boxplot([gender_ages[gender] for gender in gender_ages.index],
               labels=gender_ages.index)
    plt.ylabel('Idade')
    plt.title('Idade por Gênero', fontweight='bold', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 6. Top 15 Profissões
    plt.subplot(3, 3, 6)
    professions = df['profession'].value_counts().head(15)
    plt.barh(range(len(professions)), professions.values, color='purple', alpha=0.7)
    plt.yticks(range(len(professions)), professions.index, fontsize=8)
    plt.xlabel('Número de Amostras')
    plt.title('Top 15 Profissões', fontweight='bold', fontsize=12)
    plt.gca().invert_yaxis()
    
    # 7. Top 15 Cidades de Nascimento
    plt.subplot(3, 3, 7)
    birth_cities = df['birth_city'].value_counts().head(15)
    plt.bar(range(len(birth_cities)), birth_cities.values, color='orange', alpha=0.7)
    plt.xticks(range(len(birth_cities)), birth_cities.index, rotation=45, ha='right', fontsize=8)
    plt.ylabel('Número de Amostras')
    plt.title('Top 15 Cidades de Nascimento', fontweight='bold', fontsize=12)
    
    # 8. Top 15 Cidades Atuais
    plt.subplot(3, 3, 8)
    current_cities = df['current_city'].value_counts().head(15)
    plt.bar(range(len(current_cities)), current_cities.values, color='teal', alpha=0.7)
    plt.xticks(range(len(current_cities)), current_cities.index, rotation=45, ha='right', fontsize=8)
    plt.ylabel('Número de Amostras')
    plt.title('Top 15 Cidades Atuais', fontweight='bold', fontsize=12)
    
    # 9. Anos na Cidade Atual
    plt.subplot(3, 3, 9)
    years_in_city = df['years_on_current_city'].dropna()
    plt.hist(years_in_city, bins=30, color='brown', alpha=0.7, edgecolor='black')
    plt.xlabel('Anos na Cidade Atual')
    plt.ylabel('Frequência')
    plt.title('Distribuição: Anos na Cidade Atual', fontweight='bold', fontsize=12)
    plt.axvline(years_in_city.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Média: {years_in_city.mean():.1f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dataset_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"\nVisualizações salvas em: {save_dir}/dataset_analysis.png")
    plt.close()
    
    # Análise de balanceamento de classes
    print(f"\n{'='*60}")
    print("ANÁLISE DE BALANCEAMENTO DE CLASSES")
    print(f"{'='*60}")
    
    # Por estado de nascimento
    print("\nBalanceamento por Estado de Nascimento:")
    birth_state_pct = (birth_states / len(df) * 100).round(2)
    for state, pct in birth_state_pct.items():
        print(f"  {state}: {pct}%")
    
    # Calcular métricas de desbalanceamento
    max_class = birth_states.max()
    min_class = birth_states.min()
    imbalance_ratio = max_class / min_class
    
    print(f"\nMaior classe: {max_class} amostras")
    print(f"Menor classe: {min_class} amostras")
    print(f"Razão de desbalanceamento: {imbalance_ratio:.2f}:1")
    
    # Matriz de migração (nascimento -> atual)
    print(f"\n{'='*60}")
    print("ANÁLISE DE MIGRAÇÃO")
    print(f"{'='*60}")
    
    migration = df.groupby(['birth_state', 'current_state']).size().reset_index(name='count')
    migration_pivot = migration.pivot(index='birth_state', columns='current_state', values='count').fillna(0)
    
    # Calcular percentual de migração
    df['migrated'] = df['birth_state'] != df['current_state']
    migration_pct = df['migrated'].value_counts(normalize=True) * 100
    
    print(f"\nPercentual de pessoas que migraram: {migration_pct[True]:.2f}%")
    print(f"Percentual de pessoas na cidade natal: {migration_pct[False]:.2f}%")
    
    # Visualizar matriz de migração (top estados)
    top_states = birth_states.head(10).index
    migration_matrix = migration_pivot.loc[top_states, top_states]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(migration_matrix, annot=True, fmt='.0f', cmap='YlOrRd', 
               cbar_kws={'label': 'Número de Pessoas'})
    plt.title('Matriz de Migração - Top 10 Estados', fontweight='bold', fontsize=14)
    plt.xlabel('Estado Atual', fontsize=12)
    plt.ylabel('Estado de Nascimento', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'migration_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"Matriz de migração salva em: {save_dir}/migration_matrix.png")
    plt.close()
    
    # Análise de textos
    print(f"\n{'='*60}")
    print("ANÁLISE DE TEXTOS")
    print(f"{'='*60}")
    
    # Contar sentenças únicas
    unique_sentences = df['sentence_text'].nunique()
    total_sentences = len(df)
    
    print(f"\nNúmero de sentenças únicas: {unique_sentences}")
    print(f"Total de gravações: {total_sentences}")
    print(f"Média de gravações por sentença: {total_sentences/unique_sentences:.2f}")
    
    # Sentenças mais gravadas
    print("\nTop 5 sentenças mais gravadas:")
    top_sentences = df['sentence_text'].value_counts().head(5)
    for i, (sentence, count) in enumerate(top_sentences.items(), 1):
        print(f"\n{i}. ({count} gravações)")
        print(f"   {sentence[:100]}...")
    
    # Salvar relatório
    print(f"\n{'='*60}")
    print("SALVANDO RELATÓRIO")
    print(f"{'='*60}")
    
    report = {
        'total_samples': len(df),
        'num_states': len(birth_states),
        'num_unique_sentences': unique_sentences,
        'gender_distribution': gender_dist.to_dict(),
        'age_stats': df['age'].describe().to_dict(),
        'migration_percentage': float(migration_pct[True]),
        'class_imbalance_ratio': float(imbalance_ratio),
        'states_distribution': birth_states.to_dict()
    }
    
    import json
    with open(os.path.join(save_dir, 'dataset_report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"Relatório JSON salvo em: {save_dir}/dataset_report.json")
    print(f"\n{'='*60}")
    print("ANÁLISE COMPLETA!")
    print(f"{'='*60}")


if __name__ == "__main__":
    CSV_PATH = "sotaque-brasileiro-data/sotaque-brasileiro.csv"
    
    if not os.path.exists(CSV_PATH):
        print(f"Erro: Arquivo não encontrado: {CSV_PATH}")
    else:
        analyze_dataset(CSV_PATH)


