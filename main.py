"""
Script principal para gerenciar o detector de sotaque
"""
import argparse
import os
import sys
from datetime import datetime
import glob
import json


def analyze_command(args):
    """Executa análise exploratória dos dados"""
    from analyze_data import analyze_dataset
    
    print("\n🔍 Iniciando análise exploratória dos dados...")
    analyze_dataset(args.csv_path, args.output_dir)
    print("\n✅ Análise concluída!")


def train_command(args):
    """Executa treinamento do modelo"""
    from train import AccentDetectorTrainer
    
    print(f"\n🚀 Iniciando treinamento do modelo: {args.model}")
    print(f"   Épocas: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Workers: {args.workers}")
    
    trainer = AccentDetectorTrainer(
        model_name=args.model,
        num_classes=None,
        csv_path=args.csv_path,
        audio_base_path=args.audio_path,
        batch_size=args.batch_size,
        num_workers=args.workers,
        learning_rate=args.lr,
        device=args.device,
        mixed_precision=not args.no_mixed_precision,
        label_column=args.label_column
    )
    
    trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience
    )
    
    if not args.skip_eval:
        print("\n📊 Avaliando modelo no conjunto de teste...")
        trainer.evaluate(use_test_set=True)
    
    print("\n✅ Treinamento concluído!")


def predict_command(args):
    """Executa predição em áudio(s)"""
    from predict import AccentPredictor
    import json
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Erro: Checkpoint não encontrado: {args.checkpoint}")
        return
    
    predictor = AccentPredictor(args.checkpoint, device=args.device)
    
    # Verificar se é um arquivo ou diretório
    if os.path.isfile(args.input):
        # Predição única
        print(f"\n🎙️ Fazendo predição para: {args.input}")
        result = predictor.predict(args.input, return_probs=True)
        predictor.print_prediction(result)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"\n💾 Resultado salvo em: {args.output}")
    
    elif os.path.isdir(args.input):
        # Predição em lote
        import glob
        audio_files = glob.glob(os.path.join(args.input, "*.wav"))
        
        if not audio_files:
            print(f"❌ Nenhum arquivo .wav encontrado em: {args.input}")
            return
        
        print(f"\n🎙️ Fazendo predições para {len(audio_files)} áudios...")
        results = predictor.predict_batch(audio_files, return_probs=True)
        
        # Mostrar resultados
        for result in results:
            predictor.print_prediction(result)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\n💾 Resultados salvos em: {args.output}")
    
    else:
        print(f"❌ Erro: Caminho não encontrado: {args.input}")


def list_models_command(args):
    """Lista modelos disponíveis"""
    from config import AVAILABLE_MODELS
    
    print("\n" + "="*60)
    print("MODELOS DISPONÍVEIS")
    print("="*60)
    
    for model_id, info in AVAILABLE_MODELS.items():
        print(f"\n📊 {model_id}")
        print(f"   Nome: {info['name']}")
        print(f"   Descrição: {info['description']}")
        print(f"   Parâmetros: {info['params']}")
        print(f"   Velocidade: {info['speed']}")
        print(f"   Acurácia: {info['accuracy']}")
    
    print("\n" + "="*60)


def list_experiments_command(args):
    """Lista experimentos salvos"""
    experiments_dir = "experiments"
    
    if not os.path.exists(experiments_dir):
        print(f"❌ Diretório de experimentos não encontrado: {experiments_dir}")
        return
    
    experiments = [d for d in os.listdir(experiments_dir) 
                  if os.path.isdir(os.path.join(experiments_dir, d))]
    
    if not experiments:
        print("❌ Nenhum experimento encontrado")
        return
    
    print("\n" + "="*60)
    print("EXPERIMENTOS SALVOS")
    print("="*60)
    
    for exp in sorted(experiments, reverse=True):
        exp_path = os.path.join(experiments_dir, exp)
        best_model_path = os.path.join(exp_path, "best_model.pth")
        info_path = os.path.join(exp_path, "training_info.json")
        
        print(f"\n📁 {exp}")
        
        if os.path.exists(info_path):
            import json
            with open(info_path, 'r') as f:
                info = json.load(f)
            
            print(f"   Modelo: {info.get('model_name', 'N/A')}")
            print(f"   Classes: {info.get('num_classes', 'N/A')}")
            print(f"   Melhor Acurácia: {info.get('best_val_acc', 'N/A'):.2f}%")
            print(f"   Melhor F1: {info.get('best_val_f1', 'N/A'):.4f}")
            print(f"   Épocas: {info.get('num_epochs', 'N/A')}")
        
        if os.path.exists(best_model_path):
            print(f"   ✅ Modelo treinado disponível")
        else:
            print(f"   ⚠️ Modelo não encontrado")
        
        print(f"   Caminho: {exp_path}")
    
    print("\n" + "="*60)


def interactive_menu():
    """Menu interativo para análise de áudios"""
    print("\n" + "="*70)
    print("🎙️  DETECTOR DE SOTAQUE BRASILEIRO - MENU INTERATIVO")
    print("="*70)
    
    # 1. Listar experimentos disponíveis
    experiments_dir = "experiments"
    if not os.path.exists(experiments_dir):
        print("\n❌ Erro: Diretório de experimentos não encontrado!")
        print(f"   Crie o diretório '{experiments_dir}' e treine alguns modelos primeiro.")
        return
    
    experiments = []
    for exp_name in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, exp_name)
        if os.path.isdir(exp_path):
            best_model_path = os.path.join(exp_path, "best_model.pth")
            if os.path.exists(best_model_path):
                experiments.append({
                    'name': exp_name,
                    'path': exp_path,
                    'checkpoint': best_model_path
                })
    
    if not experiments:
        print("\n❌ Erro: Nenhum experimento treinado encontrado!")
        print("   Treine um modelo primeiro usando: python main.py train")
        return
    
    # Mostrar experimentos disponíveis
    print("\n📊 EXPERIMENTOS DISPONÍVEIS:")
    print("-" * 70)
    
    for idx, exp in enumerate(experiments, 1):
        info_path = os.path.join(exp['path'], "training_info.json")
        
        print(f"\n[{idx}] {exp['name']}")
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
            
            model_name = info.get('model_name', 'N/A')
            num_classes = info.get('num_classes', 'N/A')
            best_acc = info.get('best_val_acc', 0)
            best_f1 = info.get('best_val_f1', 0)
            
            print(f"    Modelo: {model_name}")
            print(f"    Classes: {num_classes}")
            print(f"    Melhor Acurácia: {best_acc:.2f}%")
            print(f"    Melhor F1-Score: {best_f1:.4f}")
    
    print("\n" + "-" * 70)
    
    # 2. Selecionar experimento
    while True:
        try:
            exp_choice = input(f"\n🔍 Escolha um experimento [1-{len(experiments)}] ou 'q' para sair: ").strip()
            
            if exp_choice.lower() == 'q':
                print("\n👋 Até logo!")
                return
            
            exp_idx = int(exp_choice) - 1
            if 0 <= exp_idx < len(experiments):
                selected_exp = experiments[exp_idx]
                break
            else:
                print(f"❌ Opção inválida! Escolha entre 1 e {len(experiments)}")
        except ValueError:
            print("❌ Por favor, digite um número válido ou 'q' para sair")
    
    print(f"\n✅ Experimento selecionado: {selected_exp['name']}")
    
    # 3. Listar áudios disponíveis no real_data
    real_data_dir = "real_data"
    if not os.path.exists(real_data_dir):
        print(f"\n❌ Erro: Diretório '{real_data_dir}' não encontrado!")
        return
    
    audio_files = glob.glob(os.path.join(real_data_dir, "*.wav"))
    
    if not audio_files:
        print(f"\n❌ Erro: Nenhum arquivo .wav encontrado em '{real_data_dir}'")
        return
    
    # Ordenar por nome
    audio_files.sort()
    
    print(f"\n🎵 ÁUDIOS DISPONÍVEIS em '{real_data_dir}':")
    print("-" * 70)
    
    for idx, audio_path in enumerate(audio_files, 1):
        filename = os.path.basename(audio_path)
        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        print(f"[{idx}] {filename}")
        print(f"    Tamanho: {file_size:.2f} MB")
    
    print("\n[A] Analisar TODOS os áudios")
    print("-" * 70)
    
    # 4. Selecionar áudio(s)
    while True:
        audio_choice = input(f"\n🎙️  Escolha um áudio [1-{len(audio_files)}], 'A' para todos, ou 'q' para sair: ").strip()
        
        if audio_choice.lower() == 'q':
            print("\n👋 Até logo!")
            return
        
        if audio_choice.upper() == 'A':
            selected_audios = audio_files
            break
        
        try:
            audio_idx = int(audio_choice) - 1
            if 0 <= audio_idx < len(audio_files):
                selected_audios = [audio_files[audio_idx]]
                break
            else:
                print(f"❌ Opção inválida! Escolha entre 1 e {len(audio_files)}")
        except ValueError:
            print("❌ Por favor, digite um número válido, 'A' para todos, ou 'q' para sair")
    
    # 5. Executar predição
    print("\n" + "="*70)
    print("🚀 INICIANDO ANÁLISE...")
    print("="*70)
    
    from predict import AccentPredictor
    
    try:
        predictor = AccentPredictor(selected_exp['checkpoint'])
        
        if len(selected_audios) == 1:
            # Predição única
            audio_path = selected_audios[0]
            print(f"\n🎙️  Analisando: {os.path.basename(audio_path)}")
            print("-" * 70)
            
            result = predictor.predict(audio_path, return_probs=True)
            predictor.print_prediction(result)
            
            # Perguntar se deseja salvar
            save_choice = input("\n💾 Deseja salvar o resultado? [s/N]: ").strip().lower()
            if save_choice == 's':
                output_file = f"prediction_{os.path.splitext(os.path.basename(audio_path))[0]}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)
                print(f"✅ Resultado salvo em: {output_file}")
        
        else:
            # Predição em lote
            print(f"\n🎙️  Analisando {len(selected_audios)} áudios...")
            print("-" * 70)
            
            results = predictor.predict_batch(selected_audios, return_probs=True)
            
            # Mostrar resultados
            for result in results:
                predictor.print_prediction(result)
                print("-" * 70)
            
            # Perguntar se deseja salvar
            save_choice = input("\n💾 Deseja salvar os resultados? [s/N]: ").strip().lower()
            if save_choice == 's':
                output_file = f"predictions_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                print(f"✅ Resultados salvos em: {output_file}")
        
        print("\n" + "="*70)
        print("✅ ANÁLISE CONCLUÍDA!")
        print("="*70)
        
        # Perguntar se deseja continuar
        continue_choice = input("\n🔄 Deseja analisar outro áudio? [s/N]: ").strip().lower()
        if continue_choice == 's':
            interactive_menu()
    
    except Exception as e:
        print(f"\n❌ Erro durante a análise: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="🎙️ Detector de Sotaque Brasileiro - Sistema de IA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Menu interativo
  python main.py interactive

  # Análise exploratória dos dados
  python main.py analyze

  # Treinar modelo
  python main.py train --model attention_cnn --epochs 50

  # Fazer predição
  python main.py predict --checkpoint experiments/modelo/best_model.pth --input audio.wav

  # Listar modelos disponíveis
  python main.py list-models

  # Listar experimentos salvos
  python main.py list-experiments
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a executar')
    
    # Comando: interactive (NOVO)
    subparsers.add_parser('interactive', help='Menu interativo para análise de áudios')
    
    # Comando: analyze
    analyze_parser = subparsers.add_parser('analyze', help='Análise exploratória dos dados')
    analyze_parser.add_argument('--csv-path', default='sotaque-brasileiro-data/sotaque-brasileiro.csv',
                               help='Caminho para o CSV')
    analyze_parser.add_argument('--output-dir', default='data_analysis',
                               help='Diretório de saída')
    
    # Comando: train
    train_parser = subparsers.add_parser('train', help='Treinar modelo')
    train_parser.add_argument('--model', default='attention_cnn',
                            choices=['cnn', 'resnet', 'attention_cnn', 'lstm'],
                            help='Modelo a treinar')
    train_parser.add_argument('--csv-path', default='sotaque-brasileiro-data/sotaque-brasileiro.csv',
                            help='Caminho para o CSV')
    train_parser.add_argument('--audio-path', default='sotaque-brasileiro-data',
                            help='Caminho base dos áudios')
    train_parser.add_argument('--epochs', type=int, default=50,
                            help='Número de épocas')
    train_parser.add_argument('--batch-size', type=int, default=16,
                            help='Tamanho do batch')
    train_parser.add_argument('--workers', type=int, default=4,
                            help='Número de workers para DataLoader')
    train_parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate')
    train_parser.add_argument('--patience', type=int, default=15,
                            help='Paciência para early stopping')
    train_parser.add_argument('--device', default=None,
                            help='Device (cuda ou cpu)')
    train_parser.add_argument('--no-mixed-precision', action='store_true',
                            help='Desabilitar mixed precision')
    train_parser.add_argument('--label-column', default='birth_state',
                            choices=['birth_state', 'current_state'],
                            help='Coluna de label')
    train_parser.add_argument('--skip-eval', action='store_true',
                            help='Pular avaliação final')
    
    # Comando: predict
    predict_parser = subparsers.add_parser('predict', help='Fazer predição')
    predict_parser.add_argument('--checkpoint', required=True,
                              help='Caminho para o checkpoint do modelo')
    predict_parser.add_argument('--input', required=True,
                              help='Caminho para áudio ou diretório de áudios')
    predict_parser.add_argument('--output', default=None,
                              help='Caminho para salvar resultado JSON')
    predict_parser.add_argument('--device', default=None,
                              help='Device (cuda ou cpu)')
    
    # Comando: list-models
    subparsers.add_parser('list-models', help='Listar modelos disponíveis')
    
    # Comando: list-experiments
    subparsers.add_parser('list-experiments', help='Listar experimentos salvos')
    
    args = parser.parse_args()
    
    # Se nenhum comando foi fornecido, iniciar menu interativo
    if args.command is None:
        interactive_menu()
        return
    
    # Executar comando
    if args.command == 'interactive':
        interactive_menu()
    elif args.command == 'analyze':
        analyze_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'list-models':
        list_models_command(args)
    elif args.command == 'list-experiments':
        list_experiments_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


