"""
run_lora.py

Точка входа для запуска LoRA обучения.

Использование:
    python run_lora.py --config configs/gender.yaml
    python run_lora.py --config configs/age.yaml
"""

import argparse
import yaml
from src.models.lora_trainer import train


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning для транзакционных данных")
    parser.add_argument("--config", required=True, help="Путь к конфигу (configs/gender.yaml)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    model_name   = config["lora"]["model_name"]
    metric       = config["dataset"]["metric"]
    print(f"Датасет:  {dataset_name}")
    print(f"Модель:   {model_name}")
    print(f"Метрика:  {metric}")
    print(f"Выход:    {config['output']['base_dir']}/")
    print()

    train(config)


if __name__ == "__main__":
    main()
