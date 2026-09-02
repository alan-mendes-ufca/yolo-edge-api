#!/usr/bin/env python3
"""
scripts/validate_model.py
Quality gate: bloqueia o deploy se o mAP@0.5 estiver abaixo do limiar.
Uso: python scripts/validate_model.py [--model models/yolo-epi.pt] [--dataset dataset/epi-detection/data.yaml] [--threshold 0.60]
"""
import argparse
import sys
from pathlib import Path

# Limiar padrão de qualidade (Quality Gate)
DEFAULT_THRESHOLD = 0.60

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="models/yolo-epi.pt",
        help="Caminho para os pesos do modelo YOLO (padrão: models/yolo-epi.pt)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="Limiar mínimo de mAP@0.5 (padrão: 0.60)")
    parser.add_argument("--dataset",   default="dataset/epi-detection/data.yaml",
        help="Caminho para o YAML do dataset de validação (padrão: dataset/epi-detection/data.yaml)")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"[ERRO] Modelo não encontrado: {model_path}")
        sys.exit(1)

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    # Sem dataset explícito, usa o dataset interno do modelo pré-treinado
    if args.dataset:
        print(f"[INFO] Validando com dataset: {args.dataset}")
        metrics = model.val(data=args.dataset, split="val", verbose=False)
    else:
        # Validação rápida com COCO128 (dataset padrão)
        print("[INFO] Validando com COCO128 (dataset padrão)")
        metrics = model.val(data="coco128.yaml", split="val", verbose=False)

    map50 = float(metrics.box.map50)
    print(f"[INFO] mAP@0.5 = {map50:.4f}  |  Limiar: {args.threshold:.4f}")

    if map50 < args.threshold:
        print("[FALHA] mAP abaixo do limiar. Deploy bloqueado.")
        sys.exit(1)

    print("[OK] Quality gate aprovado. Deploy autorizado.")


if __name__ == "__main__":
    main()
