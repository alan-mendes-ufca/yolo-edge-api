# %%
# Célula 1 --- Patch do torch.load e confirmação da GPU
import torch

_orig_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

print("CUDA disponível:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# %%
# Célula 2 --- Treinamento com GPU
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("models/yolov8n.pt")
    results = model.train(
        data="dataset/epi-detection/data.yaml",
        epochs=100,
        imgsz=640,
        device=0,
        patience=20,
        project="runs",
        name="epi-detection",
    )
    print("Pesos salvos em:", results.save_dir)

    # Copia automaticamente o melhor peso para models/yolo-epi.pt
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        dest = Path("models/yolo-epi.pt")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, dest)
        print(f"[OK] Modelo promovido com sucesso para: {dest}")
