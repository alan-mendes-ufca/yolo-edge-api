# Tabela de Resultados dos Experimentos de Pré-processamento

Resultados obtidos a partir da execução dos scripts contidos em [`preprocessing/experiments`](preprocessing/experiments) utilizando o modelo YOLOv8n e o dataset de detecção de EPIs (`epi-v1` e `epi-v1-dark`).

---

## Tabela Consolidada

| Experimento | Configuração | mAP@0.5 (val) | Δ vs baseline | Observação |
| :--- | :--- | :---: | :---: | :--- |
| **E1-A (baseline)** | BGR sem conversão | `0.0131` | — | Imagens passadas em BGR nativo do OpenCV; o modelo espera RGB, sofrendo forte perda de acurácia por inversão dos canais. |
| **E1-B** | RGB correto | `0.0254` | `+0.0123` | Conversão `cv2.COLOR_BGR2RGB` alinha o espaço de cores com o treinamento do YOLOv8, praticamente dobrando o mAP@0.5 (+93.9%). |
| **E2-A** | Resize simples (distorção) | `0.0147` | `+0.0016` | Redimensionamento direto para 416×416. Como as imagens do dataset já possuem proporção 1:1 (640×640), não houve distorção de aspecto. |
| **E2-B** | Letterbox correto | `0.0147` | `+0.0016` | Redimensionamento proporcional com preenchimento de bordas (padding). Garante que entradas de qualquer aspecto (ex: 16:9) mantenham a geometria dos objetos. |
| **E3-A** | Sem filtro (baseline) | `0.0254` | — | Imagem RGB sem aplicação de filtros espaciais. Serve como linha de base para análise do impacto de suavização. |
| **E3-B** | GaussianBlur 3×3, σ=0.8 | `0.0188` | `-0.0066` | Suavização Gaussiana leve atenua ruídos de alta frequência, mas remove gradientes finos essenciais para bordas de capacetes e coletes. (Custo: 0.25 ms) |
| **E3-C** | GaussianBlur 5×5, σ=1.5 | `0.0142` | `-0.0112` | Suavização mais agressiva causa borramento severo das bordas dos EPIs, reduzindo drasticamente o mAP@0.5. (Custo: 0.35 ms) |
| **E3-D** | medianBlur kernel=3 | `0.0154` | `-0.0100` | Filtro de mediana preserva mais bordas que o Gaussiano 5×5, mas ainda degrada a acurácia em comparação com a imagem sem filtro. (Custo: 0.16 ms) |
| **E4-A** | Sem equalização | `0.0188` | — | Imagens subexpostas artificialmente (curva γ=2.2 simulando iluminação desfavorável) sem correção de contraste. |
| **E4-B** | equalizeHist (global) | `0.0219` | `+0.0031` | Equalização global de histograma no canal V (HSV); melhora o contraste geral, porém tende a amplificar ruído e superexpor áreas claras. (Custo: 2.5 ms) |
| **E4-C** | CLAHE clipLimit=2, tile=8 | `0.0257` | `+0.0069` | Equalização adaptativa local com corte de contraste (canal L* do LAB); recupera objetos em sombras sem saturação excessiva (+36.7% de ganho sobre E4-A). (Custo: 5.5 ms) |

---

## Detalhamento dos Experimentos

### E1 — Espaço de Cor (`e1_color_space.py`)
- **Objetivo**: Avaliar a sensibilidade do modelo à ordem dos canais de cores (BGR padrão do OpenCV vs RGB padrão de treinamento do PyTorch/Ultralytics).
- **Resultados**:
  - `E1-A` (BGR puro): `0.0131` mAP@0.5
  - `E1-B` (RGB via `cv2.cvtColor`): `0.0254` mAP@0.5 (`+0.0123`)
  - `E1-C` (RGB via slice NumPy `[:, :, ::-1]`): `0.0254` mAP@0.5 (`+0.0123`)
- **Conclusão**: A conversão BGR → RGB é estritamente obrigatória antes de submeter os frames à inferência.

### E2 — Redimensionamento (`e2_resize.py`)
- **Objetivo**: Comparar redimensionamento por interpolação direta com distorção (*naive resize*) vs redimensionamento com preservação de aspect ratio e padding (*letterbox*).
- **Resultados**:
  - `E2-A` (Resize simples 416×416): `0.0147` mAP@0.5
  - `E2-B` (Letterbox 416×416): `0.0147` mAP@0.5
- **Conclusão**: No dataset de teste (`640×640`, proporção 1:1), ambos geram o mesmo resultado pois o letterbox não precisou adicionar barras de padding. Em produção, com fluxos RTSP / câmeras 16:9 (`1920×1080` ou `1280×720`), o letterbox com ajuste de coordenadas (`adjust_boxes`) é indispensável para evitar achatamento geométrico.

### E3 — Filtros de Suavização (`e3_filters.py`)
- **Objetivo**: Medir se a redução de ruído por filtros espaciais auxilia ou prejudica a detecção de objetos em modelos baseados em CNNs/YOLO.
- **Resultados**:
  - `E3-A` (Sem filtro): `0.0254` mAP@0.5
  - `E3-B` (GaussianBlur 3×3, σ=0.8): `0.0188` mAP@0.5 (`-0.0066`)
  - `E3-C` (GaussianBlur 5×5, σ=1.5): `0.0142` mAP@0.5 (`-0.0112`)
  - `E3-D` (medianBlur k=3): `0.0154` mAP@0.5 (`-0.0100`)
- **Benchmark de Latência (640×480)**:
  - `cvtColor apenas`: 0.04 ms/frame
  - `medianBlur (k=3)`: 0.16 ms/frame
  - `GaussianBlur (3×3)`: 0.25 ms/frame
  - `GaussianBlur (5×5)`: 0.35 ms/frame
  - `bilateralFilter`: 19.32 ms/frame (inviável para edge real-time)
- **Conclusão**: Modelos de detecção modernos já possuem robustez interna a ruídos comuns. Aplicar filtros passa-baixa adicionais borra arestas e detalhes de textura, degradando a performance. Portanto, filtros de desfoque não devem ser aplicados por padrão.

### E4 — Contraste e Iluminação (`e4_contrast.py` e `e4_generate_dark.py`)
- **Objetivo**: Avaliar métodos de equalização de histograma para recuperar a detecção em condições de baixa luminosidade (imagens escurecidas por curva de gamma $\gamma=2.2$).
- **Resultados no Dataset Subexposto**:
  - `E4-A` (Sem equalização / baixa iluminação): `0.0188` mAP@0.5
  - `E4-B` (Equalização Global no canal V do HSV): `0.0219` mAP@0.5 (`+0.0031`)
  - `E4-C` (CLAHE canal L* do LAB, clip=2, tile=8): `0.0257` mAP@0.5 (`+0.0069`)
  - `E4-D` (CLAHE canal V do HSV, clip=2, tile=8): `0.0235` mAP@0.5 (`+0.0047`)
  - `E4-F` (CLAHE canal L* do LAB, clip=4, tile=8): `0.0257` mAP@0.5 (`+0.0069`)
- **Conclusão**: O algoritmo **CLAHE** aplicado no canal de luminância do espaço LAB (`L*`) proporcionou o melhor ganho de acurácia (+36.7% relativo a E4-A), equalizando seletivamente regiões escuras sem estourar as partes claras da cena.

