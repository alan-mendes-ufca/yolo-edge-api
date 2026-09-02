# yolo-edge-api

O **`yolo-edge-api`** é um projeto de **visão computacional embarcada** (*Edge AI*) focado na aplicação prática de **MLOps** para dispositivos de borda. O projeto estabelece uma base sólida para o treinamento, otimização, deploy contínuo e manutenção de modelos de inteligência artificial em ambientes restritos.

- - -

## Propósito

O projeto nasceu durante o [curso de Edge AI do laboratório maker do PNAAT](https://fit-tecnologia.org.br/pnaat/regioes/) na UFCA, com o objetivo de aplicar conceitos fundamentais de **MLOps** em todo o ciclo de vida de um modelo de visão computacional: 

`Dados (Roboflow/DVC) ➔ Código & Modelo (Git/YOLOv8) ➔ CI/CD (GitHub Actions) ➔ Deploy (Docker/Pi 5) ➔ Observabilidade (Alloy/Grafana)`

### Principais Práticas de MLOps

* **Versionamento desacoplado**: Pesos binários (`.pt`) e datasets volumosos não inflam o repositório Git; o projeto utiliza DVC (*Data Version Control*) integrado a um storage remoto via SSH.
* **Quality Gate rigoroso**: Novos modelos só avançam para o dispositivo após atingirem a métrica mínima estipulada ($mAP@0.5 \ge 0.60$) no conjunto de validação durante o pipeline de integração contínua.
* **Acesso seguro em redes privadas**: Dispositivos de borda frequentemente operam atrás de CGNAT ou redes locais sem portas públicas expostas. Uma malha Tailscale VPN viabiliza a comunicação direta e segura ponto a ponto com os runners do CI/CD.
* **Resiliência e deploy contínuo**: O deploy automatizado via SSH no Raspberry Pi 5 executa verificações de integridade (*health checks*) pós-inicialização, realizando rollback autônomo para o container anterior caso a nova versão falhe.

Ainda, o `yolo-edge-api` documenta uma esteira completa de engenharia: **treinamento e fine-tuning** de dataset customizado para detecção de EPIs, **pipeline de pré-processamento** (*Letterbox, equalização adaptativa CLAHE e análise de filtros espaciais*), **streaming de vídeo** concorrente (*MJPEG*) e **entrega contínua** focada em hardware restrito.

- - -

## Funcionalidades

- **Inferência REST de alta eficiência**: Detecção de objetos em imagens únicas (Base64 ou URL) e em lote (`/predict/batch`).
- **Respostas visuais anotadas**: Endpoints dedicados para renderização e retorno direto de imagens JPEG com as caixas delimitadoras sobrepostas (`/predict/image` e `/predict/camera/image`).
- **Integração com câmera física na borda**: Captura via interface CSI nativa do Raspberry Pi (`rpicam-still`) ou câmeras USB convencionais (V4L2).
- **Streaming de vídeo em tempo real (MJPEG)**: Feed de vídeo ao vivo com inferência sobreposta por frame (`/stream/camera`) e interface web integrada (`/stream/view`).
- **Pipeline de pré-processamento inteligente**: Redimensionamento com Letterbox (preservando o aspect ratio sem distorcer geometrias), ajuste inverso de bboxes (`adjust_boxes`) e equalização CLAHE para baixa luminosidade.
- **Detecção de EPIs**: Suporte a pesos customizados (`yolo-epi.pt`) para detecção de equipamentos de proteção individual (capacetes, coletes, etc.).
- **Versionamento de artefatos com DVC**: Pesos do modelo e datasets sob controle de versão, sem comprometer o histórico do Git.
- **CI/CD com MLOps completo**: 4 jobs no GitHub Actions cobrindo validação de código, quality gate de performance do modelo, build de imagens ARM64 e deploy com rollback no Raspberry Pi 5.
- **Métricas e observabilidade**: Endpoint `/metrics` com totalizadores de requisições e latência média, complementado por logs em JSON estruturado para indexação.

- - -

## Stack principal

- Python 3.11+
- FastAPI
- Uvicorn
- Ultralytics YOLOv8
- PyTorch (CPU-only para ARM64 / CUDA para treino)
- OpenCV (opencv-python-headless)
- Pillow e NumPy
- DVC (Data Version Control com suporte a SSH)
- Docker e Docker Compose
- GitHub Actions (Buildx, QEMU ARM64)
- Tailscale (Mesh VPN)
- Pytest
- Ruff

- - -

## Arquitetura

### Arquitetura do Sistema

```mermaid
flowchart TD
    subgraph DataOps["Dados & Modelo (DVC)"]
        D1[Datasets / EPIs] --> DVC[(DVC Remote via SSH)]
        DVC --> M1[Pesos YOLOv8 .pt]
    end

    subgraph CI["CI/CD Pipeline (GitHub Actions)"]
        G1[Git Push] --> T1[Smoke & Unit Tests]
        T1 --> QG{Quality Gate\nmAP@0.5 >= 0.60}
        QG -- Reprovado --> Fail[Bloqueia Deploy]
        QG -- Aprovado --> B1[Build Docker ARM64]
    end

    subgraph Edge["Ambiente de Borda (Raspberry Pi 5)"]
        VPN[Tailscale VPN Mesh]
        B1 -->|Deploy SSH seguro| VPN
        VPN --> DC[Docker Container]
        
        subgraph Container["yolo-edge-api"]
            API[FastAPI / Rotas HTTP]
            PP[Pré-processamento: Letterbox / CLAHE]
            ST[Stream MJPEG: v1 / v2 / v3]
            API --> PP --> INF[Inferência YOLOv8]
            INF --> ST
        end
        DC --> Container
    end

    subgraph Obs["Observabilidade"]
        Container -->|Métricas & Logs| AL[Grafana Alloy]
        AL --> GF[(Grafana Dashboard)]
    end
```

### Estrutura do repositório

```txt
yolo-edge-api/
├── .github/workflows/   # CI/CD: testes, quality gate de mAP, build ARM64 e deploy
├── app/                 # FastAPI: schemas Pydantic, rotas REST e ciclo de vida do modelo
├── preprocessing/       # Otimizações de entrada (Letterbox, CLAHE) e benchmarks
├── stream/              # Streaming MJPEG (v1 naive, v2 threaded, v3 desacoplado)
├── scripts/             # Automação operacional: Quality Gate, deploy e rollback
├── tests/               # Suíte de testes (smoke, unitários, integração e GPU)
├── models/              # Metadados de rastreio DVC (.dvc) para pesos (.pt)
├── dataset/             # Metadados DVC para datasets de EPIs
├── client/              # Cliente de teste para validação das rotas de inferência
└── apostilas/           # Referências conceituais e notas técnicas da disciplina
```

- - -

## API

| Método   | Rota                    | Descrição                                                                 |
| -------- | ----------------------- | ------------------------------------------------------------------------- |
| `GET`    | `/health`               | Retorna o estado de saúde da aplicação e confirmação do modelo carregado. |
| `GET`    | `/metrics`              | Retorna métricas de volume de requisições, sucessos e latência média.     |
| `POST`   | `/predict`              | Executa inferência sobre imagem (Base64 ou URL) e retorna detecções JSON. |
| `POST`   | `/predict/image`        | Executa inferência sobre imagem enviada e retorna JPEG anotado com bboxes. |
| `POST`   | `/predict/camera`       | Captura foto na câmera física (CSI ou USB) e retorna detecções em JSON.   |
| `GET`    | `/predict/camera/image` | Captura foto na câmera física e retorna imagem JPEG anotada com bboxes.   |
| `GET`    | `/stream/camera`        | Transmite feed contínuo de vídeo MJPEG com detecções desenhadas nos frames.|
| `GET`    | `/stream/view`          | Página HTML interativa para visualização do streaming no navegador.       |
| `POST`   | `/predict/batch`        | Processa múltiplas imagens Base64 em lote retornando detecções e tempo.   |

Documentação interativa Swagger UI disponível em `/docs` e ReDoc em `/redoc`.

- - -

## Como rodar localmente

### Requisitos

- Python 3.11+
- Git
- Docker e Docker Compose
- DVC (para sincronização de pesos e datasets)

### Instalar dependências

```bash
python -m venv .venv
# Ativar o ambiente virtual:
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Sincronizar modelo e pesos

```bash
dvc pull
```

### Iniciar ambiente de desenvolvimento

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Ou utilizando Docker Compose:

```bash
docker compose up -d yolo-api
```

A aplicação fica disponível em `http://localhost:8000` (documentação interativa em `http://localhost:8000/docs`).

- - -

## Scripts úteis

| Script                                | Descrição                                                   |
| ------------------------------------- | ----------------------------------------------------------- |
| `uvicorn app.main:app --reload`       | Inicia a API com recarregamento automático em desenvolvimento. |
| `docker compose up -d`                | Inicia os containers da API, cliente e streaming.           |
| `docker compose logs -f yolo-api`     | Exibe os logs estruturados da API em tempo real.            |
| `docker compose down`                 | Interrompe e remove os containers locais.                   |
| `dvc pull`                            | Baixa pesos do modelo e datasets a partir do remote DVC.    |
| `pytest tests/ -v`                    | Executa a suíte de testes automatizados com relatório detalhado. |
| `ruff check app/`                     | Valida regras de lint e boas práticas no código da aplicação. |
| `python scripts/validate_model.py`    | Executa o Quality Gate avaliando o limiar de mAP@0.5 do modelo. |
| `bash scripts/deploy.sh`              | Executa deploy com pull da imagem e rollback automático se falhar. |
| `python train_epi.py`                 | Inicia o fine-tuning do modelo YOLOv8 para detecção de EPIs. |
| `python stream/mjpeg_server.py`       | Inicia o servidor dedicado de streaming de vídeo MJPEG.     |

- - -

## Testes

O projeto usa Pytest para testes automatizados. A suíte cobre integridade de endpoints, funções isoladas, decodificação de imagens, pipelines de pré-processamento e o fluxo completo de inferência:

1. Verificação de status e saúde do serviço (`/health`).
2. Consulta de métricas operacionais (`/metrics`).
3. Decodificação de imagem em Base64 e validação de dimensões e canais.
4. Redimensionamento via Letterbox e correção geométrica de bboxes.
5. Inferência com imagem de referência (`zidane.jpg`), validando detecções e formato do payload.
6. Inferência em lote (`/predict/batch`) e tratamento de erros para entradas inválidas.

Para executar:

```bash
pytest tests/ -v
```

- - - 

## Documentação

O conteúdo teórico e os estudos práticos do projeto estão organizados em diretórios dedicados:

- [`apostilas/`](./apostilas/): Material didático da disciplina (Aulas 1 a 6), contendo fundamentos de visão computacional, conteinerização Docker, MLOps, CI/CD, Tailscale e arquiteturas Edge AI.
- [`preprocessing/experiments/TABELA.md`](./preprocessing/experiments/TABELA.md): Tabela consolidada com resultados e benchmarks empíricos de pré-processamento (espaço de cor BGR vs RGB, letterbox, filtros de ruído e equalização adaptativa CLAHE para baixa luminosidade).

## Roadmap

### Milestone 1: API REST & Containerização
- [x] Criação dos endpoints de inferência `/predict`, `/predict/batch` e `/health`.
- [x] Suporte a imagens via Base64 e URL externa.
- [x] Dockerfile com PyTorch CPU-only otimizado para ARM64.
- [x] Orquestração local com Docker Compose (`yolo-api`, `yolo-client`).

### Milestone 2: MLOps, CI/CD & Deploy Contínuo
- [x] Versionamento de pesos via DVC desacoplado do Git.
- [x] Pipeline automatizado no GitHub Actions com 4 jobs sequenciais.
- [x] Quality Gate de modelo bloqueando builds com `mAP@0.5 < 0.60`.
- [x] Conexão segura via Tailscale VPN para comunicação direta com o Raspberry Pi 5.
- [x] Deploy automatizado via SSH com health check e rollback autônomo.

### Milestone 3: Câmera Física, Streaming & Otimizações
- [x] Endpoints de captura em câmera física CSI/USB (`/predict/camera` e `/predict/camera/image`).
- [x] Servidor de streaming MJPEG de baixa latência (`/stream/camera` e `/stream/view`).
- [x] Módulo de pré-processamento com Letterbox e mapeamento inverso de coordenadas.
- [x] Equalização CLAHE para ambientes de baixa iluminação.
- [x] Dataset e script de treinamento para detecção de EPIs (`train_epi.py`).

### Milestone 4: Próximos Passos
- [ ] Exportação e quantização de modelos para INT8 via ONNX Runtime / NCNN.
- [ ] Exportador de métricas Prometheus e dashboards de telemetria no Grafana.
- [ ] Pipeline de retraining contínuo a partir de detecções de baixa confiança em campo.
- [ ] Suporte a múltiplos fluxos de câmera simultâneos com aceleração V4L2.
