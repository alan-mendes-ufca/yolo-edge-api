# yolo-edge-api — MLOps e CI/CD para Edge AI

Pipeline de **entrega contínua** para a API de inferência YOLOv8 + FastAPI da Aula 2, agora
operada com práticas de MLOps: código, container e pesos do modelo são versionados
separadamente, testes automatizados bloqueiam regressões, e o deploy no Raspberry Pi 5
acontece sozinho a cada `git push`, com rollback automático se algo quebrar.

## O que o projeto faz

Resolve o problema descrito na apostila — "funciona no lab, quebra no campo" — dando
rastreabilidade e automação para as três perguntas que aparecem sempre que se opera um
container de visão computacional embarcada em produção:

- **Como atualizar o modelo em campo sem interromper o serviço?** → o binário `yolov8n.pt`
  é versionado pelo DVC (hash MD5 + storage externo), não pelo Git.
- **Como saber o que está rodando em cada dispositivo?** → toda imagem publicada carrega a
  tag `sha-<commit>`, então `docker inspect` no Pi diz exatamente qual código está ativo.
- **Como reverter uma atualização ruim sem acesso físico?** → o script de deploy faz
  health-check pós-deploy e reverte sozinho para a imagem anterior se a API não responder.

## Como funciona

Um `git push` na branch `main` dispara `.github/workflows/edge-deploy.yml`, que roda quatro
jobs em sequência no GitHub Actions:

```
push → main
  │
  ├── [1] Lint & Tests        ruff + pytest (14 testes: smoke/unit/integration)
  │
  ├── [2] Build & Push ARM64  docker buildx (QEMU) → ghcr.io/<user>/yolo-edge-api/yolo-api:sha-<hash>
  │
  ├── [3] Model Quality Gate  dvc pull + valida mAP@0.5 ≥ 0.50 (bloqueia deploy se reprovar)
  │
  └── [4] Deploy → Pi 5       SSH até o Pi, `scripts/deploy.sh`: pull da imagem, sobe,
                              healthcheck em /health, rollback automático se falhar
```

Como o Raspberry Pi normalmente fica atrás do roteador de casa (IP privado, sem porta
exposta), os jobs 1, 3 e 4 entram primeiro na tailnet via **Tailscale** para alcançar o
dispositivo — sem isso, SSH e `dvc pull` a partir dos runners do GitHub (que rodam na nuvem)
sempre dão timeout.

### Estrutura do repositório

| Caminho | Função |
|---|---|
| `app/` | API FastAPI (idêntica à da Aula 2) + `log_event()`, que emite logs estruturados em JSON no endpoint `/predict` |
| `client/` | Cliente HTTP de exemplo, consome a API |
| `tests/test_api.py` | 14 testes: smoke (`/health`, `/metrics`), unit (`_decode_image`), integration (`/predict`, `/predict/batch`) |
| `tests/assets/zidane.jpg` | Imagem de referência para os integration tests |
| `scripts/validate_model.py` | Quality gate: roda `model.val()` e aborta o pipeline (`exit 1`) se `mAP@0.5` < limiar |
| `scripts/deploy.sh` | Deploy no Pi: `docker compose pull/up`, aguarda health check, reverte para a imagem anterior em caso de falha |
| `.github/workflows/edge-deploy.yml` | Os 4 jobs descritos acima |
| `models/yolov8n.pt.dvc` | Ponteiro DVC (hash MD5) para os pesos — o `.pt` em si nunca vai para o Git |
| `Dockerfile.api` | Instala PyTorch **CPU-only** antes do resto das deps (evita puxar o stack CUDA, que não serve para um Pi sem GPU) |
| `docker-compose.yml` | Usa `image:` apontando para o `ghcr.io/...` publicado pelo pipeline, em vez de rebuildar tudo localmente a cada deploy |

## O que já está pronto neste repositório

- Todo o código da API, testes, scripts e o workflow do GitHub Actions foram escritos
  seguindo a apostila.
- O DVC já foi inicializado neste diretório (`dvc init --subdir`, já que ele vive dentro do
  monorepo da disciplina) e `models/yolov8n.pt` já está sob controle do DVC — veja
  `models/yolov8n.pt.dvc`.
- O remote configurado em `.dvc/config` aponta para um diretório **local de demonstração**
  (`~/dvc-storage-aula3`, fora do repositório) só para provar o fluxo `dvc add` → `dvc push`
  → `dvc pull` funcionando neste computador. **Isso não é o remote de produção** — veja o
  passo 3 abaixo.

## O que você precisa fazer na prática

Esta é a parte que só você pode fazer, porque depende de contas e hardware que este
ambiente não tem acesso: seu GitHub, seu Raspberry Pi físico e sua rede.

### 1. Criar o repositório e subir o código
```bash
# Crie um repositório público vazio em github.com (ex: yolo-edge-api)
# Public é necessário para usar GitHub Actions sem limite de minutos
cd aula3/yolo-edge-api
git init   # se for tratar como repositório próprio, separado do monorepo da disciplina
git remote add origin https://github.com/<seu-usuario>/yolo-edge-api.git
git add .
git commit -m "chore: importa estrutura base do projeto YOLO + FastAPI"
git push -u origin main
```
Se o push pedir senha, ela não vai funcionar — o GitHub descontinuou autenticação por
senha em linha de comando. Gere um **Personal Access Token** (Settings → Developer settings →
Personal access tokens → Tokens classic, com escopos `repo` e `workflow`) e use-o como senha.

### 2. Instalar os pesos do modelo (`yolov8n.pt`)
O binário não está no Git — ele é reconstruído via DVC. No Raspberry Pi (ou em qualquer
clone deste repositório):
```bash
pip install "dvc[ssh]" --break-system-packages
dvc pull        # baixa yolov8n.pt a partir do remote configurado em .dvc/config
```
Antes disso funcionar em produção, reconfigure o remote para apontar para o storage real no
seu Raspberry Pi (troque o remote local de demonstração por um acessível via SSH):
```bash
RPI_USER=<seu_usuario>
RPI_HOST=<ip_do_pi>   # depois da Seção "Tailscale" abaixo, use o IP 100.x.y.z
dvc remote modify local_remote url ssh://$RPI_USER@$RPI_HOST/home/$RPI_USER/dvc-storage
mkdir -p ~/dvc-storage   # no próprio Pi, cria o diretório de storage
dvc push                 # reenvia o binário para o novo remote
git add .dvc/config
git commit -m "chore: reconfigura DVC remote para acesso via SSH no CI"
```

### 3. Dar acesso SSH ao Raspberry Pi
```bash
# No seu computador (não no Pi)
ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/rpi_deploy -N ""

# No Raspberry Pi, autorize a chave pública
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "<conteúdo de rpi_deploy.pub>" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### 4. Conectar o runner do GitHub à rede do seu Pi (Tailscale)
O Pi normalmente só tem um IP privado (`192.168.x.x`), inacessível a partir dos runners do
GitHub Actions (que rodam na nuvem). Instale o Tailscale no Pi e gere uma auth key
**Reusable + Ephemeral** em `login.tailscale.com/admin/settings/keys`:
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale ip -4     # anote o IP 100.x.y.z — é esse que vai no secret RPI_HOST
```

### 5. Configurar os Secrets no GitHub
Em `Settings → Secrets and variables → Actions` do repositório, crie:

| Secret | Valor |
|---|---|
| `RPI_HOST` | IP Tailscale do Pi (`100.x.y.z`) |
| `RPI_USER` | usuário SSH do Pi |
| `RPI_SSH_KEY` | conteúdo da chave **privada** `~/.ssh/rpi_deploy` |
| `RPI_DEPLOY_PATH` | caminho do projeto no Pi, ex. `/home/pi/yolo-edge-api` |
| `TAILSCALE_AUTHKEY` | a auth key gerada no passo 4 |

Também habilite `Settings → Actions → General → Workflow permissions → Read and write
permissions`.

### 6. Ajustar o `docker-compose.yml`
Troque `seu_usuario_em_minusculo` em `docker-compose.yml` pelo seu usuário do GitHub, **em
minúsculo** (exigência do formato de nome de imagem OCI, mesmo que seu usuário do GitHub
tenha maiúsculas):
```yaml
image: ghcr.io/<seu-usuario-minusculo>/yolo-edge-api/yolo-api:latest
```

### 7. Rodar os testes localmente antes do primeiro push
No Raspberry Pi (ou em qualquer máquina Linux com o modelo em `models/yolov8n.pt`):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages
pip install -r app/requirements.txt pytest ruff --break-system-packages

sudo mkdir -p /app && sudo ln -sfn "$(pwd)/models" /app/models   # mesmo path do container

ruff check app/ --fix
pytest tests/ -v
# Esperado: 14 passed
```

### 8. Clonar no Raspberry Pi e apontar para a imagem do registry
No Pi, clone o repositório (não a partir deste monorepo — do seu `origin` no GitHub) para o
caminho definido em `RPI_DEPLOY_PATH`, e garanta que `docker`, `docker compose` e `git`
estão instalados e que o usuário tem permissão para rodar Docker.

### 9. Disparar e acompanhar o pipeline
```bash
git push origin main
```
Acompanhe em `github.com/<seu-usuario>/yolo-edge-api` → aba **Actions**. Os quatro jobs devem
ficar verdes em sequência: Lint & Tests → Build & Push → Model Quality Gate → Deploy.

### 10. Validar no Pi e testar o rollback
```bash
docker compose ps                                   # imagem com tag sha-<commit> rodando
curl http://localhost:8000/health | python3 -m json.tool   # model_loaded: true
docker compose logs --tail=10 yolo-api | jq .        # eventos estruturados (predict_complete)
```
Para provar o rollback automático, force um health check quebrado
(`docker compose run --rm -e MODEL_NAME=modelo_inexistente.pt yolo-api &`) e rode
`DEPLOY_PATH=~/yolo-edge-api bash scripts/deploy.sh` manualmente — ele deve detectar a falha
e reverter para a imagem anterior sozinho.

## Checklist de validação (do material da disciplina)

- [ ] `pytest tests/ -v` reporta `14 passed, 0 failed`
- [ ] `dvc push` executado com sucesso; `models/yolov8n.pt.dvc` commitado
- [ ] Os 4 jobs da aba Actions concluídos com sucesso
- [ ] `docker compose ps` no Pi mostra `yolo-api` rodando com a imagem mais recente
- [ ] `curl http://localhost:8000/health` retorna `model_loaded: true`
- [ ] `docker compose logs --tail=10 yolo-api | jq .` mostra eventos JSON estruturados
