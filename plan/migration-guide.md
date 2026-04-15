# SciConsist 项目迁移方案

> 源机器: 4×RTX 4090, Python 3.10, `/root/shared-nvme/` 挂载数据盘
> 数据总量: **~281GB** (`/root/shared-nvme/sciconsist_pilot/`)

---

## 0. 数据盘结构概览

```
/root/shared-nvme/sciconsist_pilot/          # 281GB 总量
├── raw/                                      # ~216GB
│   ├── scimdr/                               # ~201GB (核心数据集)
│   │   ├── arxiv/          129GB  images+papers
│   │   ├── nature/          66GB  images+papers
│   │   ├── mqa.jsonl        3.4GB
│   │   ├── tqa.jsonl        811MB
│   │   └── vqa.jsonl        1.9GB
│   ├── benchmarks/          4.8GB  (评测基准)
│   ├── s1_index_full/       9.1GB  (S1-MMAlign index)
│   ├── musciclaims_repo/    910MB
│   ├── trainable/           ?      (trainable 子集)
│   └── spiqa/               ?
├── processed/                                # ~14GB
│   ├── scimdr_sft_train.jsonl   1.4GB  ★ 必须迁移
│   ├── scimdr_sft_val.jsonl     7.2MB  ★ 必须迁移
│   ├── grpo_train_30k.jsonl     124MB  ★ 必须迁移
│   ├── table_structured/        294MB  ★ 必须迁移
│   ├── vsr_feh_cache_arxiv/     202KB  ★ 必须迁移
│   ├── vsr_feh_cache_nature/    2.5MB  ★ 必须迁移
│   ├── s1_safe_decoded/         10GB   (可重新生成)
│   ├── s1_safe_features/        952MB  (可重新生成)
│   └── real_features_*/         ~1.5GB (可重新生成，有多个版本)
├── outputs/                     19GB   (实验 checkpoint+日志)
└── cache/                       33GB   (HuggingFace 模型缓存)
```

---

## 1. 代码迁移 (Git)

新机器上执行:

```bash
# 克隆仓库
git clone git@github.com:xxm-2025/SciConsist.git
cd SciConsist

# 如果需要未提交的修改 (当前有3个未提交文件)，在旧机器先 commit 或打 patch:
# 旧机器:
git diff > /tmp/uncommitted.patch
# 新机器:
git apply /tmp/uncommitted.patch
```

当前未提交的改动:
- `sciconsist_pilot/scripts/compare_vsr_text_vs_feh_cached.py`
- `sciconsist_pilot/scripts/train_vsr_grpo.py`
- `sciconsist_pilot/src/vsr/reward.py`

---

## 2. Python 环境搭建

```bash
# 安装 uv (推荐)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
cd SciConsist/sciconsist_pilot
uv venv --python 3.10
source .venv/bin/activate
uv sync

# 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

### 当前环境关键版本 (供参考)

| 包 | 当前版本 |
|---|---------|
| Python | 3.10.12 |
| PyTorch | 2.2.0 (CUDA) |
| Transformers | 4.57.6 |
| GPU | 4×RTX 4090 (24GB) |

> **注意**: PyTorch 版本需匹配新机器的 CUDA 版本。`pyproject.toml` 中写的是 `torch>=2.1.0`，
> 建议在新机器上根据 CUDA 版本安装对应的 PyTorch:
> ```bash
> # 例如 CUDA 12.1
> uv pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 3. 数据迁移策略

### 3.1 必须迁移的数据 (~2GB，打包带走)

这些是处理好的训练数据，重新生成比较麻烦：

```bash
# 在旧机器上打包
cd /root/shared-nvme/sciconsist_pilot

tar czf /tmp/sciconsist_must_migrate.tar.gz \
  processed/scimdr_sft_train.jsonl \
  processed/scimdr_sft_val.jsonl \
  processed/grpo_train_30k.jsonl \
  processed/grpo_train_30k.stats.json \
  processed/scimdr_sft_stats.json \
  processed/table_structured/ \
  processed/vsr_feh_cache_arxiv/ \
  processed/vsr_feh_cache_nature/

# 大小约 ~1.9GB，可以 scp/rsync 到新机器
scp /tmp/sciconsist_must_migrate.tar.gz NEW_HOST:/path/to/data/
```

### 3.2 建议迁移的数据 (~25GB)

如果带宽允许，建议也带走这些：

```bash
# 实验最佳 checkpoint (挑你需要的)
tar czf /tmp/sciconsist_checkpoints.tar.gz \
  outputs/checkpoints_fixrun/feh_best.pt \
  outputs/checkpoints_cross/feh_best.pt \
  outputs/checkpoints_same/feh_best.pt
  # ... 按需添加你认为重要的 checkpoint

# raw 中的小文件
tar czf /tmp/sciconsist_raw_small.tar.gz \
  raw/musciclaims_test.jsonl \
  raw/musciclaims_test.meta.json \
  raw/s1_mmalign_stream_sample.jsonl \
  raw/scimdr/mqa.jsonl \
  raw/scimdr/tqa.jsonl \
  raw/scimdr/vqa.jsonl \
  raw/trainable/
```

### 3.3 在新机器上重新下载的数据 (~201GB)

SciMDR 的图片数据太大了，建议在新机器上用脚本重新下载：

```bash
# SciMDR 图片 (arxiv 129GB + nature 66GB)
# 使用项目内置的下载脚本:
bash sciconsist_pilot/scripts/download_scimdr_images.sh

# S1-MMAlign 数据 (9.1GB)
python sciconsist_pilot/scripts/stream_s1_index.py

# MuSciClaims
python sciconsist_pilot/scripts/download_background.py

# Benchmarks (4.8GB)
# 如有专门脚本则用脚本，否则 rsync
```

### 3.4 不需要迁移的数据

| 目录 | 大小 | 原因 |
|------|------|------|
| `cache/` (HF 模型缓存) | 33GB | 首次运行时自动下载 |
| `processed/real_features_*` | ~1.5GB | 可由 `cache_real_feh_features.py` 重新生成 |
| `processed/s1_safe_decoded/` | 10GB | 可重新生成 |
| `processed/s1_safe_features/` | 952MB | 可重新生成 |
| `outputs/` 的大部分 | ~19GB | 旧实验日志，除非需要对比 |

---

## 4. 路径配置修改

项目中有大量硬编码的 `/root/shared-nvme/sciconsist_pilot/` 路径。需要在新机器上统一替换。

### 4.1 需要修改的路径 (核心配置)

**`pilot.yaml`** — Hydra 主配置:
```yaml
# 把 vsr_grpo.data 下的路径改为新机器路径
vsr_grpo:
  data:
    train: "/NEW_PATH/processed/scimdr_sft_train.jsonl"
    val: "/NEW_PATH/processed/scimdr_sft_val.jsonl"
    table_index: "/NEW_PATH/processed/table_structured/paper_tables.jsonl"
```

### 4.2 涉及硬编码路径的文件清单

以下文件中包含 `/root/shared-nvme` 硬编码路径，到新机器后需全局替换：

```
sciconsist_pilot/run/conf/pilot.yaml
sciconsist_pilot/scripts/train_vsr_grpo.py
sciconsist_pilot/scripts/compare_vsr_text_vs_feh_cached.py
sciconsist_pilot/scripts/parse_latex_tables.py
sciconsist_pilot/scripts/test_vsr_real.py
sciconsist_pilot/scripts/download_scimdr_images.sh
sciconsist_pilot/scripts/prepare_grpo_subset.py
sciconsist_pilot/scripts/run_meta_evaluation.py
sciconsist_pilot/scripts/train_stage1_sft.py
sciconsist_pilot/scripts/sample_claims_meta_eval.py
sciconsist_pilot/scripts/run_stage1_sft.sh
sciconsist_pilot/scripts/prepare_scimdr_sft.py
sciconsist_pilot/scripts/run_gpt4o_judge.py
sciconsist_pilot/scripts/analyze_verifiability.py
sciconsist_pilot/scripts/run_stage2_softb_baseline_next.sh
sciconsist_pilot/scripts/train_stage2_policy_grpo.py
sciconsist_pilot/scripts/resume_stage2_b_c2boost_stable.sh
sciconsist_pilot/scripts/run_stage2_b_softgrid_r3.sh
sciconsist_pilot/scripts/run_stage2_b_c2boost_compare.sh
sciconsist_pilot/scripts/run_stage2_mid_grid_stable.sh
sciconsist_pilot/scripts/auto_real_pipeline.py
sciconsist_pilot/scripts/cache_real_feh_features.py
sciconsist_pilot/scripts/wait_and_extract_safe.py
sciconsist_pilot/scripts/extract_s1_features_safe.py
sciconsist_pilot/scripts/stream_s1_index.py
sciconsist_pilot/scripts/download_trainable_data.py
sciconsist_pilot/scripts/download_background.py
```

**一键替换命令 (到新机器后运行):**

```bash
# 假设新机器数据存放在 /data/sciconsist_pilot
OLD_PATH="/root/shared-nvme/sciconsist_pilot"
NEW_PATH="/data/sciconsist_pilot"   # ← 改成你的实际路径

cd SciConsist
grep -rl "$OLD_PATH" sciconsist_pilot/ | xargs sed -i "s|$OLD_PATH|$NEW_PATH|g"
```

### 4.3 HuggingFace Mirror 设置

当前代码中硬编码了 `hf-mirror.com` 作为国内镜像。如果新机器在海外，可以移除这些设置让它直连 HuggingFace Hub：

```bash
# 如果新机器能直连 HuggingFace，注释掉或删除这些行:
# export HF_ENDPOINT="https://hf-mirror.com"
# 或者保持不变，mirror 也能用
```

如果新机器也在国内，保持 `hf-mirror.com` 即可。

---

## 5. 新机器目录结构

在新机器上创建对应的目录结构：

```bash
# 假设数据放在 /data/sciconsist_pilot
NEW_DATA=/data/sciconsist_pilot

mkdir -p $NEW_DATA/{raw,processed,outputs,cache/huggingface}
mkdir -p $NEW_DATA/raw/scimdr
mkdir -p $NEW_DATA/processed/table_structured
mkdir -p $NEW_DATA/processed/vsr_feh_cache_arxiv
mkdir -p $NEW_DATA/processed/vsr_feh_cache_nature
```

---

## 6. 迁移执行 Checklist

### Phase 1: 代码 + 环境 (10 min)
- [ ] `git clone` 代码到新机器
- [ ] 应用未提交的改动 (commit 或 patch)
- [ ] 安装 `uv`，创建 venv，`uv sync`
- [ ] 安装正确版本的 PyTorch (匹配 CUDA)
- [ ] 验证 `python -c "import torch; print(torch.cuda.is_available())"`

### Phase 2: 必须迁移的数据 (~2GB, 5 min)
- [ ] 打包 `sciconsist_must_migrate.tar.gz`
- [ ] scp/rsync 到新机器
- [ ] 解压到 `$NEW_DATA/processed/`

### Phase 3: 建议迁移的数据 (~6-8GB, 15 min)
- [ ] 迁移关键 checkpoint (feh_best.pt 等)
- [ ] 迁移 raw 小文件 (jsonl 等)

### Phase 4: 路径修正 (5 min)
- [ ] 执行全局路径替换 (sed 命令)
- [ ] 检查 `pilot.yaml` 路径是否正确
- [ ] 如需调整 HF_ENDPOINT 镜像设置

### Phase 5: 在新机器下载大数据 (数小时)
- [ ] 下载 SciMDR images (`download_scimdr_images.sh`)
- [ ] 下载 S1-MMAlign index
- [ ] HuggingFace 模型缓存会在首次运行时自动下载

### Phase 6: 验证 (10 min)
- [ ] 运行一个简单测试确认数据路径正确:
  ```bash
  cd SciConsist
  python -c "
  import json
  with open('$NEW_DATA/processed/scimdr_sft_val.jsonl') as f:
      print(json.loads(f.readline()).keys())
  print('Data OK')
  "
  ```
- [ ] 跑一轮小 batch 的 FEH 训练确认 GPU 和数据流正常:
  ```bash
  python sciconsist_pilot/scripts/train_feh.py \
    --config-path ../run/conf --config-name pilot \
    training.epochs=1 training.batch_size=32
  ```

---

## 7. 安全提醒

- `run_gpt4o_judge.py` 中有硬编码的 OpenAI API key，迁移后建议改为环境变量
- `.env` 文件不在 git 中，如果旧机器有 `.env`，需要手动复制
- 确认新机器的 SSH key 已添加到 GitHub

---

## 8. 数据优先级速查表

| 优先级 | 数据 | 大小 | 方式 |
|--------|------|------|------|
| P0 必须 | processed/{sft,grpo,table,vsr_cache} | ~2GB | scp |
| P1 建议 | raw/{*.jsonl, trainable} + checkpoints | ~8GB | scp |
| P2 可选 | benchmarks, musciclaims, s1_index | ~15GB | scp 或重下 |
| P3 重下 | scimdr/arxiv+nature (图片) | ~195GB | HF download |
| P4 跳过 | cache/, 旧 outputs/, feature 缓存 | ~65GB | 自动生成 |
