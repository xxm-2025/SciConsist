#!/usr/bin/env bash
# SciMDR 图片 + nature papers 下载脚本
# 使用 hf-mirror 镜像源, 下载 arxiv/images (65.5GB), nature/images (9.9GB), nature/papers (5.4GB)
# 用法: bash sciconsist_pilot/scripts/download_scimdr_images.sh

set -euo pipefail

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=1
REPO="scimdr/SciMDR"
BASE_DIR="/root/shared-nvme/sciconsist_pilot/raw/scimdr"

echo "=== [$(date)] Starting SciMDR image downloads ==="
echo "Target: $BASE_DIR"
echo "Mirror: $HF_ENDPOINT"
echo ""

download_file() {
    local file="$1"
    local desc="$2"
    echo ">>> [$(date)] Downloading $desc: $file"
    huggingface-cli download "$REPO" "$file" \
        --repo-type dataset \
        --local-dir "$BASE_DIR" \
        --resume-download 2>&1
    echo "<<< [$(date)] Done: $file"
    echo ""
}

# arxiv images - 65.5GB, 最大最重要
download_file "arxiv/images.tar.gz" "arxiv images (65.5GB)"

# nature papers - 5.4GB
download_file "nature/papers.tar.gz" "nature papers (5.4GB)"

# nature images - 9.9GB
download_file "nature/images.tar.gz" "nature images (9.9GB)"

echo "=== [$(date)] All downloads complete ==="
echo ""

# 解压
echo "=== [$(date)] Extracting archives ==="

for archive in arxiv/images.tar.gz nature/papers.tar.gz nature/images.tar.gz; do
    full_path="$BASE_DIR/$archive"
    target_dir="$(dirname "$full_path")"
    if [ -f "$full_path" ]; then
        echo ">>> Extracting $archive -> $target_dir"
        tar -xzf "$full_path" -C "$target_dir"
        echo "<<< Done: $archive"
    else
        echo "!!! Missing: $full_path"
    fi
done

echo "=== [$(date)] All extraction complete ==="
