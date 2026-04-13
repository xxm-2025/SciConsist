from pathlib import Path
import json
import logging

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

# 定义本地保存路径
BASE_DATA_DIR = Path("data")
BASE_DATA_DIR.mkdir(exist_ok=True)

# 数据集配置
# SciClaimEval 需要你根据论文提供的真实 Hugging Face repo ID 替换
# 对于 S1-MMAlign，我们选用 Yuxiang-Luo/S1-MMAlign，可替换为 ScienceOne-AI/S1-MMAlign 如果可用
dataset_configs = {
    "MuSciClaims": "StonyBrookNLP/MuSciClaims",
    "S1-MMAlign": "Yuxiang-Luo/S1-MMAlign",
    # "SciClaimEval": "AuthorName/SciClaimEval"  # 需替换为真实 ID
}


def save_preview(dataset, output_path: Path, max_samples: int = 1000):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving preview to {output_path}")
    preview_data = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break
        preview_data.append({k: item[k] for k in item.keys() if k in {"image", "caption", "text", "label", "id"}})
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(preview_data, f, ensure_ascii=False, indent=2)
    logging.info(f"Saved {len(preview_data)} preview samples.")


def download_hf_datasets():
    for name, repo_id in dataset_configs.items():
        logging.info(f"=== Downloading {name} from {repo_id} ===")
        local_dir = BASE_DATA_DIR / name
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            dataset = load_dataset(repo_id)
            save_path = local_dir / "dataset"
            logging.info(f"Saving {name} dataset to {save_path}")
            dataset.save_to_disk(save_path)
            if isinstance(dataset, dict):
                for split_name, split_data in dataset.items():
                    save_preview(split_data, local_dir / f"preview_{split_name}.json")
            else:
                save_preview(dataset, local_dir / "preview.json")
            logging.info(f"{name} downloaded and saved successfully.")
        except Exception as exc:
            logging.warning(f"Failed to download {name} from {repo_id}: {exc}")
            if name == "S1-MMAlign" and repo_id != "Yuxiang-Luo/S1-MMAlign":
                fallback_repo = "Yuxiang-Luo/S1-MMAlign"
                logging.info(f"Retrying {name} with fallback repo {fallback_repo}")
                try:
                    dataset = load_dataset(fallback_repo)
                    save_path = local_dir / "dataset"
                    dataset.save_to_disk(save_path)
                    if isinstance(dataset, dict):
                        for split_name, split_data in dataset.items():
                            save_preview(split_data, local_dir / f"preview_{split_name}.json")
                    else:
                        save_preview(dataset, local_dir / "preview.json")
                    logging.info(f"{name} downloaded from fallback repo and saved successfully.")
                    continue
                except Exception as exc2:
                    logging.warning(f"Fallback download also failed: {exc2}")
            logging.error(f"Could not download dataset {name}. Please verify the Hugging Face repo ID.")


def organize_s1_subsets():
    s1_path = BASE_DATA_DIR / "S1-MMAlign"
    if s1_path.exists():
        logging.info("Checking S1-MMAlign directory structure...")
        for child in sorted(s1_path.iterdir()):
            logging.info(f"  - {child.name}")
    else:
        logging.warning("S1-MMAlign directory does not exist. Please download first.")


if __name__ == "__main__":
    download_hf_datasets()
    organize_s1_subsets()
