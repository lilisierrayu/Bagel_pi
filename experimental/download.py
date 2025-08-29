from huggingface_hub import snapshot_download

download_mode = 3
if download_mode == 1:
    save_dir = "models/BAGEL-7B-MoT"
    repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
    cache_dir = save_dir + "/cache"

if download_mode == 2:
    save_dir = "example_data/image_edits"
    repo_id = "sysuyy/ImgEdit"
    cache_dir = save_dir + "/cache"


if download_mode == 3:
    save_dir = "example_data/image_edits"
    repo_id = "AILab-CVC/SEED-Data-Edit-Part1-Openimages"
    cache_dir = save_dir + "/cache"


snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)

