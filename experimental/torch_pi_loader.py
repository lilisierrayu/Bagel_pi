"""
This script demonstrates how to use PyTorch's DataLoader with a our datasets.

# If you are using this in a separate repo, make sure to do the following steps:
#  export MONOPI_REPO=/home/liliyu/workspace/monopi
#  pip install --requirement $MONOPI_REPO/monopi/model/data/requirements.txt
#  pip install -e $MONOPI_REPO
"""

# gazelle:ignore torch
import torch
from monopi.model.configs import config as _config
from monopi.model.configs import registered_configs as register_cfg
from monopi.model.data import dataloader
from monopi.experimental.dibyaghosh import utils as experimental_utils
from monopi.lib.py.image import image as lib_image
import getpass

class Dataset(torch.utils.data.Dataset):
    def __init__(self, lazy_dataset):
        self.lazy_dataset = lazy_dataset

    def __len__(self):
        return len(self.lazy_dataset)

    def __getitem__(self, idx):
        return self.lazy_dataset[idx]


def create_dataset(
    config: _config.TrainConfig, *, split: str = "train", num_epochs: int = 1
):
    """Creates a PyTorch dataset from a config name."""
    ## FIX this with future images
    
    # create an dataset
    # experimental_utils.cache_specs(
    #     config.data.task_mixture_config, f"/home/{getpass.getuser()}/cached_specs"
    # )

    task_mixture = dataloader.create_task_mixture(config.data)
    mixture = task_mixture.mixtures[split]
    return Dataset(mixture.get_dataset(num_epochs=num_epochs, shuffle=True))


def main(config: _config.TrainConfig):
    dataset = create_dataset(config)
    breakpoint()
    
    v = dataset[0]['future_image']['image_0']
    vv = lib_image.decompress_image_if_needed(v)

    # breakpoint()
    num_batches = 2
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=4)
    for i, batch in enumerate(dataloader):
        print(i, batch)
        if i >= num_batches:
            break


if __name__ == "__main__":
    from monopi.model.configs import registered_configs as register_cfg
    # register_cfg.cli_with_selectable_config(main)
    config = register_cfg.get_config("robot_caption_config")
    # config = register_cfg.get_config("mar18_arx_single_baseline")

    main(config)
    
