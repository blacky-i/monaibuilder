# Copyright (c) 2023 Chernenkiy Ivan

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from spytula.builder import SpytulaBuilder

DEFAULT_FMT = "%(asctime)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(fmt=DEFAULT_FMT)
handler.setFormatter(formatter)
logger.addHandler(handler)


class BundleBuilder(object):
    """
    Class for creating MONAI bundle - https://docs.monai.io/en/stable/mb_specification.html

    Minimal configuration::

    {
        "imports": [
            ...
        ],
        "bundle_root": ".",
        "ckpt_dir": "$@bundle_root + '/models'",
        "output_dir": "$@bundle_root + '/eval'",
        "dataset_dir": "$@bundle_root + '/data'",
        "images": "...",
        "labels": "$list(glob.glob(@dataset_dir + '/*/merged.nii.gz'))",
        "val_interval": 50,
        "dont_finetune": true,
        "device": "$torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')",
        "network_def": {
        ...
        },
        "network_summary":{
            ...
        },
        "network":"$@network_def.to(@device)",
        "loss": {
            ...
            },
        "optimizer":{
            ...
        },
        "train":{
            ...
        },
        "validate":{
            ...
        },
        "run": [
            ...
        ]
    }
    """

    def __init__(self, bundle_root: Path = Path(".")) -> None:
        super().__init__()
        self.bundle_root = bundle_root
        self.configs_path = self.bundle_root / "configs"
        self.docs_path = self.bundle_root / "docs"
        self.configs_path = self.bundle_root / "scripts"
        self.builder = SpytulaBuilder()

    def add_config_variable(self, name: str, value: Any):
        self.builder.attribute(name, value)

    def set_train_section(self, config: SpytulaBuilder = SpytulaBuilder()) -> None:
        """
        Add train section to configuration.
        Must contain following section::
        {
            "deterministic_transforms":[
                ...
            ],
            "random_transforms":[
                ...
            ],
            "preprocessing":{
                ...
            },
            "dataset": {
                ...
            },
            "dataloader": {
                ...
            },
            "inferer":{
                ...
            },
            "postprocessing":{
                ...
            },
            "handlers": [
                ...
            ],
            "key_metric": {
                ...
            },
            "additional_metrics": {
                ...
            },
            "trainer" : {
                "_target_": "SupervisedTrainer",
                "max_epochs": 10000,
                "device": "@device",
                "train_data_loader": "@train#dataloader",
                "network": "@network",
                "loss_function": "@loss",
                "optimizer": "@optimizer",
                "inferer": "@train#inferer",
                "postprocessing": "@train#postprocessing",
                "key_train_metric": "@train#key_metric",
                "train_handlers": "@train#handlers",
                "additional_metrics": "@train#additional_metrics",
                "amp": true
            }
        }
        """
        with self.builder.node("train") as train_config:
            train_config.attribute("deterministic_transforms", list())
            train_config.attribute("random_transforms", list())
            train_config.attribute("preprocessing", {})
            train_config.attribute("dataset", {})
            train_config.attribute("dataloader", {})
            train_config.attribute("postprocessing", {})
            train_config.attribute("handlers", list())
            train_config.attribute("key_metric", list())
            train_config.attribute("additional_metrics", list())
            train_config.attribute("trainer", {})

    def build(self) -> None:
        self.set_train_section(SpytulaBuilder())
        # Configure the key to use underscorecase
        self.builder.key_format(underscore=True)
        json_output = self.builder.to_json(indent=4)
        logger.info(json_output)
