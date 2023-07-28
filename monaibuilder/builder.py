# Copyright (c) 2023 Chernenkiy Ivan, Sechenov University

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

import os
from pathlib import Path
from typing import Any, Dict, Union

from .extended_spytula_builder import ExtendedSpytulaBuilder
from .utils.utils import get_logger


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

    def __init__(self, bundle_root: Union[Path, str] = Path(".")) -> None:
        super().__init__()
        if isinstance(bundle_root, str):
            bundle_root = Path(bundle_root)
        self.bundle_root = bundle_root
        self.configs_path = self.bundle_root / "configs"
        self.docs_path = self.bundle_root / "docs"
        self.scripts_path = self.bundle_root / "scripts"
        self.builder = ExtendedSpytulaBuilder()

    def add_config_variable(self, name: str, value: Any):
        self.builder.attribute(name, value)

    def _add_item(
        self, builder: ExtendedSpytulaBuilder, name: str, vargs: Dict[str, Any]
    ):
        builder.attribute("_target_", name)
        for k, v in vargs.items():
            builder.attribute(k, v)

    def attribute(self, name: str, value: Any):
        self.builder.attribute(name, value)

    def add_sectioned_item(self, section_name, name: str, vargs: Dict[str, Any]):
        with self.builder.node(section_name) as section_builder:
            self._add_item(section_builder, name, vargs)

    def get_section_builder(self, section_name: str):
        return self.builder.node(section_name)

    def add_train_item(self, section_name: str, name: str, vargs: Dict[str, Any]):
        """
        Adds composite section to train section::

        {
            "train": {
                "dataset": {
                    "_target_": CacheDataset
                    ...
                }
            }
        }

        builder.add_train_item("dataset", "CacheDataset", {})
        """
        with self.builder.node("train") as train_builder:
            with train_builder.node(section_name) as section_builder:
                self._add_item(section_builder, name, vargs)

    def add_validate_item(self, section_name: str, name: str, vargs: Dict[str, Any]):
        """
        Adds composite section to validate section::

        {
            "validate": {
                "dataset": {
                    "_target_": CacheDataset
                    ...
                }
            }
        }

        builder.add_validate_item("dataset", "CacheDataset", {})
        """
        with self.builder.node("validate") as validate_builder:
            with validate_builder.node(section_name) as section_builder:
                self._add_item(section_builder, name, vargs)

    def add_deterministic_transform(
        self,
        name: str,
        vargs: Dict[str, Any],
        is_train: bool = True,
        is_validate: bool = True,
    ):
        """
        Adds preprocessing transform to train and/or validate section
        """
        train_transform_index: int = 0
        if is_train:
            with self.builder.node("train") as train_builder:
                with train_builder.nodes("deterministic_transforms") as add_transform:
                    train_transform_index = len(
                        train_builder._data["deterministic_transforms"]
                    )
                    with add_transform() as transform_builder:
                        self._add_item(transform_builder, name, vargs)

        if is_train and is_validate:
            with self.builder.node("validate") as validate_builder:
                if not validate_builder.is_exist_node("deterministic_transforms"):
                    validate_builder._data["deterministic_transforms"] = []
                validate_builder.append_value(
                    "deterministic_transforms",
                    f"@train#deterministic_transforms#{train_transform_index}",
                )
        elif is_validate:
            with self.builder.node("validate") as validate_builder:
                with validate_builder.nodes(
                    "deterministic_transforms"
                ) as add_transform:
                    with add_transform() as transform_builder:
                        self._add_item(transform_builder, name, vargs)

    def add_random_transform(self, name: str, vargs: Dict[str, Any]):
        with self.builder.node("train") as train_builder:
            with train_builder.nodes("random_transforms") as add_transform:
                with add_transform() as transform_builder:
                    self._add_item(transform_builder, name, vargs)

    def add_postprocessing_transform(
        self,
        name: str,
        vargs: Dict[str, Any],
        is_train: bool = True,
        is_validate: bool = True,
    ):
        """
        Adds postprocessing transform to train and/or validate section
        """
        train_transform_index: int = 0
        if is_train:
            with self.builder.node("train") as train_builder:
                with train_builder.nodes("postprocessing_transforms") as add_transform:
                    train_transform_index = len(
                        train_builder._data["postprocessing_transforms"]
                    )
                    with add_transform() as transform_builder:
                        self._add_item(transform_builder, name, vargs)
        if is_train and is_validate:
            with self.builder.node("validate") as validate_builder:
                if not validate_builder.is_exist_node("postprocessing_transforms"):
                    validate_builder._data["postprocessing_transforms"] = []
                validate_builder.append_value(
                    "postprocessing_transforms",
                    f"@train#postprocessing_transforms#{train_transform_index}",
                )
        elif is_validate:
            with self.builder.node("validate") as validate_builder:
                with validate_builder.nodes(
                    "postprocessing_transforms"
                ) as add_transform:
                    with add_transform() as transform_builder:
                        self._add_item(transform_builder, name, vargs)

    def add_section_handler(self, section_name: str, name: str, vargs: Dict[str, Any]):
        with self.builder.node(section_name) as section_builder:
            with section_builder.nodes("handlers") as add_transform:
                with add_transform() as transform_builder:
                    self._add_item(transform_builder, name, vargs)

    def set_train_section(
        self, config: ExtendedSpytulaBuilder = ExtendedSpytulaBuilder()
    ) -> None:
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
                "additional_metrics": "@train#additional_metrics"
            }
        }
        """

        _check_keys = [
            "preprocessing",
            "dataset",
            "dataloader",
            "postprocessing",
            "handlers",
            "key_metric",
            "trainer",
            "inferer",
        ]
        with self.builder.node("train") as train_config:
            if not train_config.is_exist_node("deterministic_transforms"):
                train_config.attribute("deterministic_transforms", list())
            if not train_config.is_exist_node("random_transforms"):
                train_config.attribute("random_transforms", list())
            train_config.attribute(
                "preprocessing",
                {
                    "_target_": "Compose",
                    "transforms": "$@train#deterministic_transforms + @train#random_transforms",
                },
            )

            train_config.attribute(
                "postprocessing",
                {
                    "_target_": "Compose",
                    "transforms": "$@train#postprocessing_transforms",
                },
            )
            train_config.attribute("additional_metrics", list())

            checks = []
            for current_key in _check_keys:
                checks.append(current_key in train_config._data.keys())
            assert all(checks), (
                f"keys '{[k for v,k in zip(checks, _check_keys) if not v ]}' "
                f"do not exists in train section"
            )

    def set_validate_section(
        self, config: ExtendedSpytulaBuilder = ExtendedSpytulaBuilder()
    ) -> None:
        """
        Add validate section to configuration.
        Must contain following section::
        {
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
            "evaluator" : {
            "_target_": "SupervisedEvaluator",
            "device": "@device",
            "val_data_loader": "@validate#dataloader",
            "network": "@network",
            "inferer": "@validate#inferer",
            "postprocessing": "@validate#postprocessing",
            "key_val_metric": "@validate#key_metric",
            "additional_metrics": "@validate#additional_metrics",
            "val_handlers": "@validate#handlers"
            }
        }
        """

        _check_keys = [
            "preprocessing",
            "postprocessing",
            "dataset",
            "dataloader",
            "handlers",
            "key_metric",
            "evaluator",
            "inferer",
        ]
        with self.builder.node("validate") as validate_config:
            if not validate_config.is_exist_node("deterministic_transforms"):
                validate_config.attribute("deterministic_transforms", list())
            validate_config.attribute(
                "preprocessing",
                {
                    "_target_": "Compose",
                    "transforms": "$@validate#deterministic_transforms",
                },
            )

            validate_config.attribute(
                "postprocessing",
                {
                    "_target_": "Compose",
                    "transforms": "$@validate#postprocessing_transforms",
                },
            )
            if not validate_config.is_exist_node("additional_metrics"):
                validate_config.attribute("additional_metrics", list())

            checks = []
            for current_key in _check_keys:
                checks.append(current_key in validate_config._data.keys())
            assert all(checks), (
                f"keys '{[k for v,k in zip(checks, _check_keys) if not v ]}' "
                f"do not exists in validate section"
            )

    def generate_logging_conf(self) -> None:
        os.makedirs(self.bundle_root, exist_ok=True)
        os.makedirs(self.configs_path, exist_ok=True)

        logfile = (
            "[loggers]\n"
            "keys=root\n\n"
            "[handlers]\n"
            "keys=consoleHandler\n\n"
            "[formatters]\n"
            "keys=fullFormatter\n\n"
            "[logger_root]\n"
            "level=INFO\n"
            "handlers=consoleHandler\n\n"
            "[handler_consoleHandler]\n"
            "class=StreamHandler\n"
            "level=INFO\n"
            "formatter=fullFormatter\n"
            "args=(sys.stdout,)\n\n"
            "[formatter_fullFormatter]\n"
            "format=%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        with open(self.configs_path / "logging.conf", "w") as f:
            f.write(logfile)

    def build(self) -> None:
        self.set_train_section(ExtendedSpytulaBuilder())
        self.set_validate_section(ExtendedSpytulaBuilder())
        self.generate_logging_conf()
        # Configure the key to use underscorecase
        os.makedirs(self.bundle_root, exist_ok=True)
        os.makedirs(self.configs_path, exist_ok=True)
        os.makedirs(self.docs_path, exist_ok=True)
        os.makedirs(self.scripts_path, exist_ok=True)

        self.builder.key_format(underscore=True)
        json_output = self.builder.to_json(indent=4)
        with open(self.configs_path / "train.json", "w") as f:
            f.write(json_output)

        logger = get_logger(self.__class__.__name__)
        logger.info(json_output)
