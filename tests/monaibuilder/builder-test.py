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

import unittest

from monaibuilder.builder import BundleBuilder


class TestBundleBuilder(unittest.TestCase):
    def test_bundle_builder(self):
        """
        Checks if build raises error, since train section not defined
        """
        builder = BundleBuilder()
        self.assertRaises(AssertionError, builder.build)

    def test_create_colon(self):
        builder = BundleBuilder(bundle_root="colon_bundle")
        builder.attribute("bundle_root", ".")
        builder.attribute("imports", ["$import glob", "$import os", "$import ignite"])
        builder.attribute("ckpt_dir", "$@bundle_root + '/models'")
        builder.attribute("output_dir", "$@bundle_root + '/eval'")
        builder.attribute("input_channels", 1)
        builder.attribute("output_channels", 3)
        builder.attribute("dataset_dir", "../../../monailabel/colon/labels/original")

        builder.attribute(
            "images", "$list(sorted(glob.glob(@dataset_dir + '/../../*.nii.gz')))"
        )
        builder.attribute(
            "labels", "$list(sorted(glob.glob(@dataset_dir + '/*.nii.gz')))"
        )
        builder.attribute("val_interval", 20)
        builder.attribute("init_lr", 1e-3)
        builder.attribute("batch_size", 1)
        builder.attribute("epochs", 5000)
        builder.attribute("pixdim", "$[1,1,2.5]")
        builder.attribute("patch_size", "$[96,96,32]")
        builder.attribute(
            "device", "$torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')"
        )
        builder.attribute("input_channels", 1)
        builder.attribute("output_channels", 3)
        builder.attribute("modelname", "model.pt")
        builder.add_sectioned_item(
            "network_def",
            "SegResNet",
            {
                "spatial_dims": 3,
                "in_channels": "@input_channels",
                "out_channels": "@output_channels",
                "init_filters": 8,
                "dropout_prob": 0.2,
            },
        )
        builder.attribute("network", "$@network_def.to(@device)")
        builder.add_sectioned_item(
            "loss", "DiceCELoss", {"to_onehot_y": True, "softmax": True}
        )
        builder.add_sectioned_item(
            "optimizer",
            "torch.optim.AdamW",
            {
                "params": "$@network.parameters()",
                "lr": "@init_lr",
                "weight_decay": 1e-05,
            },
        )
        builder.add_deterministic_transform("LoadImaged", {"keys": ["image", "label"]})
        builder.add_deterministic_transform(
            "EnsureChannelFirstd", {"keys": ["image", "label"]}
        )
        builder.add_deterministic_transform("EnsureTyped", {"keys": ["image", "label"]})
        builder.add_deterministic_transform(
            "Orientationd", {"keys": ["image", "label"], "axcodes": "RAS"}
        )
        builder.add_deterministic_transform(
            "Spacingd",
            {
                "keys": ["image", "label"],
                "pixdim": "@pixdim",
                "mode": ["bilinear", "nearest"],
            },
        )
        builder.add_deterministic_transform(
            "NormalizeIntensityd", {"keys": "image", "nonzero": True}
        )
        builder.add_deterministic_transform(
            "CropForegroundd",
            {
                "keys": ["image", "label"],
                "source_key": "image",
                "k_divisible": "@patch_size",
            },
        )
        builder.add_deterministic_transform(
            "GaussianSmoothd", {"keys": ["image"], "sigma": 0.4}
        )
        builder.add_deterministic_transform(
            "ScaleIntensityd", {"keys": "image", "minv": -1.0, "maxv": 1.0}
        )
        builder.add_deterministic_transform("EnsureTyped", {"keys": ["image", "label"]})

        builder.add_random_transform(
            "RandCropByPosNegLabeld",
            {
                "keys": ["image", "label"],
                "spatial_size": "@patch_size",
                "label_key": "label",
                "neg": 0,
                "num_samples": 1,
            },
        )
        builder.add_postprocessing_transform(
            "Activationsd", {"keys": "pred", "softmax": True}
        )
        builder.add_postprocessing_transform(
            "AsDiscreted",
            {
                "keys": ["pred", "label"],
                "argmax": [True, False],
                "to_onehot": "@output_channels",
            },
        )
        # ========= CREATE TRAIN ITEMS (dataset, dataloader, inferer, metric, handler, trainer ) =========
        builder.add_train_item(
            "dataset",
            "Dataset",
            {
                "data": "$[{'image': i, 'label': l} for i, l in zip(@images[:4], @labels[:4])]",
                "transform": "@train#preprocessing",
            },
        )
        builder.add_train_item(
            "dataloader",
            "DataLoader",
            {
                "dataset": "@train#dataset",
                "batch_size": "@batch_size",
                "shuffle": True,
                "num_workers": 4,
            },
        )
        builder.add_train_item("inferer", "SimpleInferer", {})

        with builder.get_section_builder("train") as train_builder:
            with train_builder.node("key_metric") as section_builder:
                with section_builder.node("train/accuracy") as subsection_builder:
                    builder._add_item(
                        subsection_builder,
                        "ignite.metrics.Accuracy",
                        {
                            "output_transform": "$monai.handlers.from_engine(['pred', 'label'])"
                        },
                    )

        builder.add_section_handler(
            "train",
            "ValidationHandler",
            {
                "validator": "@validate#evaluator",
                "epoch_level": True,
                "interval": "@val_interval",
            },
        )
        builder.add_section_handler(
            "train",
            "StatsHandler",
            {
                "tag_name": "train/loss",
                "output_transform": "$monai.handlers.from_engine(['loss'], first=True)",
            },
        )
        builder.add_section_handler(
            "train",
            "TensorBoardStatsHandler",
            {
                "log_dir": "@output_dir",
                "tag_name": "train/loss",
                "output_transform": "$monai.handlers.from_engine(['loss'], first=True)",
            },
        )
        builder.add_train_item(
            "trainer",
            "SupervisedTrainer",
            {
                "max_epochs": "@epochs",
                "device": "@device",
                "train_data_loader": "@train#dataloader",
                "network": "@network",
                "loss_function": "@loss",
                "optimizer": "@optimizer",
                "inferer": "@train#inferer",
                "postprocessing": "@train#postprocessing",
                "key_train_metric": "@train#key_metric",
                "train_handlers": "@train#handlers",
            },
        )

        # ========= CREATE VALIDATE ITEMS (dataset, dataloader, inferer, metric, handler, trainer ) =========

        builder.add_validate_item(
            "dataset",
            "Dataset",
            {
                "data": "$[{'image': i, 'label': l} for i, l in zip(@images[:4], @labels[:4])]",
                "transform": "@validate#preprocessing",
            },
        )
        builder.add_validate_item(
            "dataloader",
            "DataLoader",
            {
                "dataset": "@validate#dataset",
                "batch_size": "@batch_size",
                "shuffle": True,
                "num_workers": 4,
            },
        )
        builder.add_validate_item(
            "inferer",
            "SlidingWindowInferer",
            {"roi_size": "@patch_size", "sw_batch_size": 1, "overlap": 0.25},
        )

        with builder.get_section_builder("validate") as validate_builder:
            with validate_builder.node("key_metric") as section_builder:
                with section_builder.node("validate/mean_dice") as subsection_builder:
                    builder._add_item(
                        subsection_builder,
                        "MeanDice",
                        {
                            "include_background": False,
                            "output_transform": "$monai.handlers.from_engine(['pred', 'label'])",
                        },
                    )
        with builder.get_section_builder("validate") as validate_builder:
            with validate_builder.node("additional_metrics") as section_builder:
                with section_builder.node("validate/accuracy") as subsection_builder:
                    builder._add_item(
                        subsection_builder,
                        "ignite.metrics.Accuracy",
                        {
                            "output_transform": "$monai.handlers.from_engine(['pred', 'label'])"
                        },
                    )

        builder.add_section_handler(
            "validate", "StatsHandler", {"iteration_log": False}
        )
        builder.add_section_handler(
            "validate",
            "TensorBoardStatsHandler",
            {"log_dir": "@output_dir", "iteration_log": False},
        )
        builder.add_section_handler(
            "validate",
            "CheckpointSaver",
            {
                "save_dir": "@ckpt_dir",
                "save_dict": {"model": "@network"},
                "save_key_metric": True,
                "key_metric_filename": "@modelname",
            },
        )
        builder.add_validate_item(
            "evaluator",
            "SupervisedEvaluator",
            {
                "device": "@device",
                "val_data_loader": "@validate#dataloader",
                "network": "@network",
                "inferer": "@validate#inferer",
                "postprocessing": "@validate#postprocessing",
                "key_val_metric": "@validate#key_metric",
                "additional_metrics": "@validate#additional_metrics",
                "val_handlers": "@validate#handlers",
            },
        )
        # ========= CREATE RUN COMMANDS =========

        builder.attribute(
            "initialize",
            [
                "$monai.utils.set_determinism(seed=123)",
                "$setattr(torch.backends.cudnn, 'enabled', False)",
            ],
        )
        builder.attribute("evaluate", ["$@validate#evaluator.run()"])

        builder.attribute("run", ["$@train#trainer.run()"])
        builder.build()


if __name__ == "__main__":
    unittest.main()
