#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-10 15:46:54
"""

import json
import os
from typing import Dict, List, Optional


class Logger:
    def __init__(self, targetFile, separator=";"):
        self.targetFile = targetFile
        self.separator = separator

    def write_headers(self, headers):
        with open(self.targetFile, "a") as fh:
            for aHeader in headers:
                fh.write(aHeader + self.separator)
            fh.write("\n")

    def write_data_log(self, dataArray):
        with open(self.targetFile, "a") as fh:
            for dataItem in dataArray:
                fh.write(str(dataItem) + self.separator)
            fh.write("\n")


def setup_optim_outputs(
    models_out: str,
    opt_config: Dict,
    experiment_name: str,
    headers: Optional[List[str]] = None,
):
    target_logs = os.path.join(models_out, experiment_name + "/logs.csv")
    target_models = os.path.join(models_out, experiment_name)
    print("target_models", target_models)
    if not os.path.isdir(target_models):
        os.makedirs(target_models)
    log = Logger(target_logs, ";")

    if headers is None:
        log.write_headers(
            [
                "epoch",
                "train_loss",
                "train_audio_loss",
                "train_video_loss",
                "train_vfal_loss",
                "train_map",
                "val_loss",
                "val_audio_loss",
                "val_video_loss",
                "val_vfal_loss",
                "val_map",
            ]
        )
    else:
        log.write_headers(headers)

    # Dump cfg to json
    dump_cfg = opt_config.copy()
    for key, value in dump_cfg.items():
        if callable(value):
            try:
                dump_cfg[key] = value.__name__
            except:
                dump_cfg[key] = str(value)
    json_cfg = os.path.join(models_out, experiment_name + "/cfg.json")
    with open(json_cfg, "w") as json_file:
        json.dump(dump_cfg, json_file)

    models_out = os.path.join(models_out, experiment_name)
    return log, models_out
