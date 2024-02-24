#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-24 15:21:12
"""

import os
import pickle

import torch


def gen_embedding(
    model,
    dataloader,
    dataset,
    out_path,
    device,
):
    model.eval()
    dataset_store_cache = {
        "entity_data": dataset.store.entity_data,
        "speech_data": dataset.store.speech_data,
        "ts_to_entity": dataset.store.ts_to_entity,
        "entity_list": dataset.store.entity_list,
        "feature_list": dataset.store.feature_list,
    }
    with open(os.path.join(out_path, "dataset_store_cache.pkl"), "wb") as f:
        pickle.dump(dataset_store_cache, f)

    for idx, dl in enumerate(dataloader):
        print(
            "\t Gen emb iter {:d}/{:d}".format(idx, len(dataloader)),
            end="\r",
        )

        video_data, audio_data, _, _, _, entities, ts = dl
        video_data = video_data.to(device)
        audio_data = audio_data.to(device)

        with torch.set_grad_enabled(False):
            audio_out, video_out = model(audio_data, video_data)
            audio_np = audio_out.cpu().numpy()
            video_np = video_out.cpu().numpy()

        for idx, entity in enumerate(entities):
            dir = os.path.join(out_path, entity.replace(":", "_"))
            timestamp = ts[idx]
            os.makedirs(dir, exist_ok=True)
            with open(os.path.join(dir, f"audio_{timestamp}.pkl"), "wb") as f:
                pickle.dump(audio_np[idx], f)
            with open(os.path.join(dir, f"video_{timestamp}.pkl"), "wb") as f:
                pickle.dump(video_np[idx], f)