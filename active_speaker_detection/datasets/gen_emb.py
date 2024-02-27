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
    out_path,
    device,
):
    model.eval()
    os.makedirs(out_path, exist_ok=True)

    # 需要保证 dataloader 的 shuffle 为 False，这样同一 entity 的数据会在一起
    curr_entity = None
    curr_data = {}
    for idx, dl in enumerate(dataloader):
        print(
            "\t Gen emb iter {:d}/{:d}".format(idx, len(dataloader)),
            end="\r",
        )

        video_data, audio_data, _, _, _, entities, ts = dl
        video_data = video_data.to(device)
        audio_data = audio_data.to(device)

        with torch.set_grad_enabled(False):
            audio_out, video_out, _, _, _, _, _ = model(audio_data, video_data)
            audio_np = audio_out.cpu().numpy()
            video_np = video_out.cpu().numpy()

        for idx, entity in enumerate(entities):
            if entity != curr_entity:
                # 保存当前 entity 的数据
                if curr_entity is not None:
                    with open(
                        os.path.join(out_path, entity.replace(":", "_") + ".pkl"), "wb"
                    ) as f:
                        pickle.dump(curr_data, f)

                curr_entity = entity
                curr_data = {}

            timestamp = ts[idx]
            curr_data[timestamp] = [audio_np[idx], video_np[idx]]

    # 保存最后一个 entity 的数据
    if curr_entity is not None:
        with open(os.path.join(out_path, entity.replace(":", "_") + ".pkl"), "wb") as f:
            pickle.dump(curr_data, f)

    print("\t Gen emb iter {:d}/{:d}".format(len(dataloader), len(dataloader)))
    print(f"\t Done! Save to {out_path}")
