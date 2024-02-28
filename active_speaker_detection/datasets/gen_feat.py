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


def gen_feature(
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
            "\t Gen feat iter {:d}/{:d}".format(idx, len(dataloader)),
            end="\r",
        )

        video_data, audio_data, _, _, _, entities, ts = dl
        video_data = video_data.to(device)
        audio_data = audio_data.to(device)

        with torch.set_grad_enabled(False):
            audio_out, video_out, _, _, _, vf_a_emb, vf_v_emb = model(
                audio_data, video_data
            )
            audio_np = audio_out.cpu().numpy()
            video_np = video_out.cpu().numpy()
            vf_a_emb_np = vf_a_emb.cpu().numpy() if vf_a_emb is not None else None
            vf_v_emb_np = vf_v_emb.cpu().numpy() if vf_v_emb is not None else None

        for idx, entity in enumerate(entities):
            if entity != curr_entity:
                # 保存当前 entity 的数据
                if curr_entity is not None:
                    with open(
                        os.path.join(out_path, curr_entity.replace(":", "_") + ".pkl"),
                        "wb",
                    ) as f:
                        pickle.dump(curr_data, f)

                curr_entity = entity
                curr_data = {}

            timestamp = ts[idx]
            curr_data[timestamp] = [
                audio_np[idx],
                video_np[idx],
                vf_a_emb_np[idx] if vf_a_emb_np is not None else None,
                vf_v_emb_np[idx] if vf_v_emb_np is not None else None,
            ]

    # 保存最后一个 entity 的数据
    if curr_entity is not None:
        with open(
            os.path.join(out_path, curr_entity.replace(":", "_") + ".pkl"), "wb"
        ) as f:
            pickle.dump(curr_data, f)

    print("\t Gen feat iter {:d}/{:d}".format(len(dataloader), len(dataloader)))
    print(f"\t Done! Save to {out_path}")
