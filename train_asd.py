#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 训练 ASD 模型
@Author: chenkeming
@Date: 2024-02-16 20:48:53
"""

from active_speaker_detection import asd_train

"""
```Bash
python train_asd.py --name=R3D18 --stage=encoder --frmc=13 --ctx=2 --nclp=7 --strd=3 --size=160
python train_asd.py --name=R3D18 --stage=graph --frmc=13 --ctx=2 --nclp=7 --strd=3 --size=160
python train_asd.py --name=R3D18 --stage=end2end --frmc=13 --ctx=2 --nclp=7 --strd=3 --size=160
```
"""

if __name__ == "__main__":
    asd_train.train()
