#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-02-09 17:24:59
"""

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

retina_face_detection = pipeline(
    Tasks.face_detection, "damo/cv_resnet50_face-detection_retinaface", device="cpu"
)
img_path = "https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/retina_face_detection.jpg"
result = retina_face_detection(img_path)
print(f"face detection output: {result}.")
