#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2012-02-05 23:05:15
"""


datasets = {
    # 输入文件
    "csv_train_full": "/hdd1/ckm/datasets/ava/annotations_entity/ava_activespeaker_train_augmented.csv",
    "csv_val_full": "/hdd1/ckm/datasets/ava/annotations_entity/ava_activespeaker_val_augmented.csv",
    "csv_test_full": "/hdd1/ckm/datasets/ava/annotations_entity/ava_activespeaker_test_augmented.csv",
    # 数据配置
    "audio_train_dir": "/ssd1/ckm2/instance_wavs_time_train/",
    "video_train_dir": "/ssd1/ckm2/instance_crops_time_train/",
    "audio_val_dir": "/ssd1/ckm2/instance_wavs_time_val/",
    "video_val_dir": "/ssd1/ckm2/instance_crops_time_val/",
    # 数据存储
    "data_store_train_cache": "active_speaker_detection/datasets/resources/data_store_cache/dataset_train_store_cache.pkl",
    "data_store_val_cache": "active_speaker_detection/datasets/resources/data_store_cache/dataset_val_store_cache.pkl",
    # 模型保存目录
    "models_out": "./active_speaker_detection/results",
}

inference_params = {
    "encoder_type": "R3D18",
    "graph_type": "GraphAllEdgeNet",
    "encoder_enable_vf": True,
    "graph_enable_spatial": True,
    # 每刻计算特征的帧数
    "frmc": 13,
    # 上下文大小，每刻的实体数
    "ctx": 3,
    # 图的时间上下文步数，即 clip 数
    "nclp": 7,
    # 图的时间上下文步长，即 clip 之间的间隔，单位为帧
    "strd": 3,
    # 图像大小，将把人脸 crop resize 到这个大小的正方形
    "size": 112,
    "audio_sample_rate": 16000,
    # 图的边是否双向连接
    "is_edge_double": False,
    # 不同实体在不同时刻之间的连接
    "is_edge_across_entity": False,
}


train_params = [
    {
        "name": "R3D18",
        # 网络架构
        "encoder_type": "R3D18",
        "graph_type": "GraphAllEdgeNet",
        "encoder_enable_vf": True,
        "graph_enable_spatial": True,
        "encoder_train_weights": None,  # "./active_speaker_detection/results/encoder_R3D18_vf1_clip13/32.pth",
        "encoder_feature_dir": "/hdd1/ckm2/feature/R3D18",
        # 预训练权重
        "encoder_enable_grad": True,  # 开的情况：整个网络端到端、encoder 端到端后二阶段训练图；关的情况：只训练音脸后二阶段训练图
        "encoder_audio_pretrain_weights": "/hdd1/ckm/pretrain-model/2D-ResNet/results/resnet18-5c106cde.pth",
        "encoder_video_pretrain_weights": "/hdd1/ckm/pretrain-model/3D-ResNets-PyTorch/results/r3d18_K_200ep.pth",
        "graph_spatial_pretrain_weights": "/hdd1/ckm/pretrain-model/mobilenet_v2/results/mobilenet_v2-b0353104.pth",
        # encoder 优化配置
        "encoder_batch_size": 128,
        "encoder_learning_rate": 3e-4,
        "encoder_epochs": 70,
        "encoder_milestones": list(range(0, 70, 30)),
        "encoder_gamma": 0.1,
        # graph 优化配置
        "graph_batch_size": 64,
        "graph_learning_rate": 3e-6,
        "graph_epochs": 10,
        "graph_milestones": [5],
        "graph_gamma": 0.1,
        # 端到端优化配置
        "batch_size": 8,
        "accumulation_steps": 2,
        "learning_rate": 1e-3,
        "epochs": 15,
        "milestones": [6, 9],
        "gamma": 0.1,
        # 数据集加载器
        "encoder_threads": 12,
        "graph_threads": 0,  # 不为 0 的话，工作进程里就没有 cache 了
        "threads": 8,
    },
    {
        "name": "R3D50",
        # 网络架构
        "encoder_type": "R3D50",
        "graph_type": "GraphGatEdgeNet",
        "encoder_enable_vf": True,
        "graph_enable_spatial": True,
        "encoder_train_weights": None,  # "active_speaker_detection/results/encoder_R3D50_vf1_clip13/17.pth",
        "encoder_feature_dir": "active_speaker_detection/datasets/resources/features/R3D50",
        # 预训练权重
        "encoder_enable_grad": True,  # 开的情况：整个网络端到端、encoder 端到端后二阶段训练图；关的情况：只训练音脸后二阶段训练图
        "encoder_audio_pretrain_weights": "/hdd1/ckm/pretrain-model/2D-ResNet/results/resnet18-5c106cde.pth",
        "encoder_video_pretrain_weights": "/hdd1/ckm/pretrain-model/3D-ResNets-PyTorch/results/r3d50_K_200ep.pth",
        "graph_spatial_pretrain_weights": "/hdd1/ckm/pretrain-model/mobilenet_v2/results/mobilenet_v2-b0353104.pth",
        # encoder 优化配置
        "encoder_batch_size": 128,
        "encoder_learning_rate": 3e-4,
        "encoder_epochs": 70,
        "encoder_milestones": list(range(0, 70, 30)),
        "encoder_gamma": 0.1,
        # graph 优化配置
        "graph_batch_size": 64,
        "graph_learning_rate": 3e-6,
        "graph_epochs": 10,
        "graph_milestones": [5],
        "graph_gamma": 0.1,
        # 端到端优化配置
        "batch_size": 6,
        "accumulation_steps": 3,
        "learning_rate": 1e-3,
        "epochs": 15,
        "milestones": [6, 9],
        "gamma": 0.1,
        # 数据集加载器
        "encoder_threads": 12,
        "graph_threads": 0,  # 不为 0 的话，工作进程里就没有 cache 了
        "threads": 8,
    },
    {
        "name": "RES18_TSM",
        # 网络架构
        "encoder_type": "RES18_TSM",
        "graph_type": "GraphAllEdgeNet",
        "encoder_enable_vf": True,
        "graph_enable_spatial": True,
        "encoder_train_weights": None,
        "encoder_feature_dir": "/hdd1/ckm2/features/RES18_TSM",
        # 预训练权重
        "encoder_enable_grad": True,  # 开的情况：整个网络端到端、encoder 端到端后二阶段训练图；关的情况：只训练音脸后二阶段训练图
        "encoder_audio_pretrain_weights": "/hdd1/ckm/pretrain-model/2D-ResNet/results/resnet18-5c106cde.pth",
        "encoder_video_pretrain_weights": "/hdd1/ckm/pretrain-model/2D-ResNet/results/resnet18-5c106cde.pth",
        "graph_spatial_pretrain_weights": "/hdd1/ckm/pretrain-model/mobilenet_v2/results/mobilenet_v2-b0353104.pth",
        # encoder 优化配置
        "encoder_batch_size": 128,
        "encoder_learning_rate": 3e-4,
        "encoder_epochs": 70,
        "encoder_milestones": list(range(0, 70, 30)),
        "encoder_gamma": 0.1,
        # graph 优化配置
        "graph_batch_size": 64,
        "graph_learning_rate": 3e-6,
        "graph_epochs": 10,
        "graph_milestones": [5],
        "graph_gamma": 0.1,
        # 端到端优化配置
        "batch_size": 16,
        "accumulation_steps": 1,
        "learning_rate": 1e-3,
        "epochs": 15,
        "milestones": [6, 9],
        "gamma": 0.1,
        # 数据集加载器
        "encoder_threads": 12,
        "graph_threads": 0,  # 不为 0 的话，工作进程里就没有 cache 了
        "threads": 4,
    },
    {
        "name": "RES50_TSM",
        # 网络架构
        "encoder_type": "RES50_TSM",
        "graph_type": "GraphAllEdgeNet",
        "encoder_enable_vf": False,
        "graph_enable_spatial": True,
        "encoder_train_weights": None,
        "encoder_feature_dir": "active_speaker_detection/datasets/resources/features/RES50_TSM",
        # 预训练权重
        "encoder_enable_grad": True,  # 开的情况：整个网络端到端、encoder 端到端后二阶段训练图；关的情况：只训练音脸后二阶段训练图
        "encoder_audio_pretrain_weights": "/hdd1/ckm/pretrain-model/2D-ResNet/results/resnet50-19c8e357.pth",
        "encoder_video_pretrain_weights": "/hdd1/ckm/pretrain-model/2D-ResNet/results/resnet50-19c8e357.pth",
        "graph_spatial_pretrain_weights": "/hdd1/ckm/pretrain-model/mobilenet_v2/results/mobilenet_v2-b0353104.pth",
        # encoder 优化配置
        "encoder_batch_size": 128,
        "encoder_learning_rate": 3e-4,
        "encoder_epochs": 70,
        "encoder_milestones": list(range(0, 70, 30)),
        "encoder_gamma": 0.1,
        # graph 优化配置
        "graph_batch_size": 64,
        "graph_learning_rate": 3e-6,
        "graph_epochs": 10,
        "graph_milestones": [5],
        "graph_gamma": 0.1,
        # 端到端优化配置
        "batch_size": 16,
        "accumulation_steps": 1,
        "learning_rate": 1e-3,
        "epochs": 15,
        "milestones": [6, 9],
        "gamma": 0.1,
        # 数据集加载器
        "encoder_threads": 12,
        "graph_threads": 0,  # 不为 0 的话，工作进程里就没有 cache 了
        "threads": 4,
    },
    {
        "name": "RX3D50",
        # 网络架构
        "encoder_type": "RX3D50",
        "graph_type": "GraphAllEdgeNet",
        "encoder_enable_vf": True,
        "graph_enable_spatial": True,
        "encoder_train_weights": None,
        "encoder_feature_dir": "active_speaker_detection/datasets/resources/features/RX3D50",
        # 预训练权重
        "encoder_enable_grad": True,  # 开的情况：整个网络端到端、encoder 端到端后二阶段训练图；关的情况：只训练音脸后二阶段训练图
        "encoder_audio_pretrain_weights": "/hdd1/ckm/pretrain-model/2D-ResNet/results/resnet18-5c106cde.pth",
        "encoder_video_pretrain_weights": None,
        "graph_spatial_pretrain_weights": "/hdd1/ckm/pretrain-model/mobilenet_v2/results/mobilenet_v2-b0353104.pth",
        # encoder 优化配置
        "encoder_batch_size": 128,
        "encoder_learning_rate": 3e-4,
        "encoder_epochs": 70,
        "encoder_milestones": list(range(0, 70, 30)),
        "encoder_gamma": 0.1,
        # graph 优化配置
        "graph_batch_size": 64,
        "graph_learning_rate": 3e-6,
        "graph_epochs": 10,
        "graph_milestones": [5],
        "graph_gamma": 0.1,
        # 端到端优化配置
        "batch_size": 16,
        "accumulation_steps": 1,
        "learning_rate": 1e-3,
        "epochs": 15,
        "milestones": [6, 9],
        "gamma": 0.1,
        # 数据集加载器
        "encoder_threads": 12,
        "graph_threads": 0,  # 不为 0 的话，工作进程里就没有 cache 了
        "threads": 4,
    },
    {
        "name": "RX3D101",
        # 网络架构
        "encoder_type": "RX3D101",
        "graph_type": "GraphGatEdgeNet",
        "encoder_enable_vf": True,
        "graph_enable_spatial": True,
        "encoder_train_weights": None,
        "encoder_feature_dir": "active_speaker_detection/datasets/resources/features/RX3D101",
        # 预训练权重
        "encoder_enable_grad": True,  # 开的情况：整个网络端到端、encoder 端到端后二阶段训练图；关的情况：只训练音脸后二阶段训练图
        "encoder_audio_pretrain_weights": "/hdd1/ckm/pretrain-model/2D-ResNet/results/resnet18-5c106cde.pth",
        "encoder_video_pretrain_weights": "/hdd1/ckm/pretrain-model/Efficient-3DCNNs/results/kinetics_resnext_101_RGB_16_best.pth",
        "graph_spatial_pretrain_weights": "/hdd1/ckm/pretrain-model/mobilenet_v2/results/mobilenet_v2-b0353104.pth",
        # encoder 优化配置
        "encoder_batch_size": 128,
        "encoder_learning_rate": 3e-4,
        "encoder_epochs": 70,
        "encoder_milestones": list(range(0, 70, 30)),
        "encoder_gamma": 0.1,
        # graph 优化配置
        "graph_batch_size": 64,
        "graph_learning_rate": 3e-6,
        "graph_epochs": 10,
        "graph_milestones": [5],
        "graph_gamma": 0.1,
        # 端到端优化配置
        "batch_size": 8,
        "accumulation_steps": 4,
        "learning_rate": 1e-3,
        "epochs": 15,
        "milestones": [6, 9],
        "gamma": 0.1,
        # 数据集加载器
        "encoder_threads": 12,
        "graph_threads": 0,  # 不为 0 的话，工作进程里就没有 cache 了
        "threads": 4,
    },
    {
        "name": "RX3D152",
        # 网络架构
        "encoder_type": "RX3D152",
        "graph_type": "GraphAllEdgeNet",
        "encoder_enable_vf": True,
        "graph_enable_spatial": True,
        "encoder_train_weights": None,
        "encoder_feature_dir": "active_speaker_detection/datasets/resources/features/RX3D152",
        # 预训练权重
        "encoder_enable_grad": True,  # 开的情况：整个网络端到端、encoder 端到端后二阶段训练图；关的情况：只训练音脸后二阶段训练图
        "encoder_audio_pretrain_weights": "/hdd1/ckm/pretrain-model/2D-ResNet/results/resnet18-5c106cde.pth",
        "encoder_video_pretrain_weights": None,
        "graph_spatial_pretrain_weights": "/hdd1/ckm/pretrain-model/mobilenet_v2/results/mobilenet_v2-b0353104.pth",
        # encoder 优化配置
        "encoder_batch_size": 128,
        "encoder_learning_rate": 3e-4,
        "encoder_epochs": 70,
        "encoder_milestones": list(range(0, 70, 30)),
        "encoder_gamma": 0.1,
        # graph 优化配置
        "graph_batch_size": 64,
        "graph_learning_rate": 3e-6,
        "graph_epochs": 10,
        "graph_milestones": [5],
        "graph_gamma": 0.1,
        # 端到端优化配置
        "batch_size": 16,
        "accumulation_steps": 1,
        "learning_rate": 1e-3,
        "epochs": 15,
        "milestones": [6, 9],
        "gamma": 0.1,
        # 数据集加载器
        "encoder_threads": 12,
        "graph_threads": 0,  # 不为 0 的话，工作进程里就没有 cache 了
        "threads": 4,
    },
    {
        "name": "MB3DV1",
        # 网络架构
        "encoder_type": "MB3DV1",
        "graph_type": "GraphAllEdgeNet",
        "encoder_enable_vf": True,
        "graph_enable_spatial": True,
        "encoder_train_weights": None,
        "encoder_feature_dir": "active_speaker_detection/datasets/resources/features/MB3DV1",
        # 预训练权重
        "encoder_enable_grad": True,  # 开的情况：整个网络端到端、encoder 端到端后二阶段训练图；关的情况：只训练音脸后二阶段训练图
        "encoder_audio_pretrain_weights": "/hdd1/ckm/pretrain-model/2D-ResNet/results/resnet18-5c106cde.pth",
        "encoder_video_pretrain_weights": "/hdd1/ckm/pretrain-model/Efficient-3DCNNs/results/kinetics_mobilenet_2.0x_RGB_16_best.pth",
        "graph_spatial_pretrain_weights": "/hdd1/ckm/pretrain-model/mobilenet_v2/results/mobilenet_v2-b0353104.pth",
        # encoder 优化配置
        "encoder_batch_size": 128,
        "encoder_learning_rate": 3e-4,
        "encoder_epochs": 70,
        "encoder_milestones": list(range(0, 70, 30)),
        "encoder_gamma": 0.1,
        # graph 优化配置
        "graph_batch_size": 64,
        "graph_learning_rate": 3e-6,
        "graph_epochs": 10,
        "graph_milestones": [5],
        "graph_gamma": 0.1,
        # 端到端优化配置
        "batch_size": 16,
        "accumulation_steps": 1,
        "learning_rate": 1e-3,
        "epochs": 15,
        "milestones": [6, 9],
        "gamma": 0.1,
        # 数据集加载器
        "encoder_threads": 12,
        "graph_threads": 0,  # 不为 0 的话，工作进程里就没有 cache 了
        "threads": 4,
    },
    {
        "name": "MB3DV2",
        # 网络架构
        "encoder_type": "MB3DV2",
        "graph_type": "GraphAllEdgeNet",
        "encoder_enable_vf": True,
        "graph_enable_spatial": True,
        "encoder_train_weights": None,
        "encoder_feature_dir": "active_speaker_detection/datasets/resources/features/MB3DV2",
        # 预训练权重
        "encoder_enable_grad": True,  # 开的情况：整个网络端到端、encoder 端到端后二阶段训练图；关的情况：只训练音脸后二阶段训练图
        "encoder_audio_pretrain_weights": "/hdd1/ckm/pretrain-model/2D-ResNet/results/resnet18-5c106cde.pth",
        "encoder_video_pretrain_weights": "/hdd1/ckm/pretrain-model/Efficient-3DCNNs/results/kinetics_mobilenetv2_1.0x_RGB_16_best.pth",
        "graph_spatial_pretrain_weights": "/hdd1/ckm/pretrain-model/mobilenet_v2/results/mobilenet_v2-b0353104.pth",
        # encoder 优化配置
        "encoder_batch_size": 128,
        "encoder_learning_rate": 3e-4,
        "encoder_epochs": 70,
        "encoder_milestones": list(range(0, 70, 30)),
        "encoder_gamma": 0.1,
        # graph 优化配置
        "graph_batch_size": 64,
        "graph_learning_rate": 3e-6,
        "graph_epochs": 10,
        "graph_milestones": [5],
        "graph_gamma": 0.1,
        # 端到端优化配置
        "batch_size": 16,
        "accumulation_steps": 1,
        "learning_rate": 1e-3,
        "epochs": 15,
        "milestones": [6, 9],
        "gamma": 0.1,
        # 数据集加载器
        "encoder_threads": 12,
        "graph_threads": 0,  # 不为 0 的话，工作进程里就没有 cache 了
        "threads": 4,
    },
    {
        "name": "LIGHT",
        # 网络架构
        "encoder_type": "LIGHT",
        "graph_type": "GraphAllEdgeNet",
        "encoder_enable_vf": True,
        "graph_enable_spatial": True,
        "encoder_train_weights": "active_speaker_detection/results/encoder_LIGHT_clip13.0_ctx3_len7_str3/26.pth",
        "encoder_feature_dir": "/hdd1/ckm2/feature/LIGHT",
        # 预训练权重
        "encoder_enable_grad": True,  # 开的情况：整个网络端到端、encoder 端到端后二阶段训练图；关的情况：只训练音脸后二阶段训练图
        "encoder_audio_pretrain_weights": None,
        "encoder_video_pretrain_weights": None,
        "graph_spatial_pretrain_weights": "/hdd1/ckm/pretrain-model/mobilenet_v2/results/mobilenet_v2-b0353104.pth",
        # encoder 优化配置
        "encoder_batch_size": 128,
        "encoder_learning_rate": 1e-3,
        "encoder_epochs": 30,
        "encoder_milestones": list(range(30)),
        "encoder_gamma": 0.95,
        # graph 优化配置
        "graph_batch_size": 64,
        "graph_learning_rate": 3e-6,
        "graph_epochs": 10,
        "graph_milestones": [5],
        "graph_gamma": 0.1,
        # 端到端优化配置
        "batch_size": 16,
        "accumulation_steps": 1,
        "learning_rate": 1e-3,
        "epochs": 15,
        "milestones": [6, 9],
        "gamma": 0.1,
        # 数据集加载器
        "encoder_threads": 12,
        "graph_threads": 0,  # 不为 0 的话，工作进程里就没有 cache 了
        "threads": 4,
    },
]
