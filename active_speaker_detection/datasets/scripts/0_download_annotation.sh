#!/usr/bin/env bash

TARGET_DIR=active_speaker_detection/datasets/resources/annotations

if [ ! -d $TARGET_DIR ]; then
    mkdir -p $TARGET_DIR
fi

wget -O $TARGET_DIR/ava_activespeaker_train_augmented.csv https://filedn.com/l0kNCNuXuEq70c3iUHsXxJ7/active-speakers-context/ava_activespeaker_train_augmented.csv
wget -O $TARGET_DIR/ava_activespeaker_val_augmented.csv https://filedn.com/l0kNCNuXuEq70c3iUHsXxJ7/active-speakers-context/ava_activespeaker_val_augmented.csv
wget -O $TARGET_DIR/ava_activespeaker_test_augmented.csv https://filedn.com/l0kNCNuXuEq70c3iUHsXxJ7/active-speakers-context/ava_activespeaker_test_augmented.csv
