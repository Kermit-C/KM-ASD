#!/usr/bin/env bash

LIST_PATH="active_speaker_detection/datasets/resources/all_video_name.txt"
DATA_DIR="active_speaker_detection/datasets/resources/videos"
FAIL_LIST_PATH="${DATA_DIR}/_download_fail_video_name.txt"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} 不存在，创建目录"
  mkdir -p ${DATA_DIR}
fi

cat ${LIST_PATH} |
  while read vid; do
    echo "从 trainval 以 mp4 下载 ${vid}"
    wget -c "https://s3.amazonaws.com/ava-dataset/trainval/${vid}.mp4" -P ${DATA_DIR}

    if [ $? -ne 0 ]; then
      echo "从 trainval 以 mkv 下载 ${vid}"
      wget -c "https://s3.amazonaws.com/ava-dataset/trainval/${vid}.mkv" -P ${DATA_DIR}
    fi

    if [ $? -ne 0 ]; then
      echo "从 trainval 以 webm 下载 ${vid}"
      wget -c "https://s3.amazonaws.com/ava-dataset/trainval/${vid}.webm" -P ${DATA_DIR}
    fi

    if [ $? -ne 0 ]; then
      echo "从 test 以 mp4 下载 ${vid}"
      wget -c "https://s3.amazonaws.com/ava-dataset/test/${vid}.mp4" -P ${DATA_DIR}
    fi

    if [ $? -ne 0 ]; then
      echo "从 test 以 mkv 下载 ${vid}"
      wget -c "https://s3.amazonaws.com/ava-dataset/test/${vid}.mkv" -P ${DATA_DIR}
    fi

    if [ $? -ne 0 ]; then
      echo "从 test 以 webm 下载 ${vid}"
      wget -c "https://s3.amazonaws.com/ava-dataset/test/${vid}.webm" -P ${DATA_DIR}
    fi

    if [ $? -ne 0 ]; then
      echo "下载失败 ${vid}"
      echo ${vid} >>${FAIL_LIST_PATH}
    fi
  done

echo "下载完成"
