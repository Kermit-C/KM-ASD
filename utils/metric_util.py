#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Description: 
@Author: chenkeming
@Date: 2024-03-25 14:46:55
"""

import os


class MetricDumper:
    def __init__(self, target_file, separator=";"):
        self.target_file = target_file
        self.separator = separator

        if not os.path.exists(self.target_file):
            os.makedirs(os.path.dirname(self.target_file), exist_ok=True)

    def write_headers(self, headers):
        with open(self.target_file, "a") as fh:
            for aHeader in headers:
                fh.write(aHeader + self.separator)
            fh.write("\n")

    def write_metrics(self, data_array):
        with open(self.target_file, "a") as fh:
            for dataItem in data_array:
                fh.write(str(dataItem) + self.separator)
            fh.write("\n")
