#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载NLTK所需的必要资源，特别是punkt_tab分词器
"""

import nltk
import os

def download_nltk_resources():
    """下载NLTK所需的必要资源"""
    print("开始下载NLTK资源...")
    
    try:
        # 下载punkt_tab分词器资源
        print("下载punkt_tab资源...")
        nltk.download('punkt_tab')
        print("punkt_tab资源下载成功!")
        
        # 额外下载一些常用资源以确保完整功能
        print("下载punkt资源...")
        nltk.download('punkt')
        print("punkt资源下载成功!")
        
        print("所有NLTK资源下载完成!")
        return True
    except Exception as e:
        print(f"下载NLTK资源时出错: {e}")
        return False

if __name__ == "__main__":
    download_nltk_resources()
