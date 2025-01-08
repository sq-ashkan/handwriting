"""
Handwritten Character Recognition System

A deep learning-based OCR system for recognizing handwritten characters.

Author: Ashkan Sadri Ghamshi
Project: Deep Learning Character Recognition System
Course: HAWK University - Computer Science Department
Version: 1.0.0
Date: January 2025

This module is part of an academic project that implements a high-accuracy
Optical Character Recognition (OCR) system specialized in recognizing 
handwritten uppercase letters (A-Z) and digits (0-9).
"""

class Config:

    IMAGE_SIZE = (128, 32) 
    PADDING = 10
    MIN_TEXT_HEIGHT = 20
    BINARY_THRESHOLD = 128

    ROTATION_RANGE = (-5, 5)
    SCALE_RANGE = (0.9, 1.1)
    NOISE_VARIANCE = 0.01
    

    MODEL_TYPE = "crnn" 
    LEARNING_RATE = 0.0001 
    BATCH_SIZE = 64
    NUM_EPOCHS = 200