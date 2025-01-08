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

import os
import sys
import shutil
import logging
from pathlib import Path 
import tempfile

class CacheManager:
   @staticmethod
   def cleanup_temp():
       temp_dir = Path(tempfile.gettempdir())
       try:
           for temp_file in temp_dir.glob("kaggle*"):
               if temp_file.is_file():
                   os.remove(temp_file)
               elif temp_file.is_dir():
                   shutil.rmtree(temp_file)
       except Exception as e:
           logging.warning(f"Error cleaning temp files: {e}")

   @staticmethod 
   def cleanup_interrupted_downloads():
       try:
           download_patterns = ["*.part", "*.download", "*.tmp"]
           project_root = Path(__file__).parent.parent.parent
           for pattern in download_patterns:
               for file in project_root.rglob(pattern):
                   try:
                       os.remove(file)
                       logging.info(f"Removed incomplete download: {file}")
                   except Exception as e:
                       logging.warning(f"Could not remove {file}: {e}")
       except Exception as e:
           logging.warning(f"Error cleaning downloads: {e}")

   @staticmethod
   def cleanup_python_cache():
       try:
           project_root = Path(__file__).parent.parent.parent
           
           for cache_dir in project_root.rglob("__pycache__"):
               shutil.rmtree(cache_dir, ignore_errors=True)
           
           for pyc_file in project_root.rglob("*.pyc"):
               os.remove(pyc_file)
               
           for pyo_file in project_root.rglob("*.pyo"):
               os.remove(pyo_file)
               
           init_paths = [
               project_root / "src" / "services" / "processors" / "__init__.py",
               project_root / "src" / "lib" / "__init__.py"
           ]
           
           for init_path in init_paths:
               if init_path.exists():
                   try:
                       os.remove(init_path)
                       logging.info(f"Removed {init_path}")
                   except Exception as e:
                       logging.warning(f"Could not remove {init_path}: {e}")
               
       except Exception as e:
           logging.warning(f"Error cleaning Python cache: {e}")

   @staticmethod
   def reset_sys_modules():
       try:
           to_remove = []
           for module in sys.modules:
               if 'src.' in module:
                   to_remove.append(module)
           for module in to_remove:
               del sys.modules[module]
       except Exception as e:
           logging.warning(f"Error resetting modules: {e}")

   @staticmethod
   def cleanup():
       try:
           CacheManager.cleanup_python_cache()
           CacheManager.cleanup_temp()
           CacheManager.cleanup_interrupted_downloads()
           CacheManager.reset_sys_modules()
           logging.info("Cache cleanup completed successfully")
           return True
       except Exception as e:
           logging.error(f"Cache cleanup failed: {e}")
           return False

   @staticmethod
   def cleanup_data_cache():
       try:
           project_root = Path(__file__).parent.parent.parent
           cache_dir = project_root / "data" / "cache"
           
           if cache_dir.exists() and cache_dir.is_dir():
               shutil.rmtree(cache_dir)
               logging.info(f"Deleted cache directory: {cache_dir}")
           else:
               logging.info(f"Cache directory not found: {cache_dir}")
       except Exception as e:
           logging.warning(f"Error deleting cache directory: {e}")

   @staticmethod 
   def cleanup():
       try:
           CacheManager.cleanup_python_cache()
           CacheManager.cleanup_temp() 
           CacheManager.cleanup_interrupted_downloads()
           CacheManager.cleanup_data_cache()
           CacheManager.reset_sys_modules()
           logging.info("Cache cleanup completed successfully")
           return True
       except Exception as e:
           logging.error(f"Cache cleanup failed: {e}")
           return False