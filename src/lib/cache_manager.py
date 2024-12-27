# src/lib/cache_manager.py
import os
import sys
import shutil
import logging
from pathlib import Path
import tempfile

class CacheManager:
    @staticmethod
    def cleanup_temp():
        """پاک‌سازی فایل‌های موقت"""
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
        """پاک‌سازی دانلودهای ناتمام"""
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
        """پاک‌سازی کش پایتون"""
        try:
            project_root = Path(__file__).parent.parent.parent
            
            # پاک‌سازی __pycache__
            for cache_dir in project_root.rglob("__pycache__"):
                shutil.rmtree(cache_dir, ignore_errors=True)
            
            # پاک‌سازی .pyc
            for pyc_file in project_root.rglob("*.pyc"):
                os.remove(pyc_file)
                
            # پاک‌سازی .pyo
            for pyo_file in project_root.rglob("*.pyo"):
                os.remove(pyo_file)
                
        except Exception as e:
            logging.warning(f"Error cleaning Python cache: {e}")

    @staticmethod
    def reset_sys_modules():
        """بازنشانی ماژول‌های پایتون"""
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
        """اجرای تمام عملیات پاک‌سازی"""
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