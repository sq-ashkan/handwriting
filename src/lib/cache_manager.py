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
                
            # پاک‌سازی __init__.py در مسیرهای خاص
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
    @staticmethod
    def cleanup_data_cache():
        """Delete the /data/cache directory from the project root"""
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
        """Run all cleanup operations"""
        try:
            CacheManager.cleanup_python_cache()
            CacheManager.cleanup_temp()
            CacheManager.cleanup_interrupted_downloads()
            CacheManager.cleanup_data_cache()  # Add this line
            CacheManager.reset_sys_modules()
            logging.info("Cache cleanup completed successfully")
            return True
        except Exception as e:
            logging.error(f"Cache cleanup failed: {e}")
            return False