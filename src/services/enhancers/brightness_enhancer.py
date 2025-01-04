from pathlib import Path
from src.services.enhancers.base_enhancer import BaseEnhancer

class BrightnessEnhancer(BaseEnhancer):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.dataset_name = Path(dataset_path).parent.name
        self.configs = {
            "EH": {
                "target_mean": 40.0,      # Increase from 13.29
                "target_std": 60.0,
                "clip_limit": 2.0
            },
            "MNIST": {
                "target_mean": 45.0,      # More balanced
                "target_std": 70.0,
                "clip_limit": 2.5
            },
            "AZ": {
                "target_mean": 50.0,      # Keep moderate brightness
                "target_std": 80.0,
                "clip_limit": 3.0
            },
            "Chars74K": {
                "target_mean": 60.0,      # Reduce from 87.45
                "target_std": 90.0,
                "clip_limit": 2.0
            }
        }
    
    def _get_config(self) -> dict:
        """Return configuration for the current dataset"""
        default_config = {
            "target_mean": 50.0,
            "target_std": 70.0,
            "clip_limit": 2.0
        }
        return self.configs.get(self.dataset_name, default_config)