from torchvision import datasets
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import logging
import shutil
from src.lib.constants import MNIST_DIR

class MNISTDatasetDownloader:
    def __init__(self):
        self.data_dir = MNIST_DIR
        self.image_dir = MNIST_DIR / 'images'
        self.doc_path = MNIST_DIR / 'documentation.txt'
        self.current_id = 1
        self.logger = logging.getLogger("MNIST")
        
    def _create_dirs(self):
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def _generate_iam_id(self):
        writer_id = f"m{self.current_id:02d}"
        doc_id = f"{self.current_id:03d}u"
        image_id = f"{writer_id}-{doc_id}-00-00"
        self.current_id += 1
        return image_id

    def _process_image(self, image):
        image_array = np.array(image)
        pil_image = Image.fromarray(image_array)
        target_size = (113, 113)
        return pil_image.resize(target_size, Image.Resampling.LANCZOS)

    def _save_images(self, dataset, is_train=True):
        records = []
        stage = "Training" if is_train else "Test"
        desc = f"preparing {stage.lower()} images"
        total = len(dataset)
        
        for idx in tqdm(range(total), desc=desc, ncols=100, leave=False):
            image, label = dataset[idx]
            image_id = self._generate_iam_id()
            
            processed_image = self._process_image(image)
            save_path = self.image_dir / f"{image_id}.png"
            processed_image.save(str(save_path), optimize=True)
            
            x = 408 + (idx % 10) * 113
            y = 768 + (idx // 10) * 113
            record = f"{image_id} ok 154 1 {x} {y} 113 113 DT {label}"
            records.append(record)
            
            if (idx + 1) % 1000 == 0:
                self.logger.info(f"[MNIST] {stage}: {idx + 1:,}/{total:,}")
                
        return records

    def _create_documentation(self, train_records, test_records):
        with open(self.doc_path, 'w', encoding='utf-8') as f:
            f.write("# MNIST Database - IAM Format\n")
            f.write("# format: image_id ok graylevel components x y w h tag transcription\n")
            f.write("# DT: Digit Transcription\n\n")
            for record in train_records + test_records:
                f.write(f"{record}\n")

    def _cleanup_download(self):
        """Remove mnist directory after Preparing is complete"""
        mnist_download_dir = self.data_dir / 'mnist'
        if mnist_download_dir.exists():
            try:
                shutil.rmtree(mnist_download_dir)
                self.logger.info(f"Cleaned up temporary directory: {mnist_download_dir}")
            except Exception as e:
                self.logger.error(f"Failed to clean up directory {mnist_download_dir}: {str(e)}")

    def run(self):
        try:
            self._create_dirs()
            self.logger.info("[MNIST] Starting download...")
            
            datasets.MNIST.mirrors = ['https://ossci-datasets.s3.amazonaws.com/mnist/']
            
            train_dataset = datasets.MNIST(root=str(self.data_dir), train=True, download=True)
            test_dataset = datasets.MNIST(root=str(self.data_dir), train=False, download=True)
            
            train_records = self._save_images(train_dataset, is_train=True)
            test_records = self._save_images(test_dataset, is_train=False)
            
            self._create_documentation(train_records, test_records)
            self.logger.info(f"[MNIST] Complete - Train: {len(train_records):,}, Test: {len(test_records):,}")
            
            # Call cleanup
            self._cleanup_download()
            
            return True

        except Exception as e:
            self.logger.error(f"[MNIST] Error: {str(e)}")
            return False

if __name__ == "__main__":
    from src.lib.utils import setup_logging
    setup_logging()
    downloader = MNISTDatasetDownloader()
    success = downloader.run()