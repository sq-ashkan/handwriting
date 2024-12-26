from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import logging
import shutil
from src.lib.constants import EMNIST_DIR

class EMNISTDatasetDownloader:
    def __init__(self):
        self.data_dir = EMNIST_DIR
        self.image_dir = EMNIST_DIR / 'images'
        self.doc_path = EMNIST_DIR / 'documentation.txt'
        self.transform = transforms.ToTensor()
        self.current_id = 1

    def _create_dirs(self):
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def _generate_iam_id(self):
        writer_id = f"a{self.current_id:02d}"
        doc_id = f"{self.current_id:03d}u"
        image_id = f"{writer_id}-{doc_id}-00-00"
        self.current_id += 1
        return image_id

    def _process_image(self, image):
        image_array = np.array(image)
        image_array = np.rot90(image_array, k=3)
        image_array = np.fliplr(image_array)
        pil_image = Image.fromarray((image_array * 255).astype('uint8'))
        target_height = 27
        aspect_ratio = pil_image.size[0] / pil_image.size[1]
        target_width = int(target_height * aspect_ratio)
        target_width = min(max(target_width, 27), 51)
        resized_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return resized_image

    def _save_images(self, dataset):
        records = []
        for idx, (image, label) in enumerate(tqdm(dataset, desc="Preparing images")):
            image_id = self._generate_iam_id()
            processed_image = self._process_image(image)
            
            char = chr(label + ord('A') - 1)
            
            save_path = self.image_dir / f"{image_id}.png"
            processed_image.save(str(save_path))
            
            x = 408 + (idx % 10) * 51
            y = 768 + (idx // 10) * 27
            w, h = processed_image.size
            
            record = f"{image_id} ok 154 1 {x} {y} {w} {h} AT {char}"
            records.append(record)
            
        return records

    def _create_documentation(self, records):
        with open(self.doc_path, 'w', encoding='utf-8') as f:
            f.write("# EMNIST Database - IAM Format\n")
            f.write("# format: image_id ok graylevel components x y w h tag transcription\n\n")
            for record in records:
                f.write(f"{record}\n")

    def _cleanup_download(self):
        """Remove EMNIST directory after Preparing is complete"""
        emnist_download_dir = self.data_dir / 'EMNIST'
        if emnist_download_dir.exists():
            try:
                shutil.rmtree(emnist_download_dir)
                logging.info(f"Cleaned up temporary directory: {emnist_download_dir}")
            except Exception as e:
                logging.error(f"Failed to clean up directory {emnist_download_dir}: {str(e)}")

    def run(self):
        try:
            self._create_dirs()
            logging.info("Starting EMNIST download...")

            dataset = datasets.EMNIST(
                root=str(self.data_dir),
                split='letters',
                train=True,
                download=True
            )

            records = self._save_images(dataset)
            self._create_documentation(records)
            
            # Clean up after successful Preparing
            self._cleanup_download()

            logging.info("EMNIST dataset downloaded and processed successfully")
            return True

        except Exception as e:
            logging.error(f"Error: {str(e)}")
            return False

if __name__ == "__main__":
    from src.lib.utils import setup_logging
    setup_logging()
    downloader = EMNISTDatasetDownloader()
    success = downloader.run()