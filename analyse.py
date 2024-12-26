import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple

class DatasetAnalyzer:
    def __init__(self, data_path: Path):
        """
        Initialize the Dataset Analyzer class
        Input: File path
        """
        self.data_path = data_path
        self.word_info = defaultdict(int)
        self.char_counts = defaultdict(int)
        self.total_samples = 0
        self.ok_samples = 0
        self.error_samples = 0
        
    def read_data(self) -> None:
        """
        Read the file and perform initial processing
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            # Skip comment lines
            lines = [l.strip() for l in f.readlines() if not l.startswith('#')]
            
        for line in lines:
            if not line:
                continue
                
            # Split the line into parts
            parts = line.split()
            if len(parts) < 9:
                continue
                
            status = parts[1]  # Status: 'ok' or 'er'
            word = parts[-1]   # The word itself
            
            # Count samples
            self.total_samples += 1
            if status == 'ok':
                self.ok_samples += 1
                # Count characters in correct samples
                for char in word:
                    self.char_counts[char] += 1
            else:
                self.error_samples += 1
            
            # Count word lengths
            self.word_info[len(word)] += 1
            
    def print_stats(self) -> None:
        """
        Display overall dataset statistics
        """
        print("\n=== Overall Dataset Statistics ===")
        print(f"Total samples: {self.total_samples:,}")
        print(f"Correct samples: {self.ok_samples:,}")
        print(f"Error samples: {self.error_samples:,}")
        
        print("\n=== Word Length Distribution ===")
        for length, count in sorted(self.word_info.items()):
            print(f"Words with length {length}: {count:,}")
            
        print("\n=== Top 10 Most Frequent Characters ===")
        chars = sorted(self.char_counts.items(), key=lambda x: x[1], reverse=True)
        for char, count in chars[:40]:
            print(f"'{char}': {count:,}")

def main():
    # File path
    data_file = Path("data/raw/words_new.txt")
    
    # Analyze the dataset
    analyzer = DatasetAnalyzer(data_file)
    analyzer.read_data()
    analyzer.print_stats()

if __name__ == "__main__":
    main()