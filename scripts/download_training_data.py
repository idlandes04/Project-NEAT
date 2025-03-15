#!/usr/bin/env python3
"""
Script to download and prepare training data for BLT entropy estimator and MVoT visual codebook.

This script downloads and preprocesses data for both the byte-level entropy estimator
and the visual codebook training.
"""

import os
import sys
import argparse
import logging
import urllib.request
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download a file from URL to the specified output path with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_archive(archive_path, extract_dir):
    """Extract various archive formats."""
    logger.info(f"Extracting {archive_path} to {extract_dir}")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(extract_dir)
    elif archive_path.endswith('.gz') and not archive_path.endswith('.tar.gz'):
        # Single gzipped file (not tar.gz)
        output_path = archive_path[:-3]  # Remove .gz extension
        with gzip.open(archive_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        logger.error(f"Unsupported archive format: {archive_path}")


def download_byte_training_data(args):
    """Download and prepare data for the BLT entropy estimator."""
    # Create byte training and eval directories
    os.makedirs(args.byte_train_dir, exist_ok=True)
    os.makedirs(args.byte_eval_dir, exist_ok=True)
    
    # Project Gutenberg training data
    gutenberg_files = [
        # Classics, various formats and content for entropy diversity
        "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein
        "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
        "https://www.gutenberg.org/files/76/76-0.txt",      # Adventures of Huckleberry Finn
        "https://www.gutenberg.org/files/145/145-0.txt",    # Middlemarch
        "https://www.gutenberg.org/files/1400/1400-0.txt",  # Great Expectations
        "https://www.gutenberg.org/files/16389/16389-0.txt" # The Republic by Plato
    ]
    
    # Download Project Gutenberg files for training
    logger.info("Downloading Project Gutenberg text files for byte training")
    for i, url in enumerate(gutenberg_files):
        filename = f"gutenberg_{i}.txt"
        output_path = os.path.join(args.byte_train_dir, filename)
        try:
            download_url(url, output_path)
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
    
    # Sample from C4 dataset (English web text)
    # For simplicity, we'll use a small subset from Hugging Face datasets
    if args.download_c4:
        try:
            logger.info("Downloading C4 dataset sample (this may take some time)")
            from datasets import load_dataset
            
            # Load a sample from the C4 dataset
            c4_sample = load_dataset("c4", "en", split="train", streaming=True)
            
            # Save a few thousand examples
            with open(os.path.join(args.byte_train_dir, "c4_sample.txt"), "w", encoding="utf-8") as f:
                for i, example in enumerate(c4_sample):
                    f.write(example["text"] + "\n\n")
                    if i >= 2000:  # Limit to 2000 examples to keep file size manageable
                        break
            
            logger.info("C4 sample downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading C4 dataset: {e}")
            logger.info("Skipping C4 dataset, continuing with other sources")
    
    # Create evaluation data with different texts
    eval_files = [
        "https://www.gutenberg.org/files/98/98-0.txt",      # A Tale of Two Cities
        "https://www.gutenberg.org/files/1661/1661-0.txt",  # The Adventures of Sherlock Holmes
        "https://www.gutenberg.org/files/219/219-0.txt"     # Heart of Darkness
    ]
    
    logger.info("Downloading evaluation text files")
    for i, url in enumerate(eval_files):
        filename = f"eval_{i}.txt"
        output_path = os.path.join(args.byte_eval_dir, filename)
        try:
            download_url(url, output_path)
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
    
    logger.info(f"Byte-level training data downloaded to {args.byte_train_dir} and {args.byte_eval_dir}")


def download_visual_codebook_data(args):
    """Download pretrained visual codebook model or training data."""
    os.makedirs(args.visual_dir, exist_ok=True)
    
    # Since we need a pretrained VQ-VAE model, we'll download a small one
    # For demonstration, we'll use a mock VQ-VAE for testing
    logger.info("Creating mock visual codebook for testing and development")
    
    # Create a simple Python script that generates a mock visual codebook
    mock_script_path = os.path.join(args.visual_dir, "create_mock_codebook.py")
    with open(mock_script_path, "w") as f:
        f.write('import torch\n')
        f.write('import os\n\n')
        f.write('def create_mock_vqvae_codebook(output_path, codebook_size=8192, embedding_dim=512):\n')
        f.write('    """Create a mock VQVAE codebook for testing."""\n')
        f.write('    # Create random embeddings\n')
        f.write('    embeddings = torch.randn(codebook_size, embedding_dim)\n')
        f.write('    \n')
        f.write('    # Normalize embeddings\n')
        f.write('    embeddings = torch.nn.functional.normalize(embeddings, dim=1)\n')
        f.write('    \n')
        f.write('    # Create a state dict similar to a real VQVAE\n')
        f.write('    state_dict = {\n')
        f.write('        "quantize.embedding.weight": embeddings\n')
        f.write('    }\n')
        f.write('    \n')
        f.write('    # Save the state dict\n')
        f.write('    os.makedirs(os.path.dirname(output_path), exist_ok=True)\n')
        f.write('    torch.save(state_dict, output_path)\n')
        f.write('    \n')
        f.write('    print(f"Created mock VQVAE codebook at {output_path}")\n')
        f.write('    print(f"Codebook size: {codebook_size}, Embedding dim: {embedding_dim}")\n\n')
        f.write('if __name__ == "__main__":\n')
        f.write('    output_dir = os.path.dirname(os.path.abspath(__file__))\n')
        f.write('    create_mock_vqvae_codebook(os.path.join(output_dir, "mock_vqvae_codebook.pt"))\n')
    
    # Run the mock script to generate a test codebook
    logger.info("Running script to create mock codebook")
    os.system(f"python {mock_script_path}")
    
    logger.info(f"Visual codebook training data prepared in {args.visual_dir}")


def download_math_dataset(args):
    """Download the DeepMind Mathematics Dataset for synthetic data inspiration."""
    output_dir = os.path.join(args.output_dir, "mathematics_dataset")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # URL for the DeepMind Mathematics Dataset
        url = "https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz"
        archive_path = os.path.join(output_dir, "mathematics_dataset-v1.0.tar.gz")
        
        # Download the dataset
        logger.info("Downloading DeepMind Mathematics Dataset (this may take a while)")
        download_url(url, archive_path)
        
        # Extract the dataset
        logger.info("Extracting the mathematics dataset")
        extract_archive(archive_path, output_dir)
        
        logger.info(f"Mathematics dataset downloaded and extracted to {output_dir}")
    except Exception as e:
        logger.error(f"Error downloading mathematics dataset: {e}")
        logger.info("Skipping mathematics dataset, you can manually download it later.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download training data for BLT and MVoT")
    
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Directory to save the downloaded data')
    parser.add_argument('--byte_train_dir', type=str, default='./data/byte_training',
                        help='Directory to save byte-level training data')
    parser.add_argument('--byte_eval_dir', type=str, default='./data/byte_eval',
                        help='Directory to save byte-level evaluation data')
    parser.add_argument('--visual_dir', type=str, default='./data/visual_training',
                        help='Directory to save visual codebook data')
    parser.add_argument('--download_c4', action='store_true',
                        help='Download sample from C4 dataset (requires Hugging Face datasets)')
    parser.add_argument('--download_math', action='store_true',
                        help='Download DeepMind Mathematics Dataset')
    parser.add_argument('--all', action='store_true',
                        help='Download all types of data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.all or True:
        # Download all data types by default
        download_byte_training_data(args)
        download_visual_codebook_data(args)
        
        if args.download_math:
            download_math_dataset(args)
    else:
        # Selective download based on args
        if args.download_c4:
            download_byte_training_data(args)
        if args.download_math:
            download_math_dataset(args)
    
    logger.info("Data download complete!")


if __name__ == "__main__":
    main()