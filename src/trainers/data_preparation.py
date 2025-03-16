"""
Data preparation tools for the NEAT project.

This module provides functionality for preparing training and evaluation data,
including downloading datasets, generating synthetic data, and creating
data splits.
"""

import os
import sys
import logging
import random
import json
import gzip
import shutil
import time
from pathlib import Path
from tqdm import tqdm
import requests
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

logger = logging.getLogger(__name__)

# Common Crawl sample URL template
CC_WARC_URL_TEMPLATE = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2021-04/segments/1610704576172.0/warc/CC-MAIN-20210115104816-20210115134816-{:05d}.warc.gz"

def download_file(url: str, output_path: str, max_retries: int = 3) -> bool:
    """
    Download a file with retries.
    
    Args:
        url: URL to download
        output_path: Path to save the downloaded file
        max_retries: Maximum number of retries
        
    Returns:
        True if download successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {url} (attempt {attempt+1})")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(output_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=url.split('/')[-1]) as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.info(f"Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                logger.error(f"Max retries reached for {url}")
                return False

def extract_text_from_warc(warc_file: str, output_dir: str, max_docs: int = 1000, 
                           min_size: int = 1024, max_size: int = 51200) -> int:
    """
    Extract text content from WARC file.
    
    Args:
        warc_file: Path to WARC file
        output_dir: Directory to save extracted text files
        max_docs: Maximum number of documents to extract
        min_size: Minimum document size in bytes
        max_size: Maximum document size in bytes
        
    Returns:
        Number of documents extracted
    """
    try:
        # Create a temporary file from the gzipped WARC
        temp_path = warc_file + ".extracted"
        with gzip.open(warc_file, 'rb') as f_in:
            with open(temp_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Process the uncompressed WARC
        with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Split by WARC record
        records = content.split("WARC/1.0")
        extracted = 0
        
        # Process each record
        for i, record in enumerate(records):
            if "Content-Type: text/html" in record and "HTTP/1.1 200 OK" in record:
                # Extract HTML content
                try:
                    html_start = record.find("<html")
                    if html_start == -1:
                        html_start = record.find("<HTML")
                    
                    if html_start > 0:
                        html_content = record[html_start:]
                        
                        # Simple text extraction (very basic)
                        text = ""
                        in_tag = False
                        for char in html_content:
                            if char == '<':
                                in_tag = True
                            elif char == '>':
                                in_tag = False
                                text += ' '
                            elif not in_tag:
                                text += char
                        
                        # Clean up whitespace
                        text = ' '.join(text.split())
                        
                        # Check if content is of appropriate size
                        if min_size <= len(text) <= max_size:
                            output_file = os.path.join(output_dir, f"doc_{extracted:05d}.txt")
                            with open(output_file, 'w', encoding='utf-8') as f_out:
                                f_out.write(text)
                            extracted += 1
                            
                            if extracted >= max_docs:
                                break
                except Exception as e:
                    logger.error(f"Error processing record {i}: {e}")
                    continue
        
        # Clean up
        os.remove(temp_path)
        return extracted
    except Exception as e:
        logger.error(f"Error extracting from WARC file: {e}")
        return 0

def download_pile_subset(config: Any) -> Dict[str, int]:
    """
    Download and process a subset of the Pile dataset.
    
    Args:
        config: Configuration object with download settings
        
    Returns:
        Dictionary with counts of training and evaluation files
    """
    # Get parameters from config
    output_dir = config.pile_output_dir if hasattr(config, 'pile_output_dir') else './data/pile_subset'
    train_dir = config.train_dir if hasattr(config, 'train_dir') else os.path.join(output_dir, 'train')
    eval_dir = config.eval_dir if hasattr(config, 'eval_dir') else os.path.join(output_dir, 'eval')
    sample_count = config.pile_warc_count if hasattr(config, 'pile_warc_count') else 5
    
    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Initialize counters
    train_size = 0
    eval_size = 0
    
    # Download literature texts from Project Gutenberg
    try:
        logger.info("Downloading literature samples from Project Gutenberg")
        gutenberg_samples = [
            "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
            "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein
            "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
            "https://www.gutenberg.org/files/76/76-0.txt",      # Huckleberry Finn
            "https://www.gutenberg.org/files/1661/1661-0.txt",  # Sherlock Holmes
            "https://www.gutenberg.org/files/345/345-0.txt",    # Dracula
            "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
            "https://www.gutenberg.org/files/2814/2814-0.txt",  # Dubliners
            "https://www.gutenberg.org/files/174/174-0.txt",    # The Picture of Dorian Gray
            "https://www.gutenberg.org/files/1400/1400-0.txt"   # Great Expectations
        ]
        
        for i, url in enumerate(gutenberg_samples):
            try:
                target_dir = train_dir if i < 8 else eval_dir  # 80/20 split
                file_name = f"lit_{url.split('/')[-2]}_{i}.txt"
                output_path = os.path.join(target_dir, file_name)
                
                response = requests.get(url)
                content = response.text
                
                # Take a subset to keep files manageable
                content = content[:100000]  # First 100KB
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                if target_dir == train_dir:
                    train_size += 1
                else:
                    eval_size += 1
                    
                logger.info(f"Downloaded literature sample {i+1}/{len(gutenberg_samples)}")
            except Exception as e:
                logger.error(f"Error downloading literature sample {url}: {e}")
                
    except Exception as e:
        logger.error(f"Error downloading literature samples: {e}")
    
    # Add Wikipedia content for variety
    try:
        logger.info("Downloading Wikipedia sample articles")
        wiki_titles = [
            "Neural_network", "Transformer_(machine_learning_model)", 
            "Entropy_(information_theory)", "Entropy", "Machine_learning",
            "Artificial_intelligence", "Deep_learning", "Computer_vision",
            "Natural_language_processing", "Reinforcement_learning"
        ]
        
        for i, title in enumerate(wiki_titles):
            url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&titles={title}"
            response = requests.get(url)
            data = response.json()
            
            # Extract the page content
            page = next(iter(data['query']['pages'].values()))
            content = page.get('extract', '')
            
            # Save to a file
            output_path = os.path.join(train_dir, f"wiki_{i:03d}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
    except Exception as e:
        logger.error(f"Error downloading Wikipedia content: {e}")
    
    # Add scientific and technical texts
    try:
        logger.info("Downloading scientific papers and technical texts")
        paper_samples = [
            "https://arxiv.org/pdf/1706.03762.pdf",  # Transformer paper
            "https://arxiv.org/pdf/1810.04805.pdf",  # BERT paper
            "https://arxiv.org/pdf/2005.14165.pdf",  # GPT-3 paper
            "https://arxiv.org/pdf/2108.07258.pdf",  # Codex paper
            "https://arxiv.org/pdf/2203.15556.pdf",  # InstructGPT paper
            "https://arxiv.org/pdf/2204.02311.pdf",  # PaLM paper
        ]
        
        # Use technical documentation instead of PDFs which might be harder to parse
        tech_docs = [
            "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/README.rst",
            "https://raw.githubusercontent.com/pytorch/pytorch/main/README.md",
            "https://raw.githubusercontent.com/tensorflow/tensorflow/master/README.md",
            "https://raw.githubusercontent.com/huggingface/transformers/main/README.md",
            "https://raw.githubusercontent.com/numpy/numpy/main/README.md",
            "https://raw.githubusercontent.com/pandas-dev/pandas/main/README.md",
            "https://raw.githubusercontent.com/rust-lang/rust/master/README.md",
            "https://raw.githubusercontent.com/golang/go/master/README.md",
            "https://raw.githubusercontent.com/python/cpython/main/README.rst",
            "https://raw.githubusercontent.com/kubernetes/kubernetes/master/README.md"
        ]
        
        for i, url in enumerate(tech_docs):
            try:
                target_dir = train_dir if i < 8 else eval_dir  # 80/20 split
                file_name = f"tech_doc_{i}.txt"
                output_path = os.path.join(target_dir, file_name)
                
                response = requests.get(url)
                content = response.text
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                if target_dir == train_dir:
                    train_size += 1
                else:
                    eval_size += 1
                    
                logger.info(f"Downloaded technical document {i+1}/{len(tech_docs)}")
            except Exception as e:
                logger.error(f"Error downloading technical document {url}: {e}")
    except Exception as e:
        logger.error(f"Error downloading technical texts: {e}")
    
    # Add code samples
    try:
        code_samples = [
            ("python", "https://raw.githubusercontent.com/pytorch/pytorch/master/torch/nn/modules/transformer.py"),
            ("python", "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/gpt2/modeling_gpt2.py"),
            ("python", "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/python/keras/layers/core.py"),
            ("python", "https://raw.githubusercontent.com/numpy/numpy/main/numpy/core/numeric.py"),
            ("python", "https://raw.githubusercontent.com/django/django/main/django/db/models/base.py"),
            ("python", "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/linear_model/_base.py"),
            ("python", "https://raw.githubusercontent.com/matplotlib/matplotlib/main/lib/matplotlib/figure.py"),
            ("python", "https://raw.githubusercontent.com/pandas-dev/pandas/main/pandas/core/frame.py"),
            ("c++", "https://raw.githubusercontent.com/google/leveldb/main/db/db_impl.cc"),
            ("c++", "https://raw.githubusercontent.com/protocolbuffers/protobuf/main/src/google/protobuf/message.cc"),
            ("c++", "https://raw.githubusercontent.com/facebook/folly/main/folly/String.cpp"),
            ("java", "https://raw.githubusercontent.com/spring-projects/spring-framework/main/spring-core/src/main/java/org/springframework/core/io/Resource.java"),
            ("java", "https://raw.githubusercontent.com/apache/hadoop/trunk/hadoop-common-project/hadoop-common/src/main/java/org/apache/hadoop/fs/FileSystem.java"),
            ("javascript", "https://raw.githubusercontent.com/facebook/react/main/packages/react/src/React.js"),
            ("javascript", "https://raw.githubusercontent.com/d3/d3/main/src/array/index.js"),
            ("javascript", "https://raw.githubusercontent.com/lodash/lodash/master/lodash.js"),
            ("rust", "https://raw.githubusercontent.com/rust-lang/rust/master/compiler/rustc_middle/src/ty/mod.rs"),
            ("rust", "https://raw.githubusercontent.com/tokio-rs/tokio/master/tokio/src/runtime/mod.rs")
        ]
        
        for i, (lang, url) in enumerate(code_samples):
            try:
                response = requests.get(url)
                content = response.text
                
                # Save to training directory
                output_path = os.path.join(train_dir, f"code_{lang}_{i:02d}.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                train_size += 1
            except Exception as e:
                logger.error(f"Error downloading code sample {url}: {e}")
    except Exception as e:
        logger.error(f"Error downloading code samples: {e}")
    
    # Add JSON and other format samples
    try:
        json_samples = [
            "https://raw.githubusercontent.com/nlp-datasets/wikitext/master/wikitext-103/wiki.train.tokens",
            "https://raw.githubusercontent.com/huggingface/datasets/main/datasets/squad/squad.py",
            "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/pipelines/text2text_generation.py"
        ]
        
        for i, url in enumerate(json_samples):
            try:
                response = requests.get(url)
                content = response.text
                
                # Take a subset (first 50KB) to avoid huge files
                content = content[:50000]
                
                # Save to directory (alternating between train and eval)
                target_dir = train_dir if i % 2 == 0 else eval_dir
                output_path = os.path.join(target_dir, f"format_sample_{i:02d}.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                if target_dir == train_dir:
                    train_size += 1
                else:
                    eval_size += 1
            except Exception as e:
                logger.error(f"Error downloading sample {url}: {e}")
    except Exception as e:
        logger.error(f"Error downloading format samples: {e}")
    
    logger.info(f"Dataset creation complete: {train_size} training files, {eval_size} evaluation files")
    
    # Create a summary file
    summary_path = os.path.join(output_dir, "pile_subset_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Pile Subset Summary\n")
        f.write(f"=================\n\n")
        f.write(f"Training files: {train_size}\n")
        f.write(f"Evaluation files: {eval_size}\n\n")
        f.write(f"Source composition:\n")
        f.write(f"- Common Crawl web content\n")
        f.write(f"- Wikipedia articles\n")
        f.write(f"- Code samples (Python, C++, Java, JavaScript, Rust)\n")
        f.write(f"- JSON and other structured data\n\n")
        f.write(f"This small subset of The Pile is intended for BLT entropy estimator training.\n")
    
    return {
        "train_size": train_size,
        "eval_size": eval_size,
        "train_dir": train_dir,
        "eval_dir": eval_dir
    }

def prepare_data(config: Any) -> Dict[str, Any]:
    """
    Prepare data based on the specified data type.
    
    Args:
        config: Configuration object with data preparation settings
        
    Returns:
        Dictionary with data preparation results
    """
    if not hasattr(config, 'data_type'):
        logger.error("Missing data_type in configuration")
        return {"error": "Missing data_type"}
    
    # Handle different data types
    if config.data_type == "pile_subset":
        return download_pile_subset(config)
    elif config.data_type == "byte_level":
        # Not implemented yet, placeholder for future implementation
        logger.info("Byte-level data preparation not yet implemented")
        return {"error": "Not implemented"}
    elif config.data_type == "synthetic_math":
        # Not implemented yet, placeholder for future implementation
        logger.info("Synthetic math data preparation not yet implemented")
        return {"error": "Not implemented"}
    elif config.data_type == "component_test":
        # Not implemented yet, placeholder for future implementation
        logger.info("Component test data preparation not yet implemented")
        return {"error": "Not implemented"}
    else:
        logger.error(f"Unknown data type: {config.data_type}")
        return {"error": f"Unknown data type: {config.data_type}"}

def create_mock_models(config: Any) -> Dict[str, str]:
    """
    Create mock models for testing.
    
    Args:
        config: Configuration object with mock model settings
        
    Returns:
        Dictionary with paths to created mock models
    """
    import torch
    import torch.nn as nn
    
    output_dir = config.output_dir if hasattr(config, 'output_dir') else './outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directories for mock models
    blt_dir = os.path.join(output_dir, "mock_blt")
    mvot_dir = os.path.join(output_dir, "mock_mvot")
    os.makedirs(blt_dir, exist_ok=True)
    os.makedirs(mvot_dir, exist_ok=True)
    
    # Create a mock BLT model
    logger.info("Creating mock BLT model...")
    class MockByteLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(256, 64)
            self.lstm = nn.LSTM(64, 128, batch_first=True)
            self.fc = nn.Linear(128, 256)
            self.config = {"model_type": "SmallByteLM", "hidden_size": 128, "num_layers": 1}
        
        def forward(self, input_ids, labels=None):
            embedded = self.embedding(input_ids)
            output, _ = self.lstm(embedded)
            logits = self.fc(output)
            
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, 256), labels.view(-1))
                return {"loss": loss, "logits": logits}
            
            return {"logits": logits}
        
        def generate_probs(self, input_bytes):
            # Simple implementation for testing
            input_ids = torch.tensor([[b for b in input_bytes]], dtype=torch.long)
            with torch.no_grad():
                logits = self.forward(input_ids)["logits"]
                probs = torch.softmax(logits, dim=-1)
            return probs
    
    mock_blt = MockByteLM()
    blt_path = os.path.join(blt_dir, "best_model.pt")
    torch.save({
        "model_state_dict": mock_blt.state_dict(),
        "config": mock_blt.config,
        "global_step": 1000,
        "epoch": 5,
        "best_loss": 2.5
    }, blt_path)
    
    # Create a mock MVoT visual codebook
    logger.info("Creating mock MVoT visual codebook...")
    class MockVisualCodebook(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8192, 512)
            self.config = {"model_type": "VQCodebook", "embedding_dim": 512, "codebook_size": 8192}
        
        def forward(self, indices):
            return self.embedding(indices)
        
        def get_codebook(self):
            return self.embedding.weight
    
    mock_mvot = MockVisualCodebook()
    mvot_path = os.path.join(mvot_dir, "codebook.pt")
    torch.save({
        "model_state_dict": mock_mvot.state_dict(),
        "config": mock_mvot.config
    }, mvot_path)
    
    logger.info(f"Mock models created at {blt_path} and {mvot_path}")
    
    # Create some test data if requested
    if hasattr(config, 'create_training_data') and config.create_training_data:
        logger.info("Creating test training data...")
        test_data_dir = os.path.join(output_dir, "test_data")
        test_train_dir = os.path.join(test_data_dir, "train")
        test_eval_dir = os.path.join(test_data_dir, "eval")
        os.makedirs(test_train_dir, exist_ok=True)
        os.makedirs(test_eval_dir, exist_ok=True)
        
        # Create a few test files
        for i in range(10):
            with open(os.path.join(test_train_dir, f"test_{i}.txt"), "w") as f:
                f.write(f"This is test file {i} for training.\n" * 50)
        
        for i in range(5):
            with open(os.path.join(test_eval_dir, f"test_{i}.txt"), "w") as f:
                f.write(f"This is test file {i} for evaluation.\n" * 50)
        
        logger.info(f"Test data created at {test_data_dir}")
    
    return {
        "blt_path": blt_path,
        "mvot_path": mvot_path
    }