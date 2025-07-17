#!/usr/bin/env python3
"""
Command-line interface for training word embeddings using CBOW and Skip-gram models.

Usage:
    python train_embeddings.py --input document.txt --vector_dim 100 --context_window 5 --model cbow
    python train_embeddings.py --input document.txt --vector_dim 100 --context_window 5 --model skipgram
    python train_embeddings.py --input document.txt --vector_dim 100 --context_window 5 --model both
"""

import argparse
import sys
import os
from word_embeddings import WordEmbeddingsTrainer
import matplotlib.pyplot as plt

def read_document(filepath: str) -> str:
    """Read text document from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Train word embeddings using CBOW and Skip-gram models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_embeddings.py --input document.txt --vector_dim 100 --context_window 5 --model cbow
  python train_embeddings.py --input document.txt --vector_dim 50 --context_window 3 --model skipgram
  python train_embeddings.py --input document.txt --vector_dim 100 --context_window 5 --model both
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input text document'
    )
    
    parser.add_argument(
        '--vector_dim', '-d',
        type=int,
        default=100,
        help='Dimension of word vectors (default: 100)'
    )
    
    parser.add_argument(
        '--context_window', '-w',
        type=int,
        default=5,
        help='Context window size (default: 5)'
    )
    
    parser.add_argument(
        '--model', '-m',
        choices=['cbow', 'skipgram', 'both'],
        default='both',
        help='Model type to train (default: both)'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--learning_rate', '-lr',
        type=float,
        default=0.01,
        help='Learning rate (default: 0.01)'
    )
    
    parser.add_argument(
        '--min_count', '-mc',
        type=int,
        default=1,
        help='Minimum word frequency to include in vocabulary (default: 1)'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        default='./output',
        help='Output directory for models and visualizations (default: ./output)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization of embeddings'
    )
    
    parser.add_argument(
        '--test_words',
        nargs='+',
        help='Words to test similarity for (default: most frequent words)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read input document
    print(f"Reading document from: {args.input}")
    document_text = read_document(args.input)
    print(f"Document length: {len(document_text)} characters")
    
    # Initialize trainer
    trainer = WordEmbeddingsTrainer(
        vector_dim=args.vector_dim,
        context_window=args.context_window,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        min_count=args.min_count
    )
    
    # Preprocess text
    print("Preprocessing text...")
    tokens = trainer.preprocess_text(document_text)
    print(f"Number of tokens: {len(tokens)}")
    
    # Build vocabulary
    print("Building vocabulary...")
    trainer.build_vocabulary(tokens)
    
    if trainer.vocab_size == 0:
        print("Error: Empty vocabulary. Please check your input document.")
        sys.exit(1)
    
    # Training results
    results = {}
    
    # Train CBOW model
    if args.model in ['cbow', 'both']:
        print(f"\n{'='*60}")
        print("TRAINING CBOW MODEL")
        print(f"{'='*60}")
        
        cbow_losses = trainer.train_cbow(tokens)
        results['cbow_losses'] = cbow_losses
        
        # Save CBOW model
        cbow_model_path = os.path.join(args.output_dir, 'cbow_model.pkl')
        trainer.save_model(cbow_model_path)
        
        # Test CBOW model
        print(f"\n{'='*40}")
        print("CBOW MODEL RESULTS")
        print(f"{'='*40}")
        
        test_words = args.test_words if args.test_words else get_test_words(trainer)
        
        for word in test_words:
            if word in trainer.word_to_idx:
                similar_words = trainer.find_similar_words(word, top_k=5)
                print(f"Words similar to '{word}': {similar_words}")
            else:
                print(f"Word '{word}' not found in vocabulary")
    
    # Train Skip-gram model
    if args.model in ['skipgram', 'both']:
        print(f"\n{'='*60}")
        print("TRAINING SKIP-GRAM MODEL")
        print(f"{'='*60}")
        
        skipgram_losses = trainer.train_skipgram(tokens)
        results['skipgram_losses'] = skipgram_losses
        
        # Save Skip-gram model
        skipgram_model_path = os.path.join(args.output_dir, 'skipgram_model.pkl')
        trainer.save_model(skipgram_model_path)
        
        # Test Skip-gram model
        print(f"\n{'='*40}")
        print("SKIP-GRAM MODEL RESULTS")
        print(f"{'='*40}")
        
        test_words = args.test_words if args.test_words else get_test_words(trainer)
        
        for word in test_words:
            if word in trainer.word_to_idx:
                similar_words = trainer.find_similar_words(word, top_k=5)
                print(f"Words similar to '{word}': {similar_words}")
            else:
                print(f"Word '{word}' not found in vocabulary")
    
    # Plot training losses
    if results:
        plot_losses(results, args.output_dir)
    
    # Visualize embeddings
    if args.visualize:
        print(f"\n{'='*40}")
        print("GENERATING VISUALIZATION")
        print(f"{'='*40}")
        
        viz_path = os.path.join(args.output_dir, 'embeddings_visualization.png')
        trainer.visualize_embeddings(save_path=viz_path)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    print(f"Models and visualizations saved to: {os.path.abspath(args.output_dir)}")

def get_test_words(trainer: WordEmbeddingsTrainer, num_words: int = 5) -> list:
    """Get most frequent words for testing."""
    if not trainer.vocab:
        return []
    
    # Get most frequent words
    frequent_words = sorted(trainer.vocab.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in frequent_words[:num_words]]

def plot_losses(results: dict, output_dir: str):
    """Plot training losses."""
    plt.figure(figsize=(15, 5))
    
    plot_count = len(results)
    
    if 'cbow_losses' in results:
        plt.subplot(1, plot_count, 1)
        plt.plot(results['cbow_losses'])
        plt.title('CBOW Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
    
    if 'skipgram_losses' in results:
        subplot_idx = 2 if 'cbow_losses' in results else 1
        plt.subplot(1, plot_count, subplot_idx)
        plt.plot(results['skipgram_losses'])
        plt.title('Skip-gram Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    losses_path = os.path.join(output_dir, 'training_losses.png')
    plt.savefig(losses_path, dpi=300, bbox_inches='tight')
    print(f"Training losses plot saved to: {losses_path}")
    
    # Don't show the plot in background mode
    # plt.show()

if __name__ == "__main__":
    main()