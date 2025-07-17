import numpy as np
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class WordEmbeddingsTrainer:
    """
    A class to train word embeddings using both CBOW and Skip-gram models.
    """
    
    def __init__(self, vector_dim: int = 100, context_window: int = 5, learning_rate: float = 0.01, 
                 epochs: int = 100, min_count: int = 1):
        """
        Initialize the word embeddings trainer.
        
        Args:
            vector_dim: Dimension of word vectors
            context_window: Size of context window (words on each side)
            learning_rate: Learning rate for gradient descent
            epochs: Number of training epochs
            min_count: Minimum word frequency to include in vocabulary
        """
        self.vector_dim = vector_dim
        self.context_window = context_window
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_count = min_count
        
        # Vocabulary and mappings
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        
        # Neural network weights
        self.W1 = None  # Input to hidden layer
        self.W2 = None  # Hidden to output layer
        
        # Training data
        self.training_data = []
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess the input text by tokenizing and cleaning.
        
        Args:
            text: Input text document
            
        Returns:
            List of cleaned tokens
        """
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = text.split()
        
        # Remove empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def build_vocabulary(self, tokens: List[str]) -> None:
        """
        Build vocabulary from tokens.
        
        Args:
            tokens: List of tokens from the document
        """
        # Count word frequencies
        word_counts = Counter(tokens)
        
        # Filter words by minimum count
        filtered_words = {word: count for word, count in word_counts.items() 
                         if count >= self.min_count}
        
        # Create vocabulary mappings
        self.vocab = filtered_words
        self.word_to_idx = {word: idx for idx, word in enumerate(filtered_words.keys())}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.vocab)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Most common words: {sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:10]}")
    
    def generate_training_data_cbow(self, tokens: List[str]) -> List[Tuple[List[int], int]]:
        """
        Generate training data for CBOW model.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of (context_words, target_word) pairs
        """
        training_data = []
        
        for i in range(len(tokens)):
            if tokens[i] not in self.word_to_idx:
                continue
                
            target_word = self.word_to_idx[tokens[i]]
            context_words = []
            
            # Get context words
            for j in range(max(0, i - self.context_window), 
                          min(len(tokens), i + self.context_window + 1)):
                if i != j and tokens[j] in self.word_to_idx:
                    context_words.append(self.word_to_idx[tokens[j]])
            
            if context_words:
                training_data.append((context_words, target_word))
        
        return training_data
    
    def generate_training_data_skipgram(self, tokens: List[str]) -> List[Tuple[int, int]]:
        """
        Generate training data for Skip-gram model.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of (target_word, context_word) pairs
        """
        training_data = []
        
        for i in range(len(tokens)):
            if tokens[i] not in self.word_to_idx:
                continue
                
            target_word = self.word_to_idx[tokens[i]]
            
            # Get context words
            for j in range(max(0, i - self.context_window), 
                          min(len(tokens), i + self.context_window + 1)):
                if i != j and tokens[j] in self.word_to_idx:
                    context_word = self.word_to_idx[tokens[j]]
                    training_data.append((target_word, context_word))
        
        return training_data
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax function."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def initialize_weights(self) -> None:
        """Initialize neural network weights."""
        # Xavier initialization
        self.W1 = np.random.uniform(-1.0, 1.0, (self.vocab_size, self.vector_dim)) / np.sqrt(self.vocab_size)
        self.W2 = np.random.uniform(-1.0, 1.0, (self.vector_dim, self.vocab_size)) / np.sqrt(self.vector_dim)
    
    def train_cbow(self, tokens: List[str]) -> List[float]:
        """
        Train CBOW model.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of losses for each epoch
        """
        print("Training CBOW model...")
        
        # Generate training data
        training_data = self.generate_training_data_cbow(tokens)
        print(f"Generated {len(training_data)} training examples")
        
        # Initialize weights
        self.initialize_weights()
        
        losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            
            for context_words, target_word in training_data:
                # Forward pass
                # Average context word vectors
                context_vectors = self.W1[context_words]
                h = np.mean(context_vectors, axis=0)
                
                # Compute output
                u = np.dot(h, self.W2)
                y_pred = self.softmax(u)
                
                # Compute loss
                loss = -np.log(y_pred[target_word])
                epoch_loss += loss
                
                # Backward pass
                # Output layer gradients
                dL_du = y_pred.copy()
                dL_du[target_word] -= 1
                
                # Hidden layer gradients
                dL_dW2 = np.outer(h, dL_du)
                dL_dh = np.dot(self.W2, dL_du)
                
                # Input layer gradients
                dL_dW1 = np.zeros_like(self.W1)
                for word_idx in context_words:
                    dL_dW1[word_idx] += dL_dh / len(context_words)
                
                # Update weights
                self.W1 -= self.learning_rate * dL_dW1
                self.W2 -= self.learning_rate * dL_dW2
            
            avg_loss = epoch_loss / len(training_data)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def train_skipgram(self, tokens: List[str]) -> List[float]:
        """
        Train Skip-gram model.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of losses for each epoch
        """
        print("Training Skip-gram model...")
        
        # Generate training data
        training_data = self.generate_training_data_skipgram(tokens)
        print(f"Generated {len(training_data)} training examples")
        
        # Initialize weights
        self.initialize_weights()
        
        losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            
            for target_word, context_word in training_data:
                # Forward pass
                h = self.W1[target_word]
                u = np.dot(h, self.W2)
                y_pred = self.softmax(u)
                
                # Compute loss
                loss = -np.log(y_pred[context_word])
                epoch_loss += loss
                
                # Backward pass
                # Output layer gradients
                dL_du = y_pred.copy()
                dL_du[context_word] -= 1
                
                # Hidden layer gradients
                dL_dW2 = np.outer(h, dL_du)
                dL_dh = np.dot(self.W2, dL_du)
                
                # Input layer gradients
                dL_dW1 = np.zeros_like(self.W1)
                dL_dW1[target_word] = dL_dh
                
                # Update weights
                self.W1 -= self.learning_rate * dL_dW1
                self.W2 -= self.learning_rate * dL_dW2
            
            avg_loss = epoch_loss / len(training_data)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get word vector for a given word.
        
        Args:
            word: Word to get vector for
            
        Returns:
            Word vector or None if word not in vocabulary
        """
        if word in self.word_to_idx:
            return self.W1[self.word_to_idx[word]]
        return None
    
    def find_similar_words(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar words to a given word.
        
        Args:
            word: Target word
            top_k: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        if word not in self.word_to_idx:
            return []
        
        target_vector = self.get_word_vector(word)
        similarities = []
        
        for other_word in self.word_to_idx:
            if other_word != word:
                other_vector = self.get_word_vector(other_word)
                similarity = cosine_similarity([target_vector], [other_vector])[0][0]
                similarities.append((other_word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        model_data = {
            'W1': self.W1,
            'W2': self.W2,
            'vocab': self.vocab,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vector_dim': self.vector_dim,
            'context_window': self.context_window
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.W1 = model_data['W1']
        self.W2 = model_data['W2']
        self.vocab = model_data['vocab']
        self.word_to_idx = model_data['word_to_idx']
        self.idx_to_word = model_data['idx_to_word']
        self.vector_dim = model_data['vector_dim']
        self.context_window = model_data['context_window']
        self.vocab_size = len(self.vocab)
        
        print(f"Model loaded from {filepath}")
    
    def visualize_embeddings(self, words: List[str] = None, save_path: str = None) -> None:
        """
        Visualize word embeddings using PCA.
        
        Args:
            words: List of words to visualize (if None, uses most frequent words)
            save_path: Path to save the visualization
        """
        if words is None:
            # Use most frequent words
            words = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)[:20]
            words = [word for word, _ in words]
        
        # Get word vectors
        vectors = []
        valid_words = []
        
        for word in words:
            vector = self.get_word_vector(word)
            if vector is not None:
                vectors.append(vector)
                valid_words.append(word)
        
        if len(vectors) < 2:
            print("Not enough words to visualize")
            return
        
        # Reduce dimensions using PCA
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
        
        # Add labels
        for i, word in enumerate(valid_words):
            plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))
        
        plt.title('Word Embeddings Visualization (PCA)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


def main():
    """Main function to demonstrate the word embeddings trainer."""
    
    # Example usage
    sample_text = """
    Natural language processing is a subfield of linguistics, computer science, and artificial intelligence
    concerned with the interactions between computers and human language, in particular how to program
    computers to process and analyze large amounts of natural language data. The goal is a computer
    capable of understanding the contents of documents, including the contextual nuances of the language
    within them. The technology can then accurately extract information and insights contained in the
    documents as well as categorize and organize the documents themselves. Natural language processing
    has its roots in the 1950s. Already in 1950, Alan Turing published an article titled Computing
    Machinery and Intelligence which proposed what is now called the Turing test as a criterion of
    intelligence. The Georgetown experiment in 1954 involved fully automatic translation of more than
    sixty Russian sentences into English. The authors claimed that within three or five years, machine
    translation would be a solved problem. However, real progress was much slower, and after the ALPAC
    report in 1966, which found that ten-year-long research had failed to fulfill the expectations,
    funding for machine translation was dramatically reduced. Little further research in machine
    translation was conducted until the late 1980s when the first statistical machine translation
    systems were developed.
    """
    
    print("=== Word Embeddings Training Demo ===")
    print(f"Sample text length: {len(sample_text)} characters")
    
    # Initialize trainer
    trainer = WordEmbeddingsTrainer(
        vector_dim=50,
        context_window=3,
        learning_rate=0.01,
        epochs=50,
        min_count=1
    )
    
    # Preprocess text
    tokens = trainer.preprocess_text(sample_text)
    print(f"Number of tokens: {len(tokens)}")
    
    # Build vocabulary
    trainer.build_vocabulary(tokens)
    
    # Train CBOW model
    print("\n" + "="*50)
    cbow_losses = trainer.train_cbow(tokens)
    
    # Save CBOW model
    trainer.save_model("cbow_model.pkl")
    
    # Test CBOW model
    print("\n=== CBOW Model Results ===")
    test_words = ["language", "computer", "natural", "processing"]
    for word in test_words:
        similar_words = trainer.find_similar_words(word, top_k=3)
        print(f"Words similar to '{word}': {similar_words}")
    
    # Train Skip-gram model
    print("\n" + "="*50)
    skipgram_losses = trainer.train_skipgram(tokens)
    
    # Save Skip-gram model
    trainer.save_model("skipgram_model.pkl")
    
    # Test Skip-gram model
    print("\n=== Skip-gram Model Results ===")
    for word in test_words:
        similar_words = trainer.find_similar_words(word, top_k=3)
        print(f"Words similar to '{word}': {similar_words}")
    
    # Plot training losses
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(cbow_losses)
    plt.title('CBOW Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(skipgram_losses)
    plt.title('Skip-gram Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_losses.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualize embeddings
    trainer.visualize_embeddings(save_path='embeddings_visualization.png')


if __name__ == "__main__":
    main()