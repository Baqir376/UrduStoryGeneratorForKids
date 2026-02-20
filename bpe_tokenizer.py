"""
Phase II: Custom BPE (Byte Pair Encoding) Tokenizer for Urdu
- 100% custom implementation (no libraries)
- Trained on preprocessed corpus
- Vocabulary size: 250  
- Handles special tokens: EOS, EOP, EOT
"""

import json
import re
import os
from collections import Counter, defaultdict

EOS = '\u0600'
EOP = '\u0601'
EOT = '\u0602'
SPECIAL_TOKENS = [EOS, EOP, EOT]

CORPUS_FILE = 'processed_corpus.txt'
VOCAB_FILE = 'bpe_vocab.json'


class BPETokenizer:
    def __init__(self):
        self.vocab = {}          # token_id -> token_string
        self.token_to_id = {}    # token_string -> token_id
        self.merges = []         # list of (pair_a, pair_b) merge rules in order
    
    def _get_word_freqs(self, text):
        """Split text into words and count frequencies. Each word is a tuple of characters."""
        # Split on whitespace but keep special tokens as single units
        words = text.split()
        word_freqs = Counter()
        for word in words:
            # Convert word to tuple of characters, adding end-of-word marker
            chars = tuple(word) + ('</w>',)
            word_freqs[chars] += 1
        return word_freqs
    
    def _get_pair_counts(self, word_freqs):
        """Count frequency of adjacent pairs across all words."""
        pairs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += freq
        return pairs
    
    def _merge_pair(self, word_freqs, pair):
        """Merge the most frequent pair in all words."""
        new_word_freqs = {}
        merged = pair[0] + pair[1]
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def train(self, text, vocab_size=250):
        """Train BPE tokenizer on text."""
        print("Training BPE tokenizer...")
        print(f"Target vocab size: {vocab_size}")
        
        # Initialize vocabulary with all unique characters
        chars = set()
        for ch in text:
            chars.add(ch)
        chars.add('</w>')  # End of word marker
        
        # Build initial vocab: character-level
        self.vocab = {}
        self.token_to_id = {}
        
        # Reserve IDs for special tokens first
        for i, token in enumerate(SPECIAL_TOKENS):
            self.vocab[i] = token
            self.token_to_id[token] = i
        
        # Add </w> token
        idx = len(SPECIAL_TOKENS)
        self.vocab[idx] = '</w>'
        self.token_to_id['</w>'] = idx
        idx += 1
        
        # Add all unique characters (excluding special tokens already added)
        for ch in sorted(chars):
            if ch not in self.token_to_id:
                self.vocab[idx] = ch
                self.token_to_id[ch] = idx
                idx += 1
        
        print(f"Initial vocab size (characters): {len(self.vocab)}")
        
        # Get word frequencies
        word_freqs = self._get_word_freqs(text)
        print(f"Unique words: {len(word_freqs)}")
        
        # BPE merges
        self.merges = []
        num_merges = vocab_size - len(self.vocab)
        
        if num_merges <= 0:
            print(f"Character vocab already >= {vocab_size}, no merges needed.")
            return
        
        print(f"Performing {num_merges} merges...")
        
        for merge_i in range(num_merges):
            pair_counts = self._get_pair_counts(word_freqs)
            if not pair_counts:
                print(f"No more pairs to merge at step {merge_i}")
                break
            
            best_pair = pair_counts.most_common(1)[0]
            pair, count = best_pair
            
            if count < 2:
                print(f"Best pair frequency < 2 at step {merge_i}, stopping.")
                break
            
            # Merge the pair
            word_freqs = self._merge_pair(word_freqs, pair)
            merged_token = pair[0] + pair[1]
            self.merges.append(pair)
            
            # Add new token to vocab
            self.vocab[idx] = merged_token
            self.token_to_id[merged_token] = idx
            idx += 1
            
            if (merge_i + 1) % 20 == 0:
                print(f"  Merge {merge_i + 1}/{num_merges}: '{pair[0]}' + '{pair[1]}' -> '{merged_token}' (freq={count})")
        
        print(f"Final vocab size: {len(self.vocab)}")
    
    def _apply_merges(self, word):
        """Apply learned merge rules to a word (list of tokens)."""
        word = list(word)
        for pair in self.merges:
            i = 0
            while i < len(word) - 1:
                if word[i] == pair[0] and word[i + 1] == pair[1]:
                    word[i:i + 2] = [pair[0] + pair[1]]
                else:
                    i += 1
        return word
    
    def encode(self, text):
        """Encode text to list of token IDs."""
        tokens = []
        words = text.split()
        
        for word in words:
            # Start with characters
            chars = list(word) + ['</w>']
            # Apply BPE merges
            subwords = self._apply_merges(chars)
            # Convert to IDs
            for sw in subwords:
                if sw in self.token_to_id:
                    tokens.append(self.token_to_id[sw])
                else:
                    # Unknown subword: encode character by character
                    for ch in sw:
                        if ch in self.token_to_id:
                            tokens.append(self.token_to_id[ch])
                        # else skip unknown char
        return tokens
    
    def decode(self, token_ids):
        """Decode token IDs back to text."""
        tokens = []
        for tid in token_ids:
            tid = int(tid)
            if tid in self.vocab:
                tokens.append(self.vocab[tid])
        
        # Join tokens and handle </w> markers (replace with space)
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.rstrip()
    
    def save(self, vocab_size=None, filepath=None):
        """Save tokenizer to JSON file."""
        if not filepath:
            size = vocab_size if vocab_size else len(self.vocab)
            filepath = f'bpe_vocab_{size}.json'
            
        data = {
            'vocab': {str(k): v for k, v in self.vocab.items()},
            'token_to_id': self.token_to_id,
            'merges': [[p[0], p[1]] for p in self.merges]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, vocab_size=250, filepath=None):
        """Load tokenizer from JSON file."""
        if not filepath:
            filepath = f'bpe_vocab_{vocab_size}.json'
            # Fallback to default if specific one doesn't exist
            if not os.path.exists(filepath) and vocab_size == 250:
                filepath = 'bpe_vocab.json'
            elif not os.path.exists(filepath):
                # If not found, try to load default and log a warning
                print(f"Warning: {filepath} not found, trying bpe_vocab.json")
                filepath = 'bpe_vocab.json'

        if not os.path.exists(filepath):
             raise FileNotFoundError(f"Vocab file {filepath} not found.")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = {int(k): v for k, v in data['vocab'].items()}
        self.token_to_id = data['token_to_id']
        self.merges = [tuple(p) for p in data['merges']]
        print(f"Tokenizer loaded from {filepath} (vocab size: {len(self.vocab)})")
    
    def vocab_size(self):
        return len(self.vocab)


def train_tokenizer():
    """Train and save the BPE tokenizer."""
    print(f"Loading corpus from {CORPUS_FILE}...")
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Corpus length: {len(text):,} characters")
    
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=250)
    tokenizer.save()
    
    # Test encode/decode
    test_text = text[:200].split(EOT)[0]  # First part of first story
    if test_text:
        encoded = tokenizer.encode(test_text[:100])
        decoded = tokenizer.decode(encoded)
        print(f"\n--- Encode/Decode Test ---")
        print(f"Original (first 100 chars): {test_text[:100]}")
        print(f"Encoded ({len(encoded)} tokens): {encoded[:20]}...")
        print(f"Decoded: {decoded[:100]}")


if __name__ == '__main__':
    train_tokenizer()
