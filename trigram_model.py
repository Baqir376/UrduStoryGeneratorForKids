"""
Phase III: Trigram Language Model for Urdu Story Generation
- Maximum Likelihood Estimation (MLE)
- Interpolation: λ3·P_tri + λ2·P_bi + λ1·P_uni
- Temperature scaling for randomness control
- Repetition penalty to avoid loops
- Generation until <EOT> or max_length
"""

import json
import math
import random
from collections import Counter, defaultdict
from bpe_tokenizer import BPETokenizer, EOS, EOP, EOT

CORPUS_FILE = 'processed_corpus.txt'
MODEL_FILE = 'trigram_model.json'


class TrigramModel:
    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        self.total_tokens = 0
        self.vocab_size = 0
        # Interpolation weights
        self.lambda1 = 0.1   # unigram
        self.lambda2 = 0.3   # bigram
        self.lambda3 = 0.6   # trigram
    
    def train(self, token_ids):
        """Train trigram model on list of token IDs."""
        print(f"Training trigram model on {len(token_ids):,} tokens...")
        
        self.total_tokens = len(token_ids)
        self.vocab_size = len(set(token_ids))
        
        # Count unigrams
        self.unigram_counts = Counter(token_ids)
        
        # Count bigrams
        self.bigram_counts = defaultdict(Counter)
        for i in range(len(token_ids) - 1):
            self.bigram_counts[token_ids[i]][token_ids[i + 1]] += 1
        
        # Count trigrams
        self.trigram_counts = defaultdict(Counter)
        for i in range(len(token_ids) - 2):
            key = (token_ids[i], token_ids[i + 1])
            self.trigram_counts[str(key)][token_ids[i + 2]] += 1
        
        # Learn interpolation weights using held-out estimation (deleted interpolation)
        self._learn_interpolation_weights(token_ids)
        
        # With subword BPE tokens, deleted interpolation tends to over-weight unigrams
        # because subword tokens are very frequent. Override with better defaults
        # that favor higher-order n-grams for coherent generation.
        self.lambda1 = 0.05   # unigram - small weight
        self.lambda2 = 0.25   # bigram
        self.lambda3 = 0.70   # trigram - dominant
        
        print(f"Unigrams: {len(self.unigram_counts)}")
        print(f"Bigram contexts: {len(self.bigram_counts)}")
        print(f"Trigram contexts: {len(self.trigram_counts)}")
        print(f"Interpolation weights (tuned): λ1={self.lambda1:.3f}, λ2={self.lambda2:.3f}, λ3={self.lambda3:.3f}")
    
    def _learn_interpolation_weights(self, token_ids):
        """Learn interpolation weights using deleted interpolation."""
        l1, l2, l3 = 0, 0, 0
        
        for i in range(2, len(token_ids)):
            w1, w2, w3 = token_ids[i - 2], token_ids[i - 1], token_ids[i]
            
            # Trigram MLE
            tri_key = str((w1, w2))
            bi_context_count = sum(self.trigram_counts.get(tri_key, {}).values())
            if bi_context_count > 1:
                p3 = (self.trigram_counts.get(tri_key, {}).get(w3, 0) - 1) / (bi_context_count - 1)
            else:
                p3 = 0
            
            # Bigram MLE
            uni_context_count = sum(self.bigram_counts.get(w2, {}).values())
            if uni_context_count > 1:
                p2 = (self.bigram_counts.get(w2, {}).get(w3, 0) - 1) / (uni_context_count - 1)
            else:
                p2 = 0
            
            # Unigram MLE
            if self.total_tokens > 1:
                p1 = (self.unigram_counts.get(w3, 0) - 1) / (self.total_tokens - 1)
            else:
                p1 = 0
            
            # Assign to the highest probability
            max_p = max(p1, p2, p3)
            if max_p == p3 and max_p > 0:
                l3 += self.trigram_counts.get(tri_key, {}).get(w3, 0)
            elif max_p == p2 and max_p > 0:
                l2 += self.bigram_counts.get(w2, {}).get(w3, 0)
            elif max_p > 0:
                l1 += self.unigram_counts.get(w3, 0)
        
        total = l1 + l2 + l3
        if total > 0:
            self.lambda1 = l1 / total
            self.lambda2 = l2 / total
            self.lambda3 = l3 / total
    
    def _get_probability(self, w3, w1, w2):
        """Get interpolated probability P(w3 | w1, w2)."""
        # Unigram probability with Laplace smoothing
        p_uni = (self.unigram_counts.get(w3, 0) + 1) / (self.total_tokens + self.vocab_size)
        
        # Bigram probability
        bi_denom = sum(self.bigram_counts.get(w2, {}).values())
        if bi_denom > 0:
            p_bi = self.bigram_counts.get(w2, {}).get(w3, 0) / bi_denom
        else:
            p_bi = 0
        
        # Trigram probability
        tri_key = str((w1, w2))
        tri_denom = sum(self.trigram_counts.get(tri_key, {}).values())
        if tri_denom > 0:
            p_tri = self.trigram_counts.get(tri_key, {}).get(w3, 0) / tri_denom
        else:
            p_tri = 0
        
        # Interpolation
        return self.lambda1 * p_uni + self.lambda2 * p_bi + self.lambda3 * p_tri
    
    def _get_next_token_probs(self, w1, w2, temperature=1.0, repetition_penalty=1.0, recent_tokens=None):
        """Get probability distribution for next token given context."""
        if recent_tokens is None:
            recent_tokens = set()
        
        # Collect candidates
        candidates = set()
        
        # From trigram context
        tri_key = str((w1, w2))
        if tri_key in self.trigram_counts:
            candidates.update(int(k) for k in self.trigram_counts[tri_key].keys())
        
        # From bigram context
        if w2 in self.bigram_counts:
            candidates.update(int(k) for k in self.bigram_counts[w2].keys())
        
        # Add some unigram candidates for diversity
        top_unigrams = self.unigram_counts.most_common(100)
        candidates.update(t for t, _ in top_unigrams)
        
        if not candidates:
            candidates = set(self.unigram_counts.keys())
        
        # Compute probabilities
        probs = {}
        for token in candidates:
            p = self._get_probability(token, w1, w2)
            if p > 0:
                # Apply repetition penalty
                if token in recent_tokens and repetition_penalty > 1.0:
                    p /= repetition_penalty
                probs[token] = p
        
        if not probs:
            return {}
        
        # Apply temperature
        if temperature != 1.0:
            log_probs = {}
            for token, p in probs.items():
                if p > 0:
                    log_probs[token] = math.log(p) / max(temperature, 0.01)
            
            # Softmax
            max_log = max(log_probs.values())
            exp_probs = {t: math.exp(lp - max_log) for t, lp in log_probs.items()}
            total = sum(exp_probs.values())
            probs = {t: ep / total for t, ep in exp_probs.items()}
        else:
            total = sum(probs.values())
            probs = {t: p / total for t, p in probs.items()}
        
        return probs
    
    def generate(self, tokenizer, prefix="", max_length=100, temperature=0.8, repetition_penalty=1.2):
        """Generate text given a prefix."""
        # Encode prefix
        if prefix:
            tokens = tokenizer.encode(prefix)
        else:
            # Start with a random bigram from the corpus
            common_starts = self.unigram_counts.most_common(20)
            tokens = [random.choice(common_starts)[0]]
        
        # Ensure at least 2 tokens for trigram context
        while len(tokens) < 2:
            probs = {}
            for token, count in self.bigram_counts.get(tokens[-1], {}).items():
                probs[int(token)] = count
            if probs:
                total = sum(probs.values())
                probs = {t: c / total for t, c in probs.items()}
                tokens.append(self._sample(probs))
            else:
                common = self.unigram_counts.most_common(10)
                tokens.append(random.choice(common)[0])
        
        # Get EOT token ID
        eot_id = tokenizer.token_to_id.get(EOT, -1)
        
        # Track recent tokens for repetition penalty
        recent_window = 30
        
        generated = 0
        while generated < max_length:
            w1, w2 = tokens[-2], tokens[-1]
            recent = set(tokens[-recent_window:]) if len(tokens) > recent_window else set(tokens)
            
            probs = self._get_next_token_probs(w1, w2, temperature, repetition_penalty, recent)
            
            if not probs:
                break
            
            next_token = self._sample(probs)
            tokens.append(next_token)
            generated += 1
            
            # Stop at EOT
            if next_token == eot_id:
                break
        
        # Decode
        return tokenizer.decode(tokens)
    
    def generate_streaming(self, tokenizer, prefix="", max_length=100, temperature=0.8, repetition_penalty=1.2):
        """Generator that yields tokens one at a time for streaming."""
        if prefix:
            tokens = tokenizer.encode(prefix)
        else:
            common_starts = self.unigram_counts.most_common(20)
            tokens = [random.choice(common_starts)[0]]
        
        while len(tokens) < 2:
            probs = {}
            for token, count in self.bigram_counts.get(tokens[-1], {}).items():
                probs[int(token)] = count
            if probs:
                total = sum(probs.values())
                probs = {t: c / total for t, c in probs.items()}
                tokens.append(self._sample(probs))
            else:
                common = self.unigram_counts.most_common(10)
                tokens.append(random.choice(common)[0])
        
        # Yield the prefix first
        yield tokenizer.decode(tokens)
        
        eot_id = tokenizer.token_to_id.get(EOT, -1)
        recent_window = 30
        generated = 0
        
        while generated < max_length:
            w1, w2 = tokens[-2], tokens[-1]
            recent = set(tokens[-recent_window:]) if len(tokens) > recent_window else set(tokens)
            
            probs = self._get_next_token_probs(w1, w2, temperature, repetition_penalty, recent)
            
            if not probs:
                break
            
            next_token = self._sample(probs)
            tokens.append(next_token)
            generated += 1
            
            if next_token == eot_id:
                break
            
            # Yield current decoded text
            yield tokenizer.decode(tokens)
    
    def _sample(self, probs):
        """Sample from probability distribution."""
        tokens = list(probs.keys())
        weights = list(probs.values())
        return random.choices(tokens, weights=weights, k=1)[0]
    
    def save(self, filepath=MODEL_FILE):
        """Save model to JSON."""
        data = {
            'unigram_counts': dict(self.unigram_counts),
            'bigram_counts': {str(k): dict(v) for k, v in self.bigram_counts.items()},
            'trigram_counts': {k: dict(v) for k, v in self.trigram_counts.items()},
            'total_tokens': self.total_tokens,
            'vocab_size': self.vocab_size,
            'lambda1': self.lambda1,
            'lambda2': self.lambda2,
            'lambda3': self.lambda3
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath=MODEL_FILE):
        """Load model from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.unigram_counts = Counter({int(k): v for k, v in data['unigram_counts'].items()})
        self.bigram_counts = defaultdict(Counter)
        for k, v in data['bigram_counts'].items():
            self.bigram_counts[int(k)] = Counter({int(tk): tv for tk, tv in v.items()})
        self.trigram_counts = defaultdict(Counter)
        for k, v in data['trigram_counts'].items():
            self.trigram_counts[k] = Counter({int(tk): tv for tk, tv in v.items()})
        self.total_tokens = data['total_tokens']
        self.vocab_size = data['vocab_size']
        self.lambda1 = data['lambda1']
        self.lambda2 = data['lambda2']
        self.lambda3 = data['lambda3']
        print(f"Model loaded from {filepath}")
        print(f"Interpolation weights: λ1={self.lambda1:.3f}, λ2={self.lambda2:.3f}, λ3={self.lambda3:.3f}")


def train_model():
    """Train and save the trigram model."""
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load()
    
    # Load and tokenize corpus
    print("Loading corpus...")
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("Tokenizing corpus...")
    token_ids = tokenizer.encode(text)
    print(f"Total tokens: {len(token_ids):,}")
    
    # Train model
    model = TrigramModel()
    model.train(token_ids)
    model.save()
    
    # Test generation
    print("\n--- Generation Test ---")
    test_prefixes = ["ایک دن"]
    for prefix in test_prefixes:
        result = model.generate(tokenizer, prefix=prefix, max_length=80, temperature=0.7, repetition_penalty=1.3)
        print(f"\nPrefix: {prefix}")
        print(f"Generated: {result}")


if __name__ == '__main__':
    train_model()
