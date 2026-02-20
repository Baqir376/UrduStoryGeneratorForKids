"""
Generic N-gram Language Model for Urdu Story Generation
- Supports N=3 (Trigram), 5, 7, etc.
- Maximum Likelihood Estimation (MLE)
- Interpolation: λn·P_n + ... + λ1·P_uni
- Temperature scaling and repetition penalty
- Generation until <EOT> or max_length
"""

import json
import math
import random
import os
from collections import Counter, defaultdict
from bpe_tokenizer import BPETokenizer, EOS, EOP, EOT

class NGramModel:
    def __init__(self, n=3):
        self.n = n
        # counts[i] stores counts for (i+1)-grams
        # counts[0] = unigrams (token -> count)
        # counts[1] = bigrams (w1 -> {w2 -> count})
        # counts[n-1] = n-grams ((w1...wn-1) -> {wn -> count})
        self.counts = [defaultdict(Counter) if i > 0 else Counter() for i in range(n)]
        self.total_tokens = 0
        self.vocab_size = 0
        # Interpolation weights λ1 (unigram) to λn (n-gram)
        self.lambdas = [1.0 / n] * n
    
    def train(self, token_ids):
        """Train n-gram model on list of token IDs."""
        print(f"Training {self.n}-gram model on {len(token_ids):,} tokens...")
        
        self.total_tokens = len(token_ids)
        self.vocab_size = len(set(token_ids))
        
        # Unigrams
        self.counts[0] = Counter(token_ids)
        
        # Higher order grams
        for i in range(1, self.n):
            print(f"  Counting {i+1}-grams...")
            for j in range(len(token_ids) - i):
                if i == 1:
                    # Bigram: w1 -> w2
                    self.counts[i][token_ids[j]][token_ids[j+1]] += 1
                else:
                    # N-gram: (w1...wn-1) -> wn
                    context = tuple(token_ids[j:j+i])
                    # JSON key needs to be string
                    self.counts[i][str(context)][token_ids[j+i]] += 1
        
        # Simple backoff-friendly weights
        # Higher order gets more weight if possible
        total_weight = 0
        for i in range(self.n):
            weight = (i + 1) ** 2
            self.lambdas[i] = weight
            total_weight += weight
        
        self.lambdas = [w / total_weight for w in self.lambdas]
        
        print(f"Model trained. Vocab size: {self.vocab_size}")
        print(f"Interpolation weights: {', '.join([f'λ{i+1}={l:.3f}' for i, l in enumerate(self.lambdas)])}")

    def _get_ngram_prob(self, token, context_tuple):
        """Get P(token | context) for a specific order."""
        order = len(context_tuple) + 1
        if order == 1:
            # Unigram with Laplace smoothing
            return (self.counts[0].get(token, 0) + 1) / (self.total_tokens + self.vocab_size)
        
        context_key = str(context_tuple) if order > 2 else context_tuple[0]
        context_counts = self.counts[order-1].get(context_key, {})
        denom = sum(context_counts.values())
        
        if denom > 0:
            return context_counts.get(token, 0) / denom
        return 0.0

    def get_interpolated_prob(self, token, full_context_tuple):
        """Get interpolated probability P(token | context)."""
        prob = 0.0
        # λ1*P(uni) + λ2*P(bi) + ... + λn*P(n-gram)
        for i in range(self.n):
            # context for i+1-gram has length i
            if i == 0:
                p = self._get_ngram_prob(token, ())
            else:
                # Use suffix of context for lower orders
                sub_context = full_context_tuple[-i:]
                p = self._get_ngram_prob(token, sub_context)
            prob += self.lambdas[i] * p
        return prob

    def _get_next_token_probs(self, context_tuple, temperature=1.0, repetition_penalty=1.0, recent_tokens=None):
        if recent_tokens is None:
            recent_tokens = set()
            
        candidates = set()
        # Collect candidates from all orders to ensure we have options
        for i in range(self.n - 1, 0, -1):
            sub_context = context_tuple[-i:]
            ctx_key = str(sub_context) if i > 1 else sub_context[0]
            if ctx_key in self.counts[i]:
                candidates.update(int(k) for k in self.counts[i][ctx_key].keys())
            if len(candidates) > 50: break
            
        # Add some top unigrams for diversity
        top_unigrams = self.counts[0].most_common(50)
        candidates.update(t for t, _ in top_unigrams)
        
        if not candidates:
            candidates = set(self.counts[0].keys())
            
        probs = {}
        for token in candidates:
            p = self.get_interpolated_prob(token, context_tuple)
            if p > 0:
                if token in recent_tokens and repetition_penalty > 1.0:
                    p /= repetition_penalty
                probs[token] = p
                
        if not probs: return {}
        
        # Temperature & Softmax
        if temperature != 1.0:
            log_probs = {t: math.log(p) / max(temperature, 0.01) for t, p in probs.items() if p > 0}
            max_log = max(log_probs.values())
            exp_probs = {t: math.exp(lp - max_log) for t, lp in log_probs.items()}
            total = sum(exp_probs.values())
            probs = {t: ep / total for t, ep in exp_probs.items()}
        else:
            total = sum(probs.values())
            probs = {t: p / total for t, p in probs.items()}
            
        return probs

    def generate_streaming(self, tokenizer, prefix="", max_length=100, temperature=0.8, repetition_penalty=1.2):
        if prefix:
            tokens = tokenizer.encode(prefix)
        else:
            common_starts = self.counts[0].most_common(20)
            tokens = [random.choice(common_starts)[0]]
            
        # Ensure we have enough context for N-gram
        while len(tokens) < self.n - 1:
            # Generate one by one using available context
            curr_ctx = tuple(tokens)
            probs = self._get_next_token_probs(curr_ctx, temperature, repetition_penalty)
            if not probs: break
            tokens.append(self._sample(probs))

        yield tokenizer.decode(tokens)
        
        eot_id = tokenizer.token_to_id.get(EOT, -1)
        recent_window = 30
        generated = 0
        
        while generated < max_length:
            context = tuple(tokens[-(self.n-1):])
            recent = set(tokens[-recent_window:]) if len(tokens) > recent_window else set(tokens)
            probs = self._get_next_token_probs(context, temperature, repetition_penalty, recent)
            
            if not probs: break
            
            next_token = self._sample(probs)
            tokens.append(next_token)
            generated += 1
            
            if next_token == eot_id: break
            yield tokenizer.decode(tokens)

    def generate(self, tokenizer, prefix="", max_length=100, temperature=0.8, repetition_penalty=1.2):
        gen = self.generate_streaming(tokenizer, prefix, max_length, temperature, repetition_penalty)
        final_text = ""
        for text in gen:
            final_text = text
        return final_text

    def _sample(self, probs):
        tokens = list(probs.keys())
        weights = list(probs.values())
        return random.choices(tokens, weights=weights, k=1)[0]

    def save(self, filepath):
        data = {
            'n': self.n,
            'counts': [],
            'total_tokens': self.total_tokens,
            'vocab_size': self.vocab_size,
            'lambdas': self.lambdas
        }
        # Unigrams
        data['counts'].append(dict(self.counts[0]))
        # Higher orders
        for i in range(1, self.n):
            data['counts'].append({str(k): dict(v) for k, v in self.counts[i].items()})
            
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found.")
            
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.n = data['n']
        self.total_tokens = data['total_tokens']
        self.vocab_size = data['vocab_size']
        self.lambdas = data['lambdas']
        
        self.counts = [defaultdict(Counter) if i > 0 else Counter() for i in range(self.n)]
        
        # Unigrams
        self.counts[0] = Counter({int(k): v for k, v in data['counts'][0].items()})
        
        # Higher orders
        for i in range(1, self.n):
            raw_counts = data['counts'][i]
            for k, v in raw_counts.items():
                # Context key is either int (bigram) or string tuple (n-gram)
                if i == 1:
                    ctx_key = int(k)
                else:
                    ctx_key = k
                self.counts[i][ctx_key] = Counter({int(tk): tv for tk, tv in v.items()})
                
        print(f"Model loaded from {filepath} (N={self.n})")

def get_model_filename(n, vocab_size):
    return f"ngram_n{n}_v{vocab_size}.json"
