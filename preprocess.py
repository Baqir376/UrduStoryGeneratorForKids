"""
Phase I: Dataset Preprocessing for Urdu Children's Stories
- Reads urdu_stories.txt
- Cleans text: removes HTML/ads, non-Urdu chars, normalizes Unicode, standardizes punctuation
- Adds special tokens: EOS (sentence), EOP (paragraph), EOT (story)
- Saves processed corpus
"""

import re
import unicodedata

# Special token code points (unused Arabic block points)
EOS = '\u0600'  # End of Sentence
EOP = '\u0601'  # End of Paragraph  
EOT = '\u0602'  # End of Story (Text)

INPUT_FILE = 'urdu_stories.txt'
OUTPUT_FILE = 'processed_corpus.txt'


def load_stories(filepath):
    """Load and split stories from the scraped file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by the separator line
    raw_stories = re.split(r'\n*={10,}\n*', content)
    stories = [s.strip() for s in raw_stories if s.strip()]
    print(f"Loaded {len(stories)} raw stories")
    return stories


def clean_story(text):
    """Clean a single story text."""
    # Remove the title line (first line is typically English title - Article No. XXXX)
    lines = text.split('\n')
    # Skip the title line (English + Article No.)
    content_lines = []
    started = False
    for line in lines:
        line = line.strip()
        if not line:
            if started:
                content_lines.append('')  # Keep empty lines as paragraph separators
            continue
        # Skip lines that are mostly English/Latin (title lines)
        urdu_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]', line))
        total_alpha = len(re.findall(r'[a-zA-Z\u0600-\u06FF]', line))
        if total_alpha > 0 and urdu_chars / max(total_alpha, 1) < 0.3 and not started:
            continue  # Skip English title lines at the beginning
        started = True
        content_lines.append(line)
    
    text = '\n'.join(content_lines)
    
    # Unicode NFKC normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Remove any HTML tags that might remain
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove English characters and digits (keep Urdu, Arabic script, spaces, punctuation)
    # Keep: Urdu/Arabic chars, Urdu punctuation, spaces, newlines
    text = re.sub(r'[a-zA-Z0-9]', '', text)
    
    # Standardize Urdu punctuation
    text = text.replace('۔۔۔', '…')
    text = text.replace('․․․', '…')
    text = text.replace('...', '…')
    text = text.replace('؟', '?')
    text = text.replace('!', '!')
    
    # Remove excessive punctuation like ----
    text = re.sub(r'[-_=]{3,}', '', text)
    
    # Remove non-Urdu, non-space, non-punctuation characters
    # Keep: Urdu/Arabic block, common punctuation, spaces, newlines
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s۔،؛:?!٫٬\-\(\)\"\'…""\']', '', text)
    
    # Normalize whitespace within lines (but preserve newlines for paragraph detection)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = re.sub(r'[ \t]+', ' ', line).strip()
        cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)
    
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def add_special_tokens(text):
    """Add EOS, EOP tokens to a cleaned story text."""
    paragraphs = re.split(r'\n\s*\n', text)
    processed_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Add EOS after sentence-ending punctuation
        # Urdu sentences end with ۔ (full stop), ? (question mark), ! (exclamation)
        para = re.sub(r'(۔)', r'\1' + EOS, para)
        para = re.sub(r'(\?)', r'\1' + EOS, para)
        para = re.sub(r'(!)', r'\1' + EOS, para)
        
        # Add EOP at end of paragraph
        para = para.rstrip() + EOP
        processed_paragraphs.append(para)
    
    # Join paragraphs and add EOT at end of story
    result = ' '.join(processed_paragraphs) + EOT
    return result


def preprocess_corpus():
    """Main preprocessing pipeline."""
    stories = load_stories(INPUT_FILE)
    
    processed_stories = []
    skipped = 0
    
    for i, story in enumerate(stories):
        cleaned = clean_story(story)
        
        # Skip stories that are too short after cleaning
        if len(cleaned) < 100:
            skipped += 1
            continue
        
        processed = add_special_tokens(cleaned)
        processed_stories.append(processed)
    
    print(f"Processed {len(processed_stories)} stories (skipped {skipped} too-short stories)")
    
    # Join all stories into one corpus
    corpus = '\n'.join(processed_stories)
    
    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(corpus)
    
    print(f"Saved processed corpus to {OUTPUT_FILE}")
    print(f"Corpus size: {len(corpus):,} characters")
    print(f"EOS tokens: {corpus.count(EOS):,}")
    print(f"EOP tokens: {corpus.count(EOP):,}")
    print(f"EOT tokens: {corpus.count(EOT):,}")
    
    # Show a sample
    sample = corpus[:500]
    print(f"\nSample (first 500 chars):\n{sample}")


if __name__ == '__main__':
    preprocess_corpus()
