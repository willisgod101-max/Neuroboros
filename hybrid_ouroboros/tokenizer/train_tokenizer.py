import sentencepiece as spm
import argparse
import os

def train_tokenizer(input_file, vocab_size, model_prefix):
    """
    Train a SentencePiece tokenizer on the input file.
    Args:
        input_file: Path to the input text file.
        vocab_size: Desired vocabulary size.
        model_prefix: Output prefix for the model files (will generate .model and .vocab files).
    """
    # Ensure the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Train the SentencePiece model
    spm.SentencePieceTrainer.train(
        input=input_file,
        vocab_size=vocab_size,
        model_type='bpe',  # Byte Pair Encoding
        model_prefix=model_prefix,
        character_coverage=1.0,  # For English, we can set to 1.0
        pad_id=0,  # We'll define our own special tokens later if needed
        unk_id=1,
        bos_id=2,  # Beginning of sentence
        eos_id=3,  # End of sentence
    )
    print(f"Tokenizer model saved to {model_prefix}.model")
    print(f"Tokenizer vocabulary saved to {model_prefix}.vocab")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a SentencePiece tokenizer')
    parser.add_argument('--input', type=str, required=True, help='Path to input text file')
    parser.add_argument('--vocab_size', type=int, default=8000, help='Vocabulary size')
    parser.add_argument('--model_prefix', type=str, required=True, help='Output model prefix')
    args = parser.parse_args()
    
    train_tokenizer(args.input, args.vocab_size, args.model_prefix)