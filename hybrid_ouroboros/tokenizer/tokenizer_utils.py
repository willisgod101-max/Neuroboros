import sentencepiece as spm
import os

class Tokenizer:
    def __init__(self, model_path):
        """
        Initialize the tokenizer with a SentencePiece model.
        Args:
            model_path: Path to the SentencePiece model file (.model).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)

    def encode(self, text):
        """
        Encode a string to a list of token IDs.
        Args:
            text: Input string.
        Returns:
            List of token IDs.
        """
        return self.sp_model.encode(text, out_type=int)

    def decode(self, ids):
        """
        Decode a list of token IDs to a string.
        Args:
            ids: List of token IDs.
        Returns:
            Decoded string.
        """
        return self.sp_model.decode(ids)

    def batch_encode(self, texts):
        """
        Encode a list of strings to a list of lists of token IDs.
        Args:
            texts: List of input strings.
        Returns:
            List of lists of token IDs.
        """
        return [self.encode(text) for text in texts]

    def batch_decode(self, ids_list):
        """
        Decode a list of lists of token IDs to a list of strings.
        Args:
            ids_list: List of lists of token IDs.
        Returns:
            List of decoded strings.
        """
        return [self.decode(ids) for ids in ids_list]

    def get_vocab_size(self):
        """
        Get the vocabulary size.
        Returns:
            Vocabulary size as an integer.
        """
        return self.sp_model.get_piece_size()

    def pad_id(self):
        """
        Get the padding token ID.
        Returns:
            Padding token ID.
        """
        return self.sp_model.pad_id()

    def unk_id(self):
        """
        Get the unknown token ID.
        Returns:
            Unknown token ID.
        """
        return self.sp_model.unk_id()

    def bos_id(self):
        """
        Get the beginning of sentence token ID.
        Returns:
            Beginning of sentence token ID.
        """
        return self.sp_model.bos_id()

    def eos_id(self):
        """
        Get the end of sentence token ID.
        Returns:
            End of sentence token ID.
        """
        return self.sp_model.eos_id()