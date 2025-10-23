import torch
import torch.nn as nn
import numpy as np
import re
from typing import Iterable, List, Optional, Union
from torch.utils.data import DataLoader, TensorDataset


class SarcasmClassifier(nn.Module):
    """
    Simple MLP sarcasm classifier with Bag-of-Words vectorization.
    Used for inference in Flask API (expects pre-trained weights).
    """

    def __init__(
        self,
        vocab: Iterable[str],
        num_classes: int,
        lowercase: bool = True,
        token_pattern: str = r"\b\w+\b",
    ):
        super().__init__()

        # Vocabulary setup
        vocab_list = sorted(list(dict.fromkeys(vocab)))
        self.stoi = {tok: i for i, tok in enumerate(vocab_list)}
        self.itos = vocab_list
        self.vocab_size = len(self.itos)

        # Tokenization setup
        self.lowercase = lowercase
        self.tok_re = re.compile(token_pattern)
        self.num_classes = num_classes

        # Define the neural network layers (same structure as trained)
        act = nn.ReLU
        fc1_size = (self.vocab_size, 256)
        fc2_size = (256, num_classes)
        drop_out = 0.3

        layers = [
            nn.Linear(*fc1_size),
            nn.BatchNorm1d(fc1_size[1]),
            act(),
            nn.Dropout(drop_out),
            nn.Linear(*fc2_size),
        ]
        self.network = nn.Sequential(*layers)

    # ------------------------------
    # Text vectorization utilities
    # ------------------------------
    def _tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return self.tok_re.findall(text)

    def _vectorize_one(self, tokens: List[str]) -> np.ndarray:
        v = np.zeros(self.vocab_size, dtype=np.float32)
        for t in tokens:
            idx = self.stoi.get(t)
            if idx is not None:
                v[idx] += 1.0
        return v

    def transform(self, texts: List[str]) -> np.ndarray:
        """Vectorize raw text inputs into BoW format."""
        X = np.zeros((len(texts), self.vocab_size), dtype=np.float32)
        for i, s in enumerate(texts):
            tokens = self._tokenize(s)
            X[i] = self._vectorize_one(tokens)
        return X

    # ------------------------------
    # Forward + Predict
    # ------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @torch.no_grad()
    def predict(
        self, X_texts: List[str], batch_size: int = 64, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Predict class indices for given texts."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()

        X = self.transform(X_texts)
        X_t = torch.from_numpy(X.astype(np.float32)).to(device)
        logits = self(X_t)
        preds = torch.argmax(logits, dim=-1).cpu()
        return preds
