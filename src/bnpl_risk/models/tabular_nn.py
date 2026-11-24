"""ResNet-style tabular network (Gorishniy et al., 2021) with entity embeddings,
exposed as a scikit-learn classifier so it is tuned, cross-validated and calibrated by
exactly the same code as the other families.

Leak-free early stopping: the validation slice is carved out of whatever data ``fit``
receives (a training fold in CV), never from the evaluation fold or the test set.
"""

from __future__ import annotations

import copy
import io

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from bnpl_risk.config import CATEGORICAL_COLS, CATEGORY_LEVELS

DEFAULT_CARDINALITIES = tuple(len(CATEGORY_LEVELS[c]) for c in CATEGORICAL_COLS)


class _Block(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.f = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, d),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.f(x)


class _TabularResNet(nn.Module):
    def __init__(self, n_num, cardinalities, emb_dim, d, n_blocks, dropout, n_classes):
        super().__init__()
        # Index 0 of every embedding table is the reserved "unknown" category. It is
        # trained through category dropout, so unseen levels map to a learned vector
        # rather than to a random initialisation.
        self.embeddings = nn.ModuleList(nn.Embedding(c + 1, emb_dim) for c in cardinalities)
        self.proj = nn.Linear(n_num + emb_dim * len(cardinalities), d)
        self.blocks = nn.Sequential(*(_Block(d, dropout) for _ in range(n_blocks)))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.GELU(), nn.Linear(d, n_classes))

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.head(self.blocks(self.proj(torch.cat([x_num, *embs], dim=1))))


class TabularResNetClassifier(ClassifierMixin, BaseEstimator):
    """Expects the categorical ordinal codes (``-1`` = unknown) as the last
    ``len(cat_cardinalities)`` columns, which ``make_encoder("resnet")`` guarantees."""

    def __init__(
        self,
        d_block: int = 128,
        n_blocks: int = 2,
        emb_dim: int = 4,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 150,
        patience: int = 15,
        cat_dropout: float = 0.02,
        class_weight: str | None = None,
        val_fraction: float = 0.1,
        cat_cardinalities: tuple[int, ...] = DEFAULT_CARDINALITIES,
        random_state: int = 42,
        n_threads: int = 1,
    ):
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.cat_dropout = cat_dropout
        self.class_weight = class_weight
        self.val_fraction = val_fraction
        self.cat_cardinalities = cat_cardinalities
        self.random_state = random_state
        self.n_threads = n_threads

    # ------------------------------------------------------------------ helpers
    def _split(self, X) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(X, pd.DataFrame):
            tail = list(X.columns[-len(self.cat_cardinalities) :])
            if tail != CATEGORICAL_COLS:
                raise ValueError(f"Categorical columns must come last; got {tail}")
            X = X.to_numpy()
        X = np.asarray(X, dtype=np.float64)
        n_cat = len(self.cat_cardinalities)
        x_num = torch.tensor(X[:, :-n_cat], dtype=torch.float32)
        x_cat = torch.tensor(X[:, -n_cat:].astype(np.int64) + 1, dtype=torch.long)
        return x_num, x_cat.clamp_(min=0)

    def _build(self, n_num: int) -> _TabularResNet:
        return _TabularResNet(
            n_num,
            self.cat_cardinalities,
            self.emb_dim,
            self.d_block,
            self.n_blocks,
            self.dropout,
            len(self.classes_),
        )

    # ---------------------------------------------------------------------- API
    def fit(self, X, y):
        prev_threads = torch.get_num_threads()
        prev_det = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(self.n_threads)
        torch.use_deterministic_algorithms(True)
        try:
            return self._fit(X, np.asarray(y))
        finally:
            torch.set_num_threads(prev_threads)
            torch.use_deterministic_algorithms(prev_det)

    def _fit(self, X, y):
        self.classes_ = np.unique(y)
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        x_num, x_cat = self._split(X)
        y_t = torch.tensor(np.searchsorted(self.classes_, y), dtype=torch.long)

        tr, va = train_test_split(
            np.arange(len(y)),
            test_size=self.val_fraction,
            stratify=y,
            random_state=self.random_state,
        )
        tr, va = torch.tensor(tr), torch.tensor(va)

        torch.manual_seed(self.random_state)
        gen = torch.Generator().manual_seed(self.random_state)
        model = self._build(x_num.shape[1])
        weight = None
        if self.class_weight:
            w = compute_class_weight(self.class_weight, classes=self.classes_, y=y)
            weight = torch.tensor(w, dtype=torch.float32)
        train_loss = nn.CrossEntropyLoss(weight=weight)
        val_loss = nn.CrossEntropyLoss()  # selection always on the unweighted log loss
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)

        best, best_state, stale = np.inf, None, 0
        self.history_ = []
        for epoch in range(self.max_epochs):
            model.train()
            perm = tr[torch.randperm(len(tr), generator=gen)]
            for start in range(0, len(perm), self.batch_size):
                idx = perm[start : start + self.batch_size]
                xc = x_cat[idx].clone()
                if self.cat_dropout > 0:
                    drop = torch.rand(xc.shape, generator=gen) < self.cat_dropout
                    xc[drop] = 0
                opt.zero_grad()
                loss = train_loss(model(x_num[idx], xc), y_t[idx])
                loss.backward()
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                v = val_loss(model(x_num[va], x_cat[va]), y_t[va]).item()
            self.history_.append(v)
            if v < best - 1e-5:
                best, best_state, stale = v, copy.deepcopy(model.state_dict()), 0
                self.best_epoch_ = epoch + 1
            else:
                stale += 1
                if stale >= self.patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.n_num_ = x_num.shape[1]
        return self

    def predict_proba(self, X) -> np.ndarray:
        x_num, x_cat = self._split(X)
        with torch.no_grad():
            logits = self.model_(x_num, x_cat)
            return torch.softmax(logits.double(), dim=1).numpy()

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    # Pickle the weights as bytes so artifacts do not depend on torch's module pickling.
    def __getstate__(self):
        state = self.__dict__.copy()
        if "model_" in state:
            buf = io.BytesIO()
            torch.save(state.pop("model_").state_dict(), buf)
            state["_model_bytes"] = buf.getvalue()
        return state

    def __setstate__(self, state):
        blob = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        if blob is not None:
            model = self._build(self.n_num_)
            model.load_state_dict(torch.load(io.BytesIO(blob), weights_only=True))
            model.eval()
            self.model_ = model
