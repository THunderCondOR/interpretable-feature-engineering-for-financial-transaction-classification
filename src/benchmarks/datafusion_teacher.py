"""Dependency-light reproduction of the official Data Fusion 2023 RNN.

The published model calls ``torch.nn.functional.dropout`` without passing the
module's training flag.  Consequently its final pooled representation remains
stochastic after ``model.eval()``.  We intentionally preserve that behavior
and expose the inference seed rather than silently changing the baseline.
"""

from __future__ import annotations

import pickle
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


EMBEDDING_PROJECTIONS = {
    "hour": (26, 12), "mcc_code": (403, 150), "currency_rk": (5, 3),
    "transaction_amt": (103, 50), "day": (9, 4), "month": (14, 6),
    "number_day": (33, 15),
}


class OfficialTransactionsRnn(nn.Module):
    def __init__(self, rnn_units: int = 128, top_classifier_units: int = 64):
        super().__init__()
        self._transaction_cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality + 1, width, padding_idx=0)
            for cardinality, width in EMBEDDING_PROJECTIONS.values()
        ])
        self._spatial_dropout = nn.Dropout2d(0.5)
        embedded = sum(width for _, width in EMBEDDING_PROJECTIONS.values())
        self._gru = nn.GRU(
            input_size=embedded, hidden_size=rnn_units,
            batch_first=True, bidirectional=True,
        )
        self._hidden_size = rnn_units
        self._top_classifier = nn.Sequential(
            nn.Linear(rnn_units * 2 * 3, top_classifier_units), nn.ReLU(),
            nn.Linear(top_classifier_units, 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        embeddings = [
            embedding(values[:, index])
            for index, embedding in enumerate(self._transaction_cat_embeddings)
        ]
        combined = torch.cat(embeddings, dim=-1).permute(0, 2, 1).unsqueeze(3)
        combined = self._spatial_dropout(combined).squeeze(3).permute(0, 2, 1)
        states, hidden = self._gru(combined)
        max_pool = states.max(dim=1)[0]
        average_pool = states.sum(dim=1) / states.shape[1]
        hidden = hidden.permute(1, 2, 0).reshape(len(values), self._hidden_size * 2)
        pooled = torch.cat([max_pool, average_pool, hidden], dim=-1)
        # This is deliberate: it matches the published model.py exactly.
        pooled = nn.functional.dropout(pooled, p=0.5, training=True)
        return nn.functional.softmax(self._top_classifier(pooled), dim=1)


def extract_official_artifacts(model_zip: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    members = {
        "bins": "model/nn_bins.pickle",
        "weights": "model/nn_weights.ckpt",
        "source": "model/model.py",
    }
    paths = {key: output_dir / Path(member).name for key, member in members.items()}
    with zipfile.ZipFile(model_zip) as archive:
        for key, member in members.items():
            paths[key].write_bytes(archive.read(member))
    return paths


def encode_transactions(events: pd.DataFrame, bins_path: Path) -> tuple[list[str], np.ndarray]:
    frame = events.copy()
    frame["tr_datetime"] = pd.to_datetime(frame["tr_datetime"], errors="raise")
    frame = frame.dropna(subset=["mcc_code", "currency_rk", "amount", "tr_datetime"])
    frame = frame.assign(
        hour=frame["tr_datetime"].dt.hour,
        day=frame["tr_datetime"].dt.dayofweek,
        month=frame["tr_datetime"].dt.month,
        number_day=frame["tr_datetime"].dt.day,
        transaction_amt=frame["amount"],
    )
    with bins_path.open("rb") as stream:
        bins = pickle.load(stream)
    features = list(bins["features"])
    for column in features:
        frame[column] = pd.cut(
            frame[column] if column == "transaction_amt" else frame[column].astype(float).astype(int),
            bins=bins[column], labels=False,
        ).astype(int)
    users, arrays = [], []
    for customer_id, group in frame.groupby("customer_id", sort=True, observed=True):
        matrix = group[features].to_numpy().T[:, -300:]
        if matrix.shape[1] < 300:
            matrix = np.pad(matrix, ((0, 0), (0, 300 - matrix.shape[1])))
        users.append(str(customer_id))
        arrays.append(matrix)
    return users, np.asarray(arrays, dtype=np.int64)


def predict_official(
    events: pd.DataFrame,
    *,
    bins_path: Path,
    weights_path: Path,
    seed: int,
    batch_size: int = 128,
    device: str | None = None,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    users, encoded = encode_transactions(events, bins_path)
    model = OfficialTransactionsRnn()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(selected_device).eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(encoded)), batch_size=batch_size, shuffle=False)
    probabilities = []
    with torch.inference_mode():
        for (batch,) in loader:
            probabilities.append(model(batch.to(selected_device)).cpu().numpy())
    values = np.concatenate(probabilities)
    return pd.DataFrame({
        "customer_id": users,
        "teacher_prob_0": values[:, 0],
        "teacher_prob_1": values[:, 1],
        "teacher_seed": int(seed),
    })
