import numpy as np


IFOREST_CONTAMINATION = 0.08
LSTM_NORMAL_FILTER_PERCENTILE = 10
HYBRID_THRESHOLD_PERCENTILE = 92


def compute_hybrid_score(iforest_norm, lstm_norm, has_temporal_signal):
    if_values = np.asarray(iforest_norm, dtype=float)
    lstm_values = np.asarray(lstm_norm, dtype=float)
    temporal_mask = np.asarray(has_temporal_signal, dtype=bool)

    peak_signal = np.maximum(if_values, lstm_values)
    agreement_bonus = 0.10 * np.minimum(
        np.maximum(if_values, 0.0),
        np.maximum(lstm_values, 0.0),
    )

    temporal_score = (
        (0.60 * if_values)
        + (0.20 * lstm_values)
        + (0.20 * peak_signal)
        + agreement_bonus
    )

    return np.where(temporal_mask, temporal_score, if_values)
