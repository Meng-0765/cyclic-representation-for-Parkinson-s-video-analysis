# -*- coding: utf-8 -*-
"""Traditional one-dimensional motion signals for the five actions."""

import numpy as np
from scipy.spatial import ConvexHull


def extract_feet_features(data: np.ndarray, filename: str) -> np.ndarray:
    if "left" in filename.lower():
        p1, p13 = data[:, 0, :], data[:, 12, :]
        signal = np.linalg.norm(p13 - p1, axis=1)
    elif "right" in filename.lower():
        p1, p10 = data[:, 0, :], data[:, 9, :]
        signal = np.linalg.norm(p10 - p1, axis=1)
    else:
        raise ValueError("Filename must contain 'left' or 'right'.")
    return signal.reshape(-1, 1)


def extract_finger_features(data: np.ndarray) -> np.ndarray:
    p4, p8 = data[:, 4, :], data[:, 8, :]
    signal = np.linalg.norm(p8 - p4, axis=1)
    return signal.reshape(-1, 1)


def extract_palm_features(data: np.ndarray) -> np.ndarray:
    selected = data[:, [0, 8, 12, 16, 20], :]
    areas = []
    for frame in selected:
        try:
            areas.append(ConvexHull(frame).volume)
        except Exception:
            areas.append(0.0)
    return np.asarray(areas, dtype=float).reshape(-1, 1)


def extract_forearm_features(data: np.ndarray) -> np.ndarray:
    vector = data[:, 20, :] - data[:, 4, :]
    angles = np.arctan2(vector[:, 1], vector[:, 0])
    angular_velocity = np.diff(angles, prepend=angles[0])
    return angular_velocity.reshape(-1, 1)


def extract_toe_features(data: np.ndarray, filename: str) -> np.ndarray:
    if "left" in filename.lower():
        signal = data[:, 22, 1] - data[:, 0, 1]
    elif "right" in filename.lower():
        signal = data[:, 19, 1] - data[:, 0, 1]
    else:
        raise ValueError("Filename must contain 'left' or 'right'.")
    return signal.reshape(-1, 1)


def extract_signal(category: str, data: np.ndarray, filename: str) -> np.ndarray:
    extractors = {
        "data_feet": lambda: extract_feet_features(data, filename),
        "data_finger": lambda: extract_finger_features(data),
        "data_palm": lambda: extract_palm_features(data),
        "data_forearm": lambda: extract_forearm_features(data),
        "data_toe": lambda: extract_toe_features(data, filename),
    }
    try:
        return extractors[category]().ravel()
    except KeyError as exc:
        raise ValueError(f"Unknown category: {category}") from exc
