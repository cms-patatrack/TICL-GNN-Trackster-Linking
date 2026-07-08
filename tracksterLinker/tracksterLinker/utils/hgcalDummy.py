import json
import math
import os
import os.path as osp
from dataclasses import asdict, dataclass
from typing import Dict, Sequence

import awkward as ak
import numpy as np


FEATURE_KEYS = [
    "barycenter_x",
    "barycenter_y",
    "barycenter_z",
    "barycenter_eta",
    "barycenter_phi",
    "eVector0_x",
    "eVector0_y",
    "eVector0_z",
    "EV1",
    "EV2",
    "EV3",
    "sigmaPCA1",
    "sigmaPCA2",
    "sigmaPCA3",
    "num_LCs",
    "num_hits",
    "raw_energy",
    "raw_em_energy",
    "photon_prob",
    "electron_prob",
    "muon_prob",
    "neutral_pion_prob",
    "charged_hadron_prob",
    "neutral_hadron_prob",
    "z_min",
    "z_max",
    "LC_density",
    "trackster_density",
    "time",
]

HGCAL_DENSITY_VOLUME = 2 * (3.0 - 1.5) * (2 * 47)
HGCAL_Z_MIN_CM = 320.0
HGCAL_Z_MAX_CM = 520.0

SHOWER_STYLES = {
    "straight": {
        "fragment_scale": 0.58,
        "width": 0.35,
        "direction_noise": 45.0,
        "random_direction": 0.04,
        "ev": (32.0, 0.35, 0.035),
        "sigma": (4.7, 0.85, 0.70),
        "lc_scale": 1.7,
        "hit_range": (3.2, 6.0),
    },
    "curved": {
        "fragment_scale": 0.92,
        "width": 0.80,
        "direction_noise": 70.0,
        "random_direction": 0.07,
        "ev": (42.0, 0.65, 0.075),
        "sigma": (5.0, 1.00, 0.85),
        "lc_scale": 2.1,
        "hit_range": (3.8, 7.2),
    },
    "broad_tree": {
        "fragment_scale": 1.25,
        "width": 1.75,
        "direction_noise": 105.0,
        "random_direction": 0.13,
        "ev": (58.0, 1.05, 0.16),
        "sigma": (5.4, 1.20, 1.00),
        "lc_scale": 2.7,
        "hit_range": (4.2, 8.5),
    },
    "large_shower": {
        "fragment_scale": 1.60,
        "width": 1.25,
        "direction_noise": 90.0,
        "random_direction": 0.10,
        "ev": (82.0, 1.45, 0.28),
        "sigma": (6.0, 1.35, 1.10),
        "lc_scale": 3.3,
        "hit_range": (5.0, 11.0),
    },
    "multi_shower_tree": {
        "fragment_scale": 1.38,
        "width": 1.45,
        "direction_noise": 115.0,
        "random_direction": 0.16,
        "ev": (66.0, 1.25, 0.22),
        "sigma": (5.6, 1.25, 1.05),
        "lc_scale": 2.8,
        "hit_range": (4.5, 9.5),
    },
}

@dataclass
class HGCALLikeDummyConfig:
    train_files: int = 80
    val_files: int = 20
    test_files: int = 20
    events_per_file: int = 10
    signal_mean: float = 30.0
    pu_mean: float = 200.0
    close_pair_fraction: float = 0.85
    seed: int = 12345


def _wrap_phi(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


def _normalised(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _random_unit_vector(rng):
    return _normalised(rng.normal(0.0, 1.0, size=3))


def _eta_phi_z_to_xyz(eta, phi, z_abs, z_sign):
    theta = 2.0 * math.atan(math.exp(-abs(eta)))
    radius = z_abs * math.tan(theta)
    return np.array(
        [
            radius * math.cos(phi),
            radius * math.sin(phi),
            z_sign * z_abs,
        ],
        dtype=np.float64,
    )


def _particle_probabilities(pdg_id, rng):
    base = np.full(6, 0.025)
    abs_pdg = abs(int(pdg_id))
    if abs_pdg == 22:
        base[0] = 0.85
    elif abs_pdg == 11:
        base[1] = 0.82
    elif abs_pdg == 13:
        base[2] = 0.82
    elif abs_pdg == 111:
        base[3] = 0.78
    elif abs_pdg in {211, 321, 2212, 15}:
        base[4] = 0.80
    elif abs_pdg in {130, 2112}:
        base[5] = 0.80
    else:
        base[5] = 0.65

    noisy = np.clip(base + rng.normal(0.0, 0.035, size=6), 0.001, None)
    return noisy / noisy.sum()


def _sample_multiparticle_pdg(rng):
    return int(
        rng.choice(
            [22, 11, -11, 13, -13, 111, 211, -211, 130, 2112, 321, -321],
            p=[0.15, 0.055, 0.055, 0.025, 0.025, 0.10, 0.32, 0.10, 0.075, 0.045, 0.025, 0.025],
        )
    )


def _sample_pu_pdg(rng):
    return int(
        rng.choice(
            [22, 13, -13, 111, 211, -211, 130, 2112, 321, -321],
            p=[0.16, 0.015, 0.015, 0.12, 0.33, 0.15, 0.09, 0.06, 0.03, 0.03],
        )
    )


def _sample_shower_style(rng, pdg_id, is_pu):
    abs_pdg = abs(int(pdg_id))
    if is_pu:
        styles = ["straight", "curved", "broad_tree", "large_shower", "multi_shower_tree"]
        weights = np.array([0.22, 0.20, 0.24, 0.12, 0.22])
    elif abs_pdg == 13:
        styles = ["straight", "curved", "broad_tree"]
        weights = np.array([0.78, 0.17, 0.05])
    elif abs_pdg in {22, 11}:
        styles = ["straight", "curved", "large_shower", "multi_shower_tree"]
        weights = np.array([0.46, 0.18, 0.26, 0.10])
    elif abs_pdg == 111:
        styles = ["large_shower", "multi_shower_tree", "broad_tree", "curved"]
        weights = np.array([0.36, 0.32, 0.22, 0.10])
    elif abs_pdg in {130, 2112}:
        styles = ["broad_tree", "multi_shower_tree", "large_shower", "curved"]
        weights = np.array([0.36, 0.30, 0.24, 0.10])
    else:
        styles = ["curved", "broad_tree", "multi_shower_tree", "straight", "large_shower"]
        weights = np.array([0.32, 0.28, 0.24, 0.10, 0.06])

    weights = weights / weights.sum()
    return str(rng.choice(styles, p=weights))


def _sample_axis(rng, z_sign=None, eta_range=(1.5, 3.0)):
    if z_sign is None:
        z_sign = int(rng.choice([-1, 1]))
    eta_abs = rng.uniform(*eta_range)
    return {
        "eta": z_sign * eta_abs,
        "eta_abs": eta_abs,
        "phi": rng.uniform(-np.pi, np.pi),
        "z_sign": z_sign,
    }


def _decorate_axis(axis, rng, is_pu=False, crowding=1.0):
    axis = dict(axis)
    drift_scale = (0.018 if is_pu else 0.030) * crowding
    axis["eta_drift"] = float(rng.normal(0.0, drift_scale))
    axis["phi_drift"] = float(rng.normal(0.0, drift_scale))
    axis["scatter_scale"] = float(rng.lognormal(mean=0.0 if not is_pu else 0.18, sigma=0.32))
    axis["core_jitter"] = float(rng.uniform(0.001, 0.012 if not is_pu else 0.020))
    axis["time0"] = float(rng.normal(0.0, 0.08 if not is_pu else 4.5))
    return axis


def _near_axis(axis, rng, eta_sigma=0.025, phi_sigma=0.025, is_pu=False, crowding=1.0):
    eta_abs = np.clip(abs(axis["eta"]) + rng.normal(0.0, eta_sigma), 1.5, 3.0)
    near = {
        "eta": axis["z_sign"] * eta_abs,
        "eta_abs": eta_abs,
        "phi": float(_wrap_phi(axis["phi"] + rng.normal(0.0, phi_sigma))),
        "z_sign": axis["z_sign"],
    }
    return _decorate_axis(near, rng, is_pu=is_pu, crowding=crowding)


def _sample_energy(rng, axis, is_pu=False):
    if is_pu:
        # PU is numerous and mostly soft, but the high-eta cosh factor still creates
        # occasional energetic contaminants in the same HGCAL volume.
        pt = rng.gamma(shape=1.6, scale=2.2)
        if rng.random() < 0.05:
            pt += rng.exponential(12.0)
        return float(np.clip(pt * math.cosh(axis["eta_abs"]), 0.4, 180.0))

    if rng.random() < 0.35:
        pt = rng.uniform(20.0, 120.0)
        return float(np.clip(pt * math.cosh(axis["eta_abs"]), 20.0, 800.0))
    return float(np.clip(rng.lognormal(mean=4.6, sigma=0.75), 8.0, 800.0))


def _fragment_count(rng, energy, pdg_id, is_pu, style_name):
    abs_pdg = abs(int(pdg_id))
    style = SHOWER_STYLES[style_name]
    bounded_energy = min(float(energy), 500.0)

    if abs_pdg == 13:
        mean = 2.0 + 0.010 * bounded_energy
    elif abs_pdg in {22, 11}:
        mean = 3.4 + 0.020 * bounded_energy
    elif abs_pdg in {211, 130, 321, 2112, 15}:
        mean = 5.4 + 0.034 * bounded_energy
    else:
        mean = 4.4 + 0.026 * bounded_energy

    mean *= style["fragment_scale"]

    if is_pu:
        mean = 4.0 + 0.85 * mean

    if rng.random() < (0.10 if is_pu else 0.18):
        mean *= rng.uniform(1.25, 2.10)

    return int(np.clip(1 + rng.poisson(mean), 1, 34 if is_pu else 46))


def _energy_fractions(rng, n_fragments, style_name):
    if n_fragments == 1:
        return np.ones(1)

    depth_order = np.linspace(0.0, 1.0, n_fragments)
    if style_name == "straight":
        profile = 1.15 * np.exp(-2.2 * depth_order) + 0.08
    elif style_name == "curved":
        phase = rng.uniform(0.0, 2 * np.pi)
        profile = np.exp(-1.45 * depth_order) * (1.0 + 0.22 * np.cos(2 * np.pi * depth_order + phase)) + 0.08
    elif style_name == "broad_tree":
        profile = 0.85 * np.exp(-0.85 * depth_order) + 0.08
        for center in rng.uniform(0.12, 0.92, size=int(rng.integers(1, 4))):
            profile += rng.uniform(0.25, 0.80) * np.exp(-0.5 * ((depth_order - center) / rng.uniform(0.035, 0.12)) ** 2)
    elif style_name == "large_shower":
        profile = (depth_order + 0.08) ** 1.1 * np.exp(-2.4 * depth_order) + 0.12
    elif style_name == "multi_shower_tree":
        profile = 0.35 * np.exp(-1.2 * depth_order) + 0.05
        for center in np.sort(rng.uniform(0.10, 0.88, size=int(rng.integers(2, 5)))):
            profile += rng.uniform(0.35, 0.95) * np.exp(-0.5 * ((depth_order - center) / rng.uniform(0.035, 0.10)) ** 2)
    else:
        profile = np.exp(-1.1 * depth_order) + 0.08

    profile *= rng.lognormal(mean=0.0, sigma=0.38, size=n_fragments)
    profile = np.clip(profile, 0.001, None)
    alpha = 0.18 + 3.2 * profile / np.mean(profile)
    return rng.dirichlet(alpha)


def _shower_depths(rng, n_fragments, pdg_id, style_name):
    abs_pdg = abs(int(pdg_id))
    if abs_pdg in {22, 11, 13}:
        z_stop = rng.uniform(360.0, 430.0)
        z_start = rng.uniform(HGCAL_Z_MIN_CM, 345.0)
    else:
        z_stop = rng.uniform(430.0, HGCAL_Z_MAX_CM)
        z_start = rng.uniform(HGCAL_Z_MIN_CM, 375.0)

    if style_name == "straight":
        quantiles = np.linspace(0.03, 0.98, n_fragments) + rng.normal(0.0, 0.012, size=n_fragments)
        depths = z_start + np.clip(quantiles, 0.0, 1.0) * (z_stop - z_start)
    elif style_name == "curved":
        quantiles = np.sort(rng.beta(1.15, 1.35, size=n_fragments))
        depths = z_start + quantiles * (z_stop - z_start)
    elif style_name == "large_shower":
        z_stop = max(z_stop, rng.uniform(470.0, HGCAL_Z_MAX_CM))
        quantiles = np.sort(rng.beta(1.55, 1.25, size=n_fragments))
        depths = z_start + quantiles * (z_stop - z_start)
    elif style_name == "multi_shower_tree" and n_fragments > 3:
        n_cores = min(n_fragments, int(rng.integers(2, 5)))
        shower_span = max(z_stop - z_start, 1.0)
        core_margin = min(10.0, 0.20 * shower_span)
        core_low = z_start + core_margin
        core_high = z_stop - core_margin
        if core_high <= core_low:
            core_low, core_high = z_start, z_stop
        core_centers = np.sort(rng.uniform(core_low, core_high, size=n_cores))
        core_weights = rng.dirichlet(np.full(n_cores, 0.8))
        assignments = rng.choice(np.arange(n_cores), size=n_fragments, p=core_weights)
        depths = core_centers[assignments] + rng.normal(0.0, rng.uniform(5.0, 14.0), size=n_fragments)
    elif style_name == "broad_tree" and n_fragments > 3:
        split = int(rng.integers(1, n_fragments))
        early = rng.uniform(z_start, min(z_stop, z_start + 85.0), size=split)
        late = rng.uniform(max(z_start, z_stop - 85.0), z_stop, size=n_fragments - split)
        depths = np.concatenate([early, late])
    else:
        quantiles = np.sort(rng.beta(1.3, 1.4, size=n_fragments))
        depths = z_start + quantiles * (z_stop - z_start)

    jitter = 5.5 if style_name == "straight" else 8.0 if style_name == "curved" else 13.0
    depths += rng.normal(0.0, jitter if abs_pdg in {22, 11, 13} else jitter + 4.0, size=n_fragments)
    return np.sort(np.clip(depths, HGCAL_Z_MIN_CM, HGCAL_Z_MAX_CM))


def _branch_offsets(rng, n_fragments, style_name, base_spread, scatter_scale):
    if n_fragments <= 1:
        return np.zeros((n_fragments, 2))

    depth_order = np.linspace(0.0, 1.0, n_fragments)
    offsets = np.zeros((n_fragments, 2))
    base = max(base_spread * scatter_scale, 1e-4)

    if style_name == "straight":
        offsets = np.cumsum(rng.normal(0.0, base * 0.08, size=(n_fragments, 2)), axis=0)
    elif style_name == "curved":
        curve = rng.normal(0.0, base * rng.uniform(5.0, 9.0), size=2)
        offsets = curve * depth_order[:, None] ** 2
        offsets += rng.normal(0.0, base * 0.28, size=(n_fragments, 2))
    elif style_name == "large_shower":
        offsets = rng.normal(0.0, base * (1.2 + 2.4 * depth_order[:, None]), size=(n_fragments, 2))
    elif style_name == "multi_shower_tree":
        n_cores = min(n_fragments, int(rng.integers(2, 5)))
        core_offsets = rng.normal(0.0, base * rng.uniform(3.5, 7.5), size=(n_cores, 2))
        assignments = np.floor(depth_order * n_cores).astype(int)
        assignments = np.clip(assignments + rng.integers(-1, 2, size=n_fragments), 0, n_cores - 1)
        offsets = core_offsets[assignments] + rng.normal(0.0, base * (0.55 + 1.25 * depth_order[:, None]), size=(n_fragments, 2))
        for idx in range(1, n_fragments):
            offsets[idx] += 0.28 * offsets[idx - 1]
    else:
        for idx in range(1, n_fragments):
            parent = int(rng.integers(max(0, idx - 6), idx))
            kick = rng.normal(0.0, base * (0.9 + 2.8 * depth_order[idx]), size=2)
            if rng.random() < 0.18:
                kick += rng.normal(0.0, base * rng.uniform(4.0, 8.0), size=2)
            offsets[idx] = 0.82 * offsets[parent] + kick

    return offsets


def _sample_evector(rng, position, style_name, is_pu):
    style = SHOWER_STYLES[style_name]
    random_fraction = style["random_direction"] + (0.04 if is_pu else 0.0)
    if rng.random() < random_fraction:
        return _random_unit_vector(rng)

    noise_scale = style["direction_noise"] * (1.35 if is_pu else 1.0)
    return _normalised(position + rng.normal(0.0, noise_scale, size=3))


def _sample_shape_features(rng, raw_energy, style_name, is_pu):
    style = SHOWER_STYLES[style_name]
    energy_scale = np.clip(raw_energy / 4.0, 0.45, 7.0) ** 0.10
    ev_medians = np.asarray(style["ev"], dtype=float) * energy_scale
    if is_pu:
        ev_medians *= rng.lognormal(mean=-0.04, sigma=0.18, size=3)

    ev1, ev2, ev3 = rng.lognormal(np.log(ev_medians), sigma=[0.62, 0.72, 0.88])
    ev2 = max(ev2, ev3 * 1.08)
    ev1 = max(ev1, ev2 * 1.8)

    sigma_medians = np.asarray(style["sigma"], dtype=float) * np.clip(raw_energy / 4.0, 0.70, 5.0) ** 0.05
    sigma1, sigma2, sigma3 = rng.lognormal(np.log(sigma_medians), sigma=[0.32, 0.30, 0.34])
    return ev1, ev2, ev3, sigma1, sigma2, sigma3


def _sample_time(rng, depth_frac, raw_energy, style_name, is_pu, time0):
    invalid_prob = 0.60 if is_pu else 0.36
    if raw_energy < 1.0:
        invalid_prob += 0.14
    if style_name in {"broad_tree", "multi_shower_tree", "large_shower"}:
        invalid_prob += 0.05
    if style_name == "straight":
        invalid_prob -= 0.10

    if rng.random() < np.clip(invalid_prob, 0.05, 0.86):
        return -99.0

    if is_pu:
        return float(time0 + rng.normal(0.0, 5.8) + depth_frac * rng.normal(0.0, 1.4))
    return float(time0 + rng.normal(0.0, 0.12) + depth_frac * rng.normal(0.04, 0.08))


def _make_tracksters_for_shower(rng, axis, energy, pdg_id, sim_id, is_pu):
    style_name = _sample_shower_style(rng, pdg_id, is_pu)
    n_fragments = _fragment_count(rng, energy, pdg_id, is_pu, style_name)
    fractions = _energy_fractions(rng, n_fragments, style_name)
    depths = _shower_depths(rng, n_fragments, pdg_id, style_name)
    pid = _particle_probabilities(pdg_id, rng)
    rows = []
    labels = []
    pu_flags = []

    abs_pdg = abs(int(pdg_id))
    style = SHOWER_STYLES[style_name]
    angular_spread = 0.0045 if abs_pdg in {22, 11, 13} else 0.016
    time0 = axis.get("time0", rng.normal(0.0, 0.08 if not is_pu else 4.5))
    scatter_scale = axis.get("scatter_scale", 1.0)
    core_jitter = axis.get("core_jitter", 0.004)
    branch_eta = rng.normal(0.0, angular_spread * scatter_scale * style["width"])
    branch_phi = rng.normal(0.0, angular_spread * scatter_scale * style["width"])
    offsets = _branch_offsets(rng, n_fragments, style_name, angular_spread, scatter_scale)

    for idx, (z_abs, frac) in enumerate(zip(depths, fractions)):
        depth_frac = (z_abs - HGCAL_Z_MIN_CM) / (HGCAL_Z_MAX_CM - HGCAL_Z_MIN_CM)
        width_growth = 0.7 + 1.8 * depth_frac
        heavy_tail_prob = 0.10 if style_name == "straight" else 0.34
        tail = rng.standard_t(df=3) if rng.random() < heavy_tail_prob else rng.normal()
        local_eta_spread = angular_spread * scatter_scale * width_growth * style["width"]
        local_phi_spread = angular_spread * scatter_scale * width_growth * style["width"]
        if abs_pdg not in {22, 11, 13} and rng.random() < 0.38:
            local_eta_spread *= rng.uniform(1.35, 3.4)
            local_phi_spread *= rng.uniform(1.35, 3.4)

        eta_center = (
            axis["eta"]
            + axis["z_sign"] * axis.get("eta_drift", 0.0) * depth_frac
            + branch_eta * depth_frac
            + offsets[idx, 0]
        )
        phi_center = _wrap_phi(
            axis["phi"]
            + axis.get("phi_drift", 0.0) * depth_frac
            + branch_phi * depth_frac
            + offsets[idx, 1]
        )

        eta = eta_center + axis["z_sign"] * (rng.normal(0.0, core_jitter) + tail * local_eta_spread)
        eta_abs = np.clip(abs(eta), 1.5, 3.0)
        eta = axis["z_sign"] * eta_abs
        phi = float(_wrap_phi(phi_center + rng.normal(0.0, core_jitter) + rng.normal(0.0, local_phi_spread)))
        position = _eta_phi_z_to_xyz(eta, phi, z_abs, axis["z_sign"])

        direction = _sample_evector(rng, position, style_name, is_pu)
        raw_energy = max(0.02, energy * frac * rng.lognormal(mean=0.0, sigma=0.18 if not is_pu else 0.35))
        em_fraction = rng.uniform(0.72, 0.95) if abs_pdg in {22, 11} else rng.beta(1.8, 3.6)
        raw_em_energy = raw_energy * em_fraction

        lc_mean = (5.8 if is_pu else 6.8) + style["lc_scale"] * math.sqrt(raw_energy)
        num_lcs = int(max(1 if is_pu else 2, rng.poisson(lc_mean)))
        hit_low, hit_high = style["hit_range"]
        num_hits = int(max(num_lcs, rng.poisson(num_lcs * rng.uniform(hit_low, hit_high))))
        ev1, ev2, ev3, sigma1, sigma2, sigma3 = _sample_shape_features(rng, raw_energy, style_name, is_pu)

        z_span = max(1.0, math.sqrt(ev1) * rng.uniform(0.55, 1.35))
        z_a = axis["z_sign"] * (z_abs - z_span / 2)
        z_b = axis["z_sign"] * (z_abs + z_span / 2)
        z_min = min(z_a, z_b)
        z_max = max(z_a, z_b)
        time = _sample_time(rng, depth_frac, raw_energy, style_name, is_pu, time0)

        rows.append(
            [
                position[0],
                position[1],
                position[2],
                eta,
                phi,
                direction[0],
                direction[1],
                direction[2],
                ev1,
                ev2,
                ev3,
                sigma1,
                sigma2,
                sigma3,
                num_lcs,
                num_hits,
                raw_energy,
                raw_em_energy,
                pid[0],
                pid[1],
                pid[2],
                pid[3],
                pid[4],
                pid[5],
                z_min,
                z_max,
                num_lcs / HGCAL_DENSITY_VOLUME,
                0.0,
                time,
            ]
        )
        labels.append(sim_id)
        pu_flags.append(int(is_pu))

    return rows, labels, pu_flags


def _make_activity_centers(rng):
    centers = []
    for _ in range(int(rng.integers(3, 7))):
        centers.append(_decorate_axis(_sample_axis(rng, eta_range=(1.65, 2.85)), rng, crowding=1.6))
    return centers


def _sample_axis_near_centers(rng, centers, is_pu=False, close_pair_fraction=0.85):
    if centers and rng.random() < close_pair_fraction:
        center = centers[int(rng.integers(0, len(centers)))]
        if is_pu and rng.random() < 0.14:
            eta_sigma = rng.uniform(0.012, 0.045)
            phi_sigma = rng.uniform(0.012, 0.045)
        else:
            eta_sigma = rng.uniform(0.050, 0.22 if is_pu else 0.12)
            phi_sigma = rng.uniform(0.050, 0.22 if is_pu else 0.12)
        return _near_axis(center, rng, eta_sigma=eta_sigma, phi_sigma=phi_sigma, is_pu=is_pu, crowding=1.8)
    return _decorate_axis(_sample_axis(rng), rng, is_pu=is_pu, crowding=1.2)


def _event_axes(rng, signal_mean, close_pair_fraction):
    centers = _make_activity_centers(rng)
    n_particles = max(8, int(rng.poisson(signal_mean)))
    axes = []
    pdgs = []
    for _ in range(n_particles):
        axis = _sample_axis_near_centers(rng, centers, is_pu=False, close_pair_fraction=close_pair_fraction)
        axes.append(axis)
        pdgs.append(_sample_multiparticle_pdg(rng))
    return axes, pdgs, [False] * len(axes), centers


def generate_event(rng, signal_mean=30.0, pu_mean=200.0, close_pair_fraction=0.85):
    axes, pdgs, pu_flags, centers = _event_axes(rng, signal_mean, close_pair_fraction)
    n_pu = int(rng.poisson(pu_mean))

    for _ in range(n_pu):
        axis = _sample_axis_near_centers(rng, centers + axes, is_pu=True, close_pair_fraction=0.62)
        axes.append(axis)
        pdgs.append(_sample_pu_pdg(rng))
        pu_flags.append(True)

    rows = []
    labels = []
    is_pu = []
    for sim_id, (axis, pdg_id, pu_flag) in enumerate(zip(axes, pdgs, pu_flags)):
        energy = _sample_energy(rng, axis, is_pu=pu_flag)
        shower_rows, shower_labels, shower_pu = _make_tracksters_for_shower(rng, axis, energy, pdg_id, sim_id, pu_flag)
        rows.extend(shower_rows)
        labels.extend(shower_labels)
        is_pu.extend(shower_pu)

    rows = np.asarray(rows, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    is_pu = np.asarray(is_pu, dtype=np.int64)
    if len(rows) == 0:
        raise RuntimeError("Generated an empty event")

    order = rng.permutation(len(rows))
    rows = rows[order]
    labels = labels[order]
    is_pu = is_pu[order]
    trackster_density = len(rows) / HGCAL_DENSITY_VOLUME
    rows[:, FEATURE_KEYS.index("trackster_density")] = trackster_density

    event = {name: rows[:, idx] for idx, name in enumerate(FEATURE_KEYS)}
    event["y"] = labels
    event["isPU"] = is_pu
    return event


def generate_events(n_events, rng, signal_mean=30.0, pu_mean=200.0, close_pair_fraction=0.85):
    return ak.Array(
        [
            generate_event(
                rng,
                signal_mean=signal_mean,
                pu_mean=pu_mean,
                close_pair_fraction=close_pair_fraction,
            )
            for _ in range(n_events)
        ]
    )


def write_dataset(output_dir, config: HGCALLikeDummyConfig):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    split_files = {
        "train": config.train_files,
        "val": config.val_files,
        "test": config.test_files,
    }

    for split, n_files in split_files.items():
        split_dir = osp.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for file_idx in range(n_files):
            events = generate_events(
                config.events_per_file,
                rng,
                signal_mean=config.signal_mean,
                pu_mean=config.pu_mean,
                close_pair_fraction=config.close_pair_fraction,
            )
            ak.to_parquet(events, osp.join(split_dir, f"dummy_{split}_{file_idx:05d}.parquet"))

    metadata = asdict(config)
    metadata["feature_keys"] = FEATURE_KEYS
    metadata["notes"] = [
        "Synthetic public dummy data; no CMS event content is copied.",
        "Ranges follow the thesis baseline: HGCAL eta 1.5-3.0, full phi, 47 layers/endcap density convention.",
        "Default topology is crowded multiparticle hard scatter with about 200 overlapping PU showers.",
        "Particles are sampled from five internal shower styles: straight, curved, broad tree, large shower, and multi-shower tree.",
        "Shower fragments include depth-dependent energy degradation, branch splitting, heavy-tailed angular scatter, broad PU timing, and invalid time markers.",
    ]
    with open(osp.join(output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def summarise_parquet_files(files: Sequence[str]) -> Dict[str, float]:
    n_events = 0
    n_tracksters = []
    n_pu = []
    n_signal_labels = []
    energies = []
    for file_name in files:
        data = ak.from_parquet(file_name)
        for event in data:
            n_events += 1
            n_tracksters.append(len(event["y"]))
            n_pu.append(int(np.asarray(event["isPU"]).sum()))
            labels = np.asarray(event["y"])[np.asarray(event["isPU"]) == 0]
            n_signal_labels.append(len(np.unique(labels)))
            energies.extend(np.asarray(event["raw_energy"], dtype=float).tolist())

    if n_events == 0:
        return {}

    return {
        "events": float(n_events),
        "tracksters_mean": float(np.mean(n_tracksters)),
        "tracksters_p95": float(np.percentile(n_tracksters, 95)),
        "pu_tracksters_mean": float(np.mean(n_pu)),
        "signal_simtracksters_mean": float(np.mean(n_signal_labels)),
        "energy_median": float(np.median(energies)),
        "energy_p95": float(np.percentile(energies, 95)),
    }
