import json
import math
import os
import os.path as osp
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence

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
    return int(rng.choice([22, 11, -11, 111, 211, -211, 130, 2112, 321, -321], p=[0.16, 0.06, 0.06, 0.10, 0.34, 0.10, 0.08, 0.05, 0.025, 0.025]))


def _sample_pu_pdg(rng):
    return int(rng.choice([22, 111, 211, -211, 130, 2112, 321, -321], p=[0.18, 0.12, 0.34, 0.16, 0.09, 0.06, 0.025, 0.025]))


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
    axis["time0"] = float(rng.normal(0.0, 0.035 if not is_pu else 0.22))
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


def _fragment_count(rng, energy, pdg_id, is_pu):
    abs_pdg = abs(int(pdg_id))
    if abs_pdg in {22, 11}:
        mean = 1.4 + 0.007 * energy
    elif abs_pdg in {211, 130, 321, 2112, 15}:
        mean = 3.0 + 0.014 * energy
    else:
        mean = 2.0 + 0.010 * energy

    if is_pu:
        mean = 0.8 + 0.35 * mean

    if rng.random() < (0.10 if is_pu else 0.18):
        mean *= rng.uniform(1.4, 2.4)

    return int(np.clip(1 + rng.poisson(mean), 1, 18 if is_pu else 32))


def _energy_fractions(rng, n_fragments, pdg_id):
    if n_fragments == 1:
        return np.ones(1)

    abs_pdg = abs(int(pdg_id))
    alpha = np.full(n_fragments, 0.45)
    alpha[0] = 4.0
    if abs_pdg in {211, 130, 321, 15} and n_fragments > 4:
        alpha[n_fragments // 2] = 2.2

    fractions = rng.dirichlet(alpha)
    rng.shuffle(fractions)
    return fractions


def _shower_depths(rng, n_fragments, pdg_id):
    abs_pdg = abs(int(pdg_id))
    if abs_pdg in {22, 11}:
        z_stop = rng.uniform(360.0, 430.0)
        z_start = rng.uniform(HGCAL_Z_MIN_CM, 345.0)
    else:
        z_stop = rng.uniform(430.0, HGCAL_Z_MAX_CM)
        z_start = rng.uniform(HGCAL_Z_MIN_CM, 375.0)

    if abs_pdg in {211, 130, 321, 2112, 15} and n_fragments > 3 and rng.random() < 0.35:
        split = int(rng.integers(1, n_fragments))
        early = rng.uniform(z_start, min(z_stop, z_start + 85.0), size=split)
        late = rng.uniform(max(z_start, z_stop - 85.0), z_stop, size=n_fragments - split)
        depths = np.concatenate([early, late])
    else:
        quantiles = np.sort(rng.beta(1.3, 1.4, size=n_fragments))
        depths = z_start + quantiles * (z_stop - z_start)

    depths += rng.normal(0.0, 7.5 if abs_pdg in {22, 11} else 12.0, size=n_fragments)
    return np.sort(np.clip(depths, HGCAL_Z_MIN_CM, HGCAL_Z_MAX_CM))


def _make_tracksters_for_shower(rng, axis, energy, pdg_id, sim_id, is_pu):
    n_fragments = _fragment_count(rng, energy, pdg_id, is_pu)
    fractions = _energy_fractions(rng, n_fragments, pdg_id)
    depths = _shower_depths(rng, n_fragments, pdg_id)
    pid = _particle_probabilities(pdg_id, rng)
    rows = []
    labels = []
    pu_flags = []

    abs_pdg = abs(int(pdg_id))
    angular_spread = 0.005 if abs_pdg in {22, 11} else 0.018
    time0 = axis.get("time0", rng.normal(0.0, 0.035 if not is_pu else 0.22))
    scatter_scale = axis.get("scatter_scale", 1.0)
    core_jitter = axis.get("core_jitter", 0.004)
    branch_eta = rng.normal(0.0, angular_spread * scatter_scale)
    branch_phi = rng.normal(0.0, angular_spread * scatter_scale)

    for idx, (z_abs, frac) in enumerate(zip(depths, fractions)):
        depth_frac = (z_abs - HGCAL_Z_MIN_CM) / (HGCAL_Z_MAX_CM - HGCAL_Z_MIN_CM)
        width_growth = 0.7 + 1.8 * depth_frac
        tail = rng.standard_t(df=3) if rng.random() < 0.22 else rng.normal()
        local_eta_spread = angular_spread * scatter_scale * width_growth
        local_phi_spread = angular_spread * scatter_scale * width_growth
        if abs_pdg not in {22, 11} and rng.random() < 0.30:
            local_eta_spread *= rng.uniform(1.4, 3.0)
            local_phi_spread *= rng.uniform(1.4, 3.0)

        eta_center = axis["eta"] + axis["z_sign"] * axis.get("eta_drift", 0.0) * depth_frac + branch_eta * depth_frac
        phi_center = _wrap_phi(axis["phi"] + axis.get("phi_drift", 0.0) * depth_frac + branch_phi * depth_frac)

        eta = eta_center + axis["z_sign"] * (rng.normal(0.0, core_jitter) + tail * local_eta_spread)
        eta_abs = np.clip(abs(eta), 1.5, 3.0)
        eta = axis["z_sign"] * eta_abs
        phi = float(_wrap_phi(phi_center + rng.normal(0.0, core_jitter) + rng.normal(0.0, local_phi_spread)))
        position = _eta_phi_z_to_xyz(eta, phi, z_abs, axis["z_sign"])

        direction = _normalised(position + rng.normal(0.0, 5.0 if not is_pu else 8.0, size=3))
        raw_energy = max(0.02, energy * frac * rng.lognormal(mean=0.0, sigma=0.18 if not is_pu else 0.35))
        em_fraction = rng.uniform(0.72, 0.95) if abs_pdg in {22, 11} else rng.beta(1.8, 3.6)
        raw_em_energy = raw_energy * em_fraction

        num_lcs = int(max(1 if is_pu else 2, rng.poisson(1.6 + 1.35 * math.sqrt(raw_energy))))
        num_hits = int(max(num_lcs, rng.poisson(num_lcs * rng.uniform(2.0, 5.5))))
        local_length = rng.uniform(3.0, 18.0 if abs_pdg in {22, 11} else 42.0)
        transverse_width = rng.uniform(0.4, 2.8 if abs_pdg in {22, 11} else 8.0) * scatter_scale
        ev1 = local_length**2 * rng.uniform(0.7, 1.3)
        ev2 = transverse_width**2 * rng.uniform(0.7, 1.4)
        ev3 = transverse_width**2 * rng.uniform(0.5, 1.2)
        sigma1 = math.sqrt(ev1) * rng.uniform(0.05, 0.22)
        sigma2 = math.sqrt(ev2) * rng.uniform(0.30, 0.80)
        sigma3 = math.sqrt(ev3) * rng.uniform(0.30, 0.80)

        z_span = max(1.0, local_length * rng.uniform(0.4, 0.9))
        z_a = axis["z_sign"] * (z_abs - z_span / 2)
        z_b = axis["z_sign"] * (z_abs + z_span / 2)
        z_min = min(z_a, z_b)
        z_max = max(z_a, z_b)
        time = time0 + rng.normal(0.0, 0.030 if not is_pu else 0.16) + depth_frac * rng.normal(0.006, 0.012)

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
        if is_pu and rng.random() < 0.22:
            eta_sigma = rng.uniform(0.010, 0.035)
            phi_sigma = rng.uniform(0.010, 0.035)
        else:
            eta_sigma = rng.uniform(0.035, 0.16 if is_pu else 0.12)
            phi_sigma = rng.uniform(0.035, 0.16 if is_pu else 0.12)
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
        axis = _sample_axis_near_centers(rng, centers + axes, is_pu=True, close_pair_fraction=0.82)
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
        "Shower fragments include depth-dependent drift, widening, heavy-tailed angular scatter, and broad PU timing.",
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
