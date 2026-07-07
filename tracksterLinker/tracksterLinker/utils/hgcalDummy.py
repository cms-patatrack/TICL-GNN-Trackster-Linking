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
    scenario: str = "mixed"
    train_files: int = 80
    val_files: int = 20
    test_files: int = 20
    events_per_file: int = 10
    signal_mean: float = 12.0
    pu_mean: float = 35.0
    close_pair_fraction: float = 0.35
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


def _sample_signal_pdg(rng):
    # Matches the pion-heavy SingleParticle140PU mix described in the thesis.
    return int(rng.choice([211, 22, 11, -11, 15, 130, 321, -321], p=[0.80, 0.05, 0.025, 0.025, 0.05, 0.02, 0.015, 0.015]))


def _sample_multiparticle_pdg(rng):
    return int(rng.choice([22, 11, -11, 211, 130, 321, -321]))


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


def _near_axis(axis, rng, eta_sigma=0.025, phi_sigma=0.025):
    eta_abs = np.clip(abs(axis["eta"]) + rng.normal(0.0, eta_sigma), 1.5, 3.0)
    return {
        "eta": axis["z_sign"] * eta_abs,
        "eta_abs": eta_abs,
        "phi": float(_wrap_phi(axis["phi"] + rng.normal(0.0, phi_sigma))),
        "z_sign": axis["z_sign"],
    }


def _sample_energy(rng, axis, use_pt=False):
    if use_pt:
        pt = rng.uniform(10.0, 100.0)
        return float(np.clip(pt * math.cosh(axis["eta_abs"]), 10.0, 600.0))
    return float(rng.uniform(10.0, 600.0))


def _fragment_count(rng, energy, pdg_id, is_pu):
    abs_pdg = abs(int(pdg_id))
    if abs_pdg in {22, 11}:
        mean = 1.8 + 0.010 * energy
    elif abs_pdg in {211, 130, 321, 15}:
        mean = 3.5 + 0.018 * energy
    else:
        mean = 2.0 + 0.012 * energy

    if is_pu:
        mean *= 0.45

    return int(np.clip(1 + rng.poisson(mean), 1, 24))


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
        z_stop = rng.uniform(365.0, 430.0)
    else:
        z_stop = rng.uniform(445.0, HGCAL_Z_MAX_CM)

    z_start = rng.uniform(HGCAL_Z_MIN_CM, 345.0)
    depths = np.linspace(z_start, z_stop, n_fragments)
    depths += rng.normal(0.0, 5.0, size=n_fragments)
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
    angular_spread = 0.006 if abs_pdg in {22, 11} else 0.014
    time0 = rng.normal(0.0, 0.035 if not is_pu else 0.18)

    for idx, (z_abs, frac) in enumerate(zip(depths, fractions)):
        eta = axis["eta"] + axis["z_sign"] * rng.normal(0.0, angular_spread)
        eta_abs = np.clip(abs(eta), 1.5, 3.0)
        eta = axis["z_sign"] * eta_abs
        phi = float(_wrap_phi(axis["phi"] + rng.normal(0.0, angular_spread)))
        position = _eta_phi_z_to_xyz(eta, phi, z_abs, axis["z_sign"])

        direction = _normalised(position + rng.normal(0.0, 3.0, size=3))
        raw_energy = max(0.05, energy * frac * rng.lognormal(mean=0.0, sigma=0.08))
        em_fraction = 0.85 if abs_pdg in {22, 11} else rng.uniform(0.15, 0.55)
        raw_em_energy = raw_energy * em_fraction

        num_lcs = int(max(2, rng.poisson(2.5 + 1.6 * math.sqrt(raw_energy))))
        num_hits = int(max(num_lcs, rng.poisson(num_lcs * rng.uniform(2.0, 5.5))))
        local_length = rng.uniform(3.0, 18.0 if abs_pdg in {22, 11} else 32.0)
        transverse_width = rng.uniform(0.4, 2.5 if abs_pdg in {22, 11} else 5.0)
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
        time = time0 + rng.normal(0.0, 0.025 if not is_pu else 0.10) + idx * rng.normal(0.001, 0.003)

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


def _event_axes_for_scenario(rng, scenario, signal_mean, close_pair_fraction):
    if scenario == "closeby_pions":
        base = _sample_axis(rng, z_sign=int(rng.choice([-1, 1])), eta_range=(1.7, 2.7))
        return [base, _near_axis(base, rng, eta_sigma=0.020, phi_sigma=0.020)], [211, 211], [False, False]

    if scenario == "multiparticle":
        n_particles = int(rng.integers(10, 51))
        base = _sample_axis(rng, z_sign=int(rng.choice([-1, 1])), eta_range=(1.7, 2.7))
        axes = [base]
        for _ in range(n_particles - 1):
            axes.append(_near_axis(base, rng, eta_sigma=0.12, phi_sigma=0.12))
        return axes, [_sample_multiparticle_pdg(rng) for _ in axes], [False] * len(axes)

    if scenario == "single_particle_pu":
        axes = [_sample_axis(rng, z_sign=1), _sample_axis(rng, z_sign=-1)]
        return axes, [_sample_signal_pdg(rng), _sample_signal_pdg(rng)], [False, False]

    n_particles = max(2, int(rng.poisson(signal_mean)))
    axes = []
    pdgs = []
    is_pu = []
    for idx in range(n_particles):
        if idx > 0 and axes and rng.random() < close_pair_fraction:
            axis = _near_axis(axes[int(rng.integers(0, len(axes)))], rng, eta_sigma=0.04, phi_sigma=0.04)
        else:
            axis = _sample_axis(rng)
        axes.append(axis)
        pdgs.append(_sample_signal_pdg(rng))
        is_pu.append(False)
    return axes, pdgs, is_pu


def generate_event(rng, scenario="mixed", signal_mean=12.0, pu_mean=35.0, close_pair_fraction=0.35):
    if scenario == "mixed":
        scenario = str(rng.choice(["closeby_pions", "multiparticle", "single_particle_pu"], p=[0.25, 0.35, 0.40]))

    axes, pdgs, pu_flags = _event_axes_for_scenario(rng, scenario, signal_mean, close_pair_fraction)
    if scenario == "single_particle_pu":
        n_pu = int(rng.poisson(pu_mean))
    else:
        n_pu = int(rng.poisson(max(0.0, 0.15 * pu_mean)))

    for _ in range(n_pu):
        axis = _sample_axis(rng)
        axes.append(axis)
        pdgs.append(_sample_signal_pdg(rng))
        pu_flags.append(True)

    rows = []
    labels = []
    is_pu = []
    for sim_id, (axis, pdg_id, pu_flag) in enumerate(zip(axes, pdgs, pu_flags)):
        energy = _sample_energy(rng, axis, use_pt=(scenario == "single_particle_pu" and not pu_flag))
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


def generate_events(n_events, rng, scenario="mixed", signal_mean=12.0, pu_mean=35.0, close_pair_fraction=0.35):
    return ak.Array(
        [
            generate_event(
                rng,
                scenario=scenario,
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
                scenario=config.scenario,
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
        "Signal mixture is pion-dominated and PU-like events include many non-signal showers.",
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
