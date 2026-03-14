import os
import urllib.request
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MD17Sample:
    position:  torch.Tensor  # (n_atoms, 3)  or  (B, n_atoms, 3) when batched
    energy:    torch.Tensor  # scalar         or  (B,)            when batched
    force:     torch.Tensor  # (n_atoms, 3)  or  (B, n_atoms, 3) when batched
    atom_type: torch.Tensor  # (n_atoms, D)  or  (B, n_atoms, D) when batched


def collate_md17(samples: List[MD17Sample]) -> MD17Sample:
    return MD17Sample(
        position=torch.stack([s.position  for s in samples]),
        energy=torch.stack([s.energy     for s in samples]),
        force=torch.stack([s.force       for s in samples]),
        atom_type=torch.stack([s.atom_type for s in samples]),
    )

_BASE_URL = "https://www.quantum-machine.org/gdml/data/npz/"

_MOLECULES = {
    "aspirin",
    "azobenzene",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "paracetamol",
    "salicylic",
    "toluene",
    "uracil",
}


def _local_path(molecule: str) -> str:
    return os.path.join(os.path.expanduser("~/.cache/aperol/md17"), f"{molecule}_dft.npz")


def _download(molecule: str) -> str:
    if molecule not in _MOLECULES:
        raise ValueError(f"Unknown molecule '{molecule}'. Available: {sorted(_MOLECULES)}")
    path = _local_path(molecule)
    url = _BASE_URL + f"md17_{molecule}.npz"
    print(f"Downloading {url} -> {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


class MD17Dataset(Dataset):
    """MD17 molecular dynamics dataset.

    Parameters
    ----------
    path : str
        Path to the ``<molecule>_dft.npz`` file. If the file does not exist it
        is downloaded from the official quantum-machine.org repository.
    indices : array-like, optional
        Subset of sample indices to use. If ``None``, all samples are used.
    seed : int
        Random seed for reproducible permutation.
    """

    def __init__(self, molecule: str, indices=None, seed: int = 2666):
        path = _local_path(molecule)
        if not os.path.exists(path):
            _download(molecule)
        data = np.load(path)
        np.random.seed(seed)
        perm = np.random.permutation(len(data["R"]))

        R = data["R"][perm]          # (n_samples, n_atoms, 3)
        E = data["E"][perm].squeeze(-1)  # (n_samples,)
        F = data["F"][perm]          # (n_samples, n_atoms, 3)
        z = data["z"]         # (n_atoms,)  atomic numbers, constant across frames

        # normalize energy to zero mean / unit std
        E = (E - E.mean()) / E.std()

        if indices is not None:
            R = R[indices]
            E = E[indices]
            F = F[indices]

        self.position  = torch.tensor(R, dtype=torch.float32)   # (N, n_atoms, 3)
        self.energy    = torch.tensor(E, dtype=torch.float32)   # (N,)
        self.force     = torch.tensor(F, dtype=torch.float32)   # (N, n_atoms, 3)

        # one-hot encode atom types: (n_atoms, n_species)
        z_tensor = torch.tensor(z, dtype=torch.int64)
        self.atom_type = torch.nn.functional.one_hot(z_tensor).float()  # (n_atoms, D)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.energy)

    def __getitem__(self, idx):
        return MD17Sample(
            position=self.position[idx],
            energy=self.energy[idx],
            force=self.force[idx],
            atom_type=self.atom_type,
        )

    # ------------------------------------------------------------------
    @property
    def n_atom_features(self) -> int:
        return self.atom_type.shape[-1]


def load_md17(molecule: str, n_tr: int, n_vl: int = 0, seed: int = 2666):
    """Return train / validation / test ``MD17Dataset`` splits.

    Parameters
    ----------
    molecule : str
        Molecule name, e.g. ``"malonaldehyde"``.
    n_tr : int
        Number of training samples.
    n_vl : int
        Number of validation samples. Defaults to ``n_tr`` when 0.
    seed : int
        Random seed passed to all splits (same permutation).

    Returns
    -------
    train, val, test : MD17Dataset
    """
    if molecule not in _MOLECULES:
        raise ValueError(f"Unknown molecule '{molecule}'. Available: {sorted(_MOLECULES)}")

    if n_vl == 0:
        n_vl = n_tr

    path = _local_path(molecule)
    if not os.path.exists(path):
        _download(molecule)

    n_total = len(np.load(path)["R"])
    all_idx = np.arange(n_total)

    train = MD17Dataset(molecule, indices=all_idx[:n_tr],            seed=seed)
    val   = MD17Dataset(molecule, indices=all_idx[n_tr:n_tr + n_vl], seed=seed)
    test  = MD17Dataset(molecule, indices=all_idx[n_tr + n_vl:],     seed=seed)

    return train, val, test
