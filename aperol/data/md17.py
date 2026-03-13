import numpy as np
import torch
from torch.utils.data import Dataset


class MD17Dataset(Dataset):
    """MD17 molecular dynamics dataset.

    Parameters
    ----------
    path : str
        Path to the ``<molecule>_dft.npz`` file.
    indices : array-like, optional
        Subset of sample indices to use. If ``None``, all samples are used.
    seed : int
        Random seed for reproducible permutation.
    """

    def __init__(self, path: str, indices=None, seed: int = 2666):
        data = np.load(path)
        np.random.seed(seed)
        perm = np.random.permutation(len(data["R"]))

        R = data["R"][perm]   # (n_samples, n_atoms, 3)
        E = data["E"][perm]   # (n_samples,)
        F = data["F"][perm]   # (n_samples, n_atoms, 3)
        z = data["z"]         # (n_atoms,)  atomic numbers, constant across frames

        # normalise energy to zero mean / unit std
        E = (E - E.mean()) / E.std()

        if indices is not None:
            R = R[indices]
            E = E[indices]
            F = F[indices]

        self.positions = torch.tensor(R, dtype=torch.float32)   # (N, n_atoms, 3)
        self.energies  = torch.tensor(E, dtype=torch.float32)   # (N,)
        self.forces    = torch.tensor(F, dtype=torch.float32)   # (N, n_atoms, 3)

        # one-hot encode atom types: (n_atoms, n_species)
        z_tensor = torch.tensor(z, dtype=torch.int64)
        self.atom_types = torch.nn.functional.one_hot(z_tensor).float()  # (n_atoms, D)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.energies)

    def __getitem__(self, idx):
        return self.positions[idx], self.energies[idx], self.forces[idx]

    # ------------------------------------------------------------------
    @property
    def n_atom_features(self) -> int:
        return self.atom_types.shape[-1]


def load_md17(path: str, n_tr: int, n_vl: int = 0, seed: int = 2666):
    """Return train / validation / test ``MD17Dataset`` splits.

    Parameters
    ----------
    path : str
        Path to the ``<molecule>_dft.npz`` file.
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
    if n_vl == 0:
        n_vl = n_tr

    n_total = len(np.load(path)["R"])
    all_idx = np.arange(n_total)

    train = MD17Dataset(path, indices=all_idx[:n_tr],            seed=seed)
    val   = MD17Dataset(path, indices=all_idx[n_tr:n_tr + n_vl], seed=seed)
    test  = MD17Dataset(path, indices=all_idx[n_tr + n_vl:],     seed=seed)

    return train, val, test
