"""
conformer_filter.py
====================
Lightweight, tool-agnostic framework for conformer filtering
and deduplication across MD and conformer generation pipelines.

Developed for Rowan Scientific open-source contribution.
Author: Yash Singh Sengar
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ================================================================
#  SECTION 1: COORDINATE LOADING
#  Supports: PDB, XYZ, numpy arrays
# ================================================================

def load_pdb_coords(pdb_file):
    """
    Load CA (alpha carbon) coordinates from a PDB file.
    Returns: numpy array of shape (N_atoms, 3)
    """
    coords = []
    atom_names = []

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                # For proteins: use CA atoms only
                # For small molecules: use all heavy atoms
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                    atom_names.append(atom_name)
                except ValueError:
                    continue

    return np.array(coords), atom_names


def load_xyz_coords(xyz_file):
    """
    Load coordinates from XYZ format file.
    Returns: numpy array of shape (N_atoms, 3)
    """
    coords = []
    atom_types = []

    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            n_atoms = int(line)
            i += 2  # skip comment line
            for _ in range(n_atoms):
                parts = lines[i].strip().split()
                atom_types.append(parts[0])
                coords.append([float(parts[1]),
                                float(parts[2]),
                                float(parts[3])])
                i += 1
        else:
            i += 1

    return np.array(coords), atom_types


# ================================================================
#  SECTION 2: RMSD CALCULATION
#  The mathematical heart of structural similarity
# ================================================================

def center_coords(coords):
    """
    Translate coordinates so center of mass = origin.
    This is Step 1 before RMSD calculation.
    """
    centroid = coords.mean(axis=0)
    return coords - centroid


def kabsch_rotation(P, Q):
    """
    Kabsch algorithm: find optimal rotation matrix R
    that minimizes RMSD between P and Q.

    Math:
    1. Compute covariance matrix H = P^T * Q
    2. SVD: H = U * S * V^T
    3. R = V * U^T  (handle reflection)

    Args:
        P: reference coords (N, 3)
        Q: mobile coords   (N, 3)
    Returns:
        R: rotation matrix (3, 3)
    """
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)

    # Handle reflection (det = -1 case)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])

    R = Vt.T @ D @ U.T
    return R


def calculate_rmsd(coords1, coords2, align=True):
    """
    Calculate RMSD between two conformers.

    If align=True: uses Kabsch algorithm for optimal superposition
    If align=False: just calculates raw RMSD (faster)

    RMSD = sqrt( (1/N) * sum( |ri - ri'|^2 ) )

    Args:
        coords1: reference coordinates (N, 3)
        coords2: mobile coordinates   (N, 3)
        align:   whether to align first

    Returns:
        rmsd: float (Angstroms)
    """
    if len(coords1) != len(coords2):
        raise ValueError(
            f"Atom count mismatch: {len(coords1)} vs {len(coords2)}"
        )

    # Center both structures
    P = center_coords(coords1.copy())
    Q = center_coords(coords2.copy())

    if align:
        # Apply Kabsch rotation
        R = kabsch_rotation(P, Q)
        Q = Q @ R.T

    # Calculate RMSD
    diff = P - Q
    rmsd = np.sqrt((diff ** 2).sum() / len(P))
    return rmsd


# ================================================================
#  SECTION 3: RMSD MATRIX
#  Pairwise RMSD between ALL conformer pairs
# ================================================================

def compute_rmsd_matrix(conformers, align=True, verbose=True):
    """
    Compute N x N pairwise RMSD matrix for a list of conformers.

    Args:
        conformers: list of numpy arrays, each shape (N_atoms, 3)
        align:      use Kabsch alignment
        verbose:    show progress

    Returns:
        rmsd_matrix: numpy array (N_conf, N_conf)
    """
    n = len(conformers)
    rmsd_matrix = np.zeros((n, n))

    total_pairs = n * (n - 1) // 2
    computed = 0

    if verbose:
        print(f"  Computing {total_pairs} pairwise RMSDs "
              f"for {n} conformers...")

    for i in range(n):
        for j in range(i + 1, n):
            rmsd = calculate_rmsd(conformers[i], conformers[j],
                                  align=align)
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd
            computed += 1

        if verbose and (i + 1) % 10 == 0:
            pct = computed / total_pairs * 100
            print(f"    Progress: {pct:.1f}%", end='\r')

    if verbose:
        print(f"    Done! Matrix shape: {rmsd_matrix.shape}")

    return rmsd_matrix


# ================================================================
#  SECTION 4: CLUSTERING
#  Group similar conformers together
# ================================================================

def greedy_rmsd_clustering(conformers, rmsd_matrix,
                           threshold=2.0, verbose=True):
    """
    Greedy clustering: iteratively pick the most 'central'
    conformer as cluster representative, assign all within
    threshold to that cluster.

    This is fast and physically interpretable.

    Args:
        conformers:  list of coordinate arrays
        rmsd_matrix: precomputed pairwise RMSD matrix
        threshold:   RMSD cutoff in Angstroms (default 2.0 A)
        verbose:     print cluster info

    Returns:
        clusters:       dict {rep_idx: [member_indices]}
        representatives: list of representative indices
    """
    n = len(conformers)
    assigned = [False] * n
    clusters = {}
    representatives = []

    # Sort by how many neighbors each conformer has
    # (most "popular" conformer becomes representative)
    neighbor_counts = [
        np.sum(rmsd_matrix[i] < threshold) for i in range(n)
    ]
    priority_order = np.argsort(neighbor_counts)[::-1]

    for idx in priority_order:
        if assigned[idx]:
            continue

        # This conformer becomes a cluster representative
        members = []
        for j in range(n):
            if not assigned[j] and rmsd_matrix[idx, j] <= threshold:
                members.append(j)
                assigned[j] = True

        clusters[idx] = members
        representatives.append(idx)

    if verbose:
        print(f"\n  Clustering Results (threshold = {threshold} Å):")
        print(f"  Input conformers  : {n}")
        print(f"  Clusters found    : {len(clusters)}")
        print(f"  Reduction         : "
              f"{(1 - len(clusters)/n)*100:.1f}% redundancy removed")
        for rep, members in clusters.items():
            print(f"  Cluster {rep:3d}: "
                  f"{len(members):3d} members")

    return clusters, representatives


def hierarchical_clustering(rmsd_matrix, threshold=2.0):
    """
    Scipy hierarchical clustering on RMSD matrix.
    Alternative to greedy — gives dendrogram.

    Returns: cluster labels array
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    # Convert square matrix to condensed form
    condensed = squareform(rmsd_matrix)

    # Ward linkage
    Z = linkage(condensed, method='average')

    # Cut at threshold
    labels = fcluster(Z, t=threshold, criterion='distance')

    return labels, Z


# ================================================================
#  SECTION 5: MAIN FILTER CLASS
#  Clean API — this is what users interact with
# ================================================================

class ConformerFilter:
    """
    Main class for conformer filtering and deduplication.

    Usage:
    ------
    cf = ConformerFilter(threshold=2.0)
    cf.load_from_arrays(coords_list)
    cf.fit()
    diverse_conformers = cf.get_representatives()
    cf.plot_rmsd_matrix()
    cf.plot_cluster_summary()
    """

    def __init__(self, threshold=2.0, align=True,
                 method='greedy', verbose=True):
        """
        Args:
            threshold: RMSD cutoff in Angstroms
            align:     use Kabsch alignment
            method:    'greedy' or 'hierarchical'
            verbose:   print progress
        """
        self.threshold  = threshold
        self.align      = align
        self.method     = method
        self.verbose    = verbose

        # Internal state
        self.conformers      = []
        self.rmsd_matrix     = None
        self.clusters        = None
        self.representatives = None
        self.labels          = None
        self._fitted         = False

    def load_from_arrays(self, coords_list):
        """Load conformers from list of numpy arrays."""
        self.conformers = [np.array(c) for c in coords_list]
        if self.verbose:
            print(f"Loaded {len(self.conformers)} conformers.")
        return self

    def load_from_pdbs(self, pdb_files, atom_filter='CA'):
        """Load conformers from list of PDB files."""
        self.conformers = []
        for pdb in pdb_files:
            coords, names = load_pdb_coords(pdb)
            if atom_filter == 'CA':
                mask = [i for i, n in enumerate(names) if n == 'CA']
                if mask:
                    coords = coords[mask]
            self.conformers.append(coords)
        if self.verbose:
            print(f"Loaded {len(self.conformers)} conformers "
                  f"from PDB files.")
        return self

    def fit(self):
        """Run RMSD matrix computation and clustering."""
        if not self.conformers:
            raise ValueError("No conformers loaded!")

        # Step 1: Compute RMSD matrix
        if self.verbose:
            print("\n[1/2] Computing pairwise RMSD matrix...")
        self.rmsd_matrix = compute_rmsd_matrix(
            self.conformers,
            align=self.align,
            verbose=self.verbose
        )

        # Step 2: Cluster
        if self.verbose:
            print("\n[2/2] Clustering conformers...")

        if self.method == 'greedy':
            self.clusters, self.representatives = \
                greedy_rmsd_clustering(
                    self.conformers,
                    self.rmsd_matrix,
                    threshold=self.threshold,
                    verbose=self.verbose
                )
        else:
            self.labels, self.Z = hierarchical_clustering(
                self.rmsd_matrix,
                threshold=self.threshold
            )
            unique_labels = np.unique(self.labels)
            self.representatives = [
                np.where(self.labels == l)[0][0]
                for l in unique_labels
            ]

        self._fitted = True
        return self

    def get_representatives(self):
        """Return coordinates of representative conformers."""
        if not self._fitted:
            raise RuntimeError("Run .fit() first!")
        return [self.conformers[i] for i in self.representatives]

    def summary(self):
        """Print summary statistics."""
        if not self._fitted:
            raise RuntimeError("Run .fit() first!")

        n_in  = len(self.conformers)
        n_out = len(self.representatives)

        print("\n" + "="*50)
        print("  CONFORMER FILTER SUMMARY")
        print("="*50)
        print(f"  Input conformers    : {n_in}")
        print(f"  Unique clusters     : {n_out}")
        print(f"  Redundancy removed  : "
              f"{(1-n_out/n_in)*100:.1f}%")
        print(f"  RMSD threshold      : {self.threshold} Å")
        print(f"  Alignment method    : "
              f"{'Kabsch' if self.align else 'None'}")
        print(f"\n  RMSD Matrix Stats:")
        upper = self.rmsd_matrix[
            np.triu_indices_from(self.rmsd_matrix, k=1)
        ]
        print(f"    Min RMSD  : {upper.min():.3f} Å")
        print(f"    Max RMSD  : {upper.max():.3f} Å")
        print(f"    Mean RMSD : {upper.mean():.3f} Å")
        print(f"    Std RMSD  : {upper.std():.3f} Å")
        print("="*50)


# ================================================================
#  SECTION 6: VISUALIZATION
# ================================================================

def plot_rmsd_matrix(rmsd_matrix, title="Pairwise RMSD Matrix",
                     save_path=None):
    """
    Heatmap of pairwise RMSD matrix.
    Dark = similar, Light = different.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(rmsd_matrix, cmap='viridis_r',
                   aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='RMSD (Å)')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Conformer Index', fontsize=11)
    ax.set_ylabel('Conformer Index', fontsize=11)

    # Add mean RMSD annotation
    upper = rmsd_matrix[np.triu_indices_from(rmsd_matrix, k=1)]
    ax.text(0.02, 0.98,
            f'Mean RMSD: {upper.mean():.2f} Å\n'
            f'Max RMSD:  {upper.max():.2f} Å',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_rmsd_distribution(rmsd_matrix, threshold=2.0,
                           save_path=None):
    """
    Histogram of all pairwise RMSD values.
    Shows where threshold cuts the distribution.
    """
    import matplotlib.pyplot as plt

    upper = rmsd_matrix[np.triu_indices_from(rmsd_matrix, k=1)]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(upper, bins=50, color='steelblue',
            edgecolor='white', linewidth=0.5, alpha=0.85,
            label='Pairwise RMSDs')

    ax.axvline(threshold, color='red', lw=2, ls='--', 
               label=f'Threshold = {threshold} Å')

    below = np.sum(upper < threshold)
    pct   = below / len(upper) * 100
    ax.text(threshold * 1.05, ax.get_ylim()[1] * 0.9,
            f'{pct:.1f}% pairs\nbelow threshold',
            color='red', fontsize=9)

    ax.set_xlabel('RMSD (Å)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Pairwise RMSD Values',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_cluster_summary(clusters, rmsd_matrix,
                         save_path=None):
    """
    Bar chart showing cluster sizes.
    Larger bars = more redundancy removed.
    """
    import matplotlib.pyplot as plt

    cluster_sizes = [len(v) for v in clusters.values()]
    cluster_ids   = list(range(len(cluster_sizes)))

    # Sort by size descending
    sorted_pairs = sorted(zip(cluster_sizes, cluster_ids),
                          reverse=True)
    sizes, ids   = zip(*sorted_pairs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Clustering Results Summary',
                 fontsize=13, fontweight='bold')

    # Left: cluster sizes
    colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
    axes[0].bar(range(len(sizes)), sizes,
                color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Cluster Rank', fontsize=11)
    axes[0].set_ylabel('Number of Members', fontsize=11)
    axes[0].set_title('Cluster Sizes\n'
                      '(Each bar = 1 unique conformer)', fontsize=11)
    axes[0].axhline(1, color='gray', ls=':', lw=1,
                    label='Size = 1 (unique)')
    axes[0].legend(fontsize=9)

    # Right: before vs after
    n_before = sum(sizes)
    n_after  = len(sizes)
    axes[1].bar(['Before\nFiltering', 'After\nFiltering'],
                [n_before, n_after],
                color=['#E74C3C', '#2ECC71'],
                edgecolor='black', linewidth=0.8,
                width=0.5)
    axes[1].set_ylabel('Number of Conformers', fontsize=11)
    axes[1].set_title('Redundancy Reduction',
                      fontsize=11)
    for i, v in enumerate([n_before, n_after]):
        axes[1].text(i, v + n_before * 0.01, str(v),
                     ha='center', fontweight='bold', fontsize=12)
    reduction = (1 - n_after/n_before) * 100
    axes[1].text(0.5, 0.85,
                 f'{reduction:.1f}% redundancy\nremoved',
                 transform=axes[1].transAxes,
                 ha='center', fontsize=11,
                 color='#27AE60', fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig
