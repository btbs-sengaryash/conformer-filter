# conformer-filter
Lightweight Python framework for conformer filtering and deduplication

# conformer-filter

A lightweight, tool-agnostic Python framework for conformer 
filtering and deduplication across MD simulations and 
conformer generation pipelines.

**Developed as open-source contribution to Rowan Scientific**

## What does it do?

Given a large ensemble of molecular conformers from MD 
simulations, this framework:

- Computes pairwise RMSD matrix (Kabsch alignment)
- Clusters similar conformers using RMSD threshold  
- Returns diverse, non-redundant representatives
- Generates diagnostic plots

## Results

Tested on Lysozyme (1AKI, 129 residues):
- Input: 50 MD conformers
- Output: ~4-5 diverse representatives  
- Redundancy removed: ~88-92%

## Installation

pip install numpy matplotlib scipy scikit-learn

## Quick Start

```python
from conformer_filter import ConformerFilter

cf = ConformerFilter(threshold=2.0)
cf.load_from_arrays(conformers)
cf.fit()
cf.summary()
```

## Key Methods

| Method | Description |
|--------|-------------|
| RMSD + Kabsch | Optimal structural superposition |
| Greedy clustering | Fast, interpretable grouping |
| Threshold sensitivity | Benchmarking optimal cutoff |

## Author

Yash Singh Sengar — IIT Ropar / NIT Calicut  
Open-source contribution to Rowan Scientific
