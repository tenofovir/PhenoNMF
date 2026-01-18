# PhenoNMF

PhenoNMF, a multimodal EHR phenotyping framework based on modified joint nonnegative matrix factorization. PhenoNMF simultaneously learns sparse phenotypes from diagnoses, laboratory tests, and medications. By excluding age from the decomposition and reintroducing it through age-weighted contribution projections, PhenoNMF simultaneously captures multimodality co-occurrence within each common pattern module (CPM) and compares them across age groups. Within each CPM, we define an Age Network Coupling Score (ANCS) that ranks diagnosis–laboratory–medication triads by the strength and age specificity of their cross-modality associations.

![Method Overview](Graphical abstract.pdf)

## Input Data

Sample data located in `sample_data/`:
- `X1`: Diagnosis matrix
- `X2`: Lab tests matrix
- `X3`: Medication matrix

## Files

| File | Description |
|------|-------------|
| `classHexaNMF_mask.py` | Core PhenoNMF algorithm |
| `classRun_HexaNMF_CCLE_mask_multi_phenonmf.py` | Runner for PhenoNMF with consensus matrix |
| `classSaveData_of_PhenoNMF.py` | Save output matrices |
| `classHexaConsensusMatrix_silhouette.py` | Consensus matrix and clustering metrics |
| `sampling.py` | Patient sampling |
| `sample2real_Consistency_Evalulation.py` | Evaluate clustering consistency |

## How to Run

1. Run rank selection:
```bash
python calc_select_Rank_k_silhouette_phenonmf.py
```

2. (Optional) Run sampling:
```bash
python sampling.py
```

3. (Optional) Evaluate consistency:
```bash
python sample2real_Consistency_Evalulation.py
```
