# Input Data

This directory documents the input expected by `src/pv_arc_fault_fft_benchmark.py`. Large or restricted data files are intentionally ignored by Git.

## Public source dataset

The article cites:

> Michel Oliveira, Filipe Ramos, José Neto, and Bruno Lima. “Photovoltaic Inverter Input Arc-Fault and Normal Operation Waveforms Dataset.” IEEE DataPort, 2025. DOI: [10.21227/WZFE-7973](https://doi.org/10.21227/WZFE-7973).

## Labeled workflow input

The reproduction script needs 16 CSV files with sample-level labels. Place them in one of these forms at the repository root:

```text
data_labeled.zip
```

or:

```text
data_labeled/Experiment_1.csv
...
data_labeled/Experiment_16.csv
```

Names `Experiment_01.csv` through `Experiment_16.csv` are also accepted.

Required header:

```csv
CH1,CH2,CLASSIFIER
```

- `CH1`: inverter input current.
- `CH2`: DC-bus voltage.
- `CLASSIFIER`: `Normal` or `Arc`.

Run:

```powershell
python scripts/check_input_data.py
```

The validator checks file coverage, accepted names, headers, and label values without importing TensorFlow or scikit-learn.

## Redistribution status

The labeled derivative package supplied during manuscript review is not included here because its independent public-redistribution authorization and license have not been confirmed. Do not commit it merely because the raw measurements are publicly discoverable. If release is authorized, publish the labeled archive with an explicit dataset license and provenance statement, preferably as a versioned archival dataset or a GitHub Release asset rather than normal Git history.

For internal integrity checking, the reviewer archive's nested `data_labeled.zip` had SHA-256:

```text
c77dc5e63def86df00d2e40fbdf92ad3532db5859bd48385ab4eea9760b35372
```
