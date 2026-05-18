# Run Outputs

Large paper-side result files are not tracked in git.

Download the run-output archive from Google Drive:

```text
https://drive.google.com/file/d/1JpvzcwgPaw4TWVa1Ajmdo8SwFLPldFW-/view?usp=sharing
```

Expected file:

```text
ptnas_run_outputs_20260517.tar.gz
```

Extract from the repository root:

```bash
tar -xzf run_outputs/ptnas_run_outputs_20260517.tar.gz -C run_outputs
```

Expected layout:

```text
run_outputs/
├── code/      # table aggregation and plotting scripts
├── data/      # compact result files used by the paper
├── example_ptnas_10s.csv
└── example_ptnas_10s.log
```

The two example files show the CSV and console log produced by a 10-second pTNAS RelBench run.
