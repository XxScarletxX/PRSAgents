# Stage 2 Next Steps: Teammate-Fitted Target Scores

Use this checklist after the teammate exports `target_scores.csv`.

## 1. Place and validate the export

Confirm that the file is available at:

```text
outputs/stage1_demo/target_scores.csv
```

Check the columns and target IID alignment:

```powershell
python -c "import pandas as p; s=p.read_csv('outputs/stage1_demo/target_scores.csv'); print(s.shape); print(s.columns.tolist()); print(s.head())"
```

The file should contain:

- `iid`
- `EUR_viprs`
- `AFR_viprs`
- `EAS_viprs`
- `SAS_viprs`
- `AMR_viprs`

The export must contain five expert score columns and one row per target individual. The target IIDs must align with the target cohort. Confirm that each score was normalized using training individuals only.

## 2. Run the full baseline once

Run the fitted-score baseline on the shared-effects scenario:

```powershell
python -m stages.stage2.stage2_baselines data/shared_effects --scores outputs/stage1_demo/target_scores.csv --out outputs/stage2_fitted_scores
```

This comparison should include:

- Best single PRS
- MultiPRS
- Genotype ridge
- Stage 2 MoEPRS

Keep this run separate from the supplied-VIPRS baseline in `outputs/stage2_demo`.

## 3. Create fitted-score diagnostics

Generate plots from the fitted-score exports:

```powershell
python -m stages.stage2.stage2_plots outputs/stage2_fitted_scores --out outputs/stage2_figures/fitted_scores
```

Review:

- Validation incremental R2
- Validation MSE
- Training likelihood
- Gate weights versus PC1
- Gate weights versus PC2
- Whether every gate row sums to one

Do not mix fitted-score and supplied-VIPRS results in the same headline plot.

## 4. Rerun the training-size study

The current training-size script does not accept a `--scores` argument. Before rerunning it, add a `--scores` option to `stage2_training_sizes.py` and pass it to `stage2_baselines.run`.

The intended commands are:

```powershell
python -m stages.stage2.stage2_training_sizes data/shared_effects --replicate 0 --scores outputs/stage1_demo/target_scores.csv --out outputs/stage2_fitted_training_sizes --figures outputs/stage2_figures/fitted_scores
```

This should produce the same 250, 500, 1000, and 2000 training-size experiment using fitted experts.

## 5. Rerun the replicate sweep

The current replicate script does not accept a `--scores` argument. Before rerunning it, add a `--scores` option to `stage2_replicates.py` and pass it to `stage2_baselines.run`.

The intended command is:

```powershell
python -m stages.stage2.stage2_replicates data/shared_effects --replicates 0-9 --scores outputs/stage1_demo/target_scores.csv --out outputs/stage2_fitted_replicates --figures outputs/stage2_figures/fitted_scores
```

Create separate aggregate files for the fitted experts. Do not append them to the supplied-VIPRS `replicate_metrics.csv`.

## 6. Compare the expert sources

Use separate tables or plots for:

- Supplied VIPRS experts
- Teammate-fitted Stage 1 experts

Report:

- Validation incremental R2
- Validation MSE
- Mean paired differences
- Standard errors across replicates
- Training-size trends
- Whether fitted experts consistently improve Stage 2 or the baselines
- Any negative or inconclusive findings

## 7. Freeze the Stage 2 incumbent

Choose one expert source for the Stage 3 incumbent. Record:

- Supplied VIPRS or teammate-fitted target scores
- Exact export path and target-score source
- `n_pc=10`
- `l2_gate=0.01`
- Linear gate
- Linear expert basis
- Seed `0`
- Three-fold/seeded tuning policy
- Use of training-only preprocessing
- Method settings
- Five exact expert column names

Reuse these settings in Stage 3. Do not alternate between expert sources between controller comparisons.

## 8. Final local checks

- Gate weights are finite and sum to one.
- Predictions are finite.
- Test prediction files contain `iid,prediction`.
- Test IDs are the 1,200 withheld-test IDs.
- No test labels or oracle metadata were used locally.
- No `data/` or `outputs/` contents are committed.
- Supplied-VIPRS and fitted-score results are clearly labeled as separate experiments.
