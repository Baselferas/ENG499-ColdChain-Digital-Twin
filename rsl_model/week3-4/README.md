# CSE-2 – RSL Prediction (Week 2)

Kinetic baseline vs Gradient Boosting for Remaining Shelf Life (RSL), with a `predict_rsl()` function.

## Files
| File | Purpose |
|---|---|
| `rsl_common.py` | Feature groups, target, spoilage limit (log10 20,000) |
| `train_compare.py` | Trains K1/K2 kinetic baselines, Ridge, Gradient Boosting on the **same lot_id split**; MAE/RMSE; lab checks; saves model |
| `predict_rsl.py` | `predict_rsl(data)` → `{su_id, lot_id, predicted_days, uncertainty, model}` |
| `models/rsl_models.joblib` | Final models (fit on all rows) |
| `outputs/comparison_mae_rmse.csv` | MAE/RMSE table (hold-out + 5-fold GroupKFold) |
| `outputs/ablation_gb.csv`, `mae_by_checkpoint.csv`, `gb_feature_importance.csv` | Extra analysis |
| `outputs/lab_matched.csv`, `results.json` | Lab matching and all numbers |
| `outputs/M1_Section4_RSL_Prediction_Approach.docx` | Report section 4 |

## Run
```
pip install pandas numpy scikit-learn scipy joblib matplotlib
python3 train_compare.py path/to/rsl_training_table.csv path/to/lab_quality_checks.csv
python3 predict_rsl.py path/to/rsl_training_table.csv      # demo
```
```python
from predict_rsl import predict_rsl
predict_rsl(row_dict)   # -> {"su_id": "SU000001", "lot_id": "L00001", "predicted_days": 6.81, "uncertainty": 1.04, "model": "gradient_boosting"}
```
Input columns: those of `rsl_training_table.csv` listed in `rsl_common.ML_FEATURES` (+ `su_id`/`lot_id`). Output `uncertainty` = 1 SD in days; `predicted_days` ≥ 0.

## Key results (held-out lots, 44 lots)
Linear kinetic baseline MAE 0.854 / RMSE 1.129; Gradient Boosting MAE 0.861 / RMSE 1.127 – no meaningful difference. The remaining error is lot-to-lot variation not visible in process features.

## Limits
- Lab data are bacterial counts, not RSL days: no MAE/RMSE on lab data until a growth rate is agreed with the supervisor.
- Only 3 lab samples exceed the limit. Output field names still to be confirmed with IE-2 / CSE-3.
