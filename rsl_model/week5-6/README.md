# `rsl_model/` — Remaining Shelf Life (RSL) Prediction Module (ENG499)

**Owner:** CSE-2 (RSL Modeling Lead — Urwa Osama Abu Hashem)

## 1. Quickstart
```bash
pip install -r requirements.txt
python train_compare.py --data rsl_training_table.csv --output-dir .
python -m unittest test_predict_rsl.py
```

## 2. Team Integration (`predict_rsl`)
```python
from rsl_model.predict_rsl import predict_rsl

res = predict_rsl({
    "su_id": "SU_001",
    "lot_id": "LOT_0104",
    "equiv_age_at_4c_days": 4.93,
    "meas_mean_c": 4.3,
    "checkpoint": "CP3_HUB",
    "store": "STORE_A"
})
# Returns rsl_pred_days, rsl_pred_hours, rsl_lower_p10_days, rsl_upper_p90_days, risk_status
```
