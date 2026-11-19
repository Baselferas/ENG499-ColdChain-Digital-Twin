# ENG499 Cold Chain Digital Twin — IE-3 Evaluation & Project Coordination

**Role:** IE-3 (Evaluation & Project Coordination Lead)  
**Author:** Murat Meriç Kocamaz  
**Milestone:** M2 (Weeks 5–6)

---

## Overview

This directory contains the KPI evaluation tool (`evaluate_plan.py`) and project coordination artifacts for the **ENG499 IoT-Based Digital Twin for Remaining Shelf-Life (RSL) Prediction and Distribution Planning Optimization** project.

While the optimization model (IE-2) builds distribution plans using predicted shelf life (`rsl_predicted_days`), `evaluate_plan.py` scores any plan against the ground-truth remaining shelf life (`rsl_true_days`) from the `optimizer_instances/` dataset to measure real-world operational performance.

---

## Computed KPIs

For any submitted distribution plan (FIFO baseline or RSL-optimized), `evaluate_plan.py` calculates:

1. **Waste:** Spoiled crates, waste rate (%), and waste cost (`480 TRY/crate`).
2. **Minimum RSL Violations:** Crates delivered with `rsl_true_days < 2.0 days` (count and %).
3. **Total Distance:** Total route distance driven across active vehicles (`km`).
4. **Total Cost:** Fixed vehicle cost (`1,500 TRY/vehicle-day`) + variable transport cost (`14 TRY/km`) + waste cost (`480 TRY/crate`).
5. **Lateness:** Number of late stops, late delivery rate (%), and total lateness (`minutes`) beyond store time windows.

---

## Requirements

- **Python 3.8+**
- Uses only Python standard libraries (`argparse`, `csv`, `json`, `pathlib`, `typing`). No external package installation is required.

---

## Usage

### 1. Run Built-In Sanity-Check Demo (6-Store Small Instance)
```bash
python evaluate_plan.py --demo
```

### 2. Score a Single Distribution Plan
```bash
python evaluate_plan.py --plan path/to/rsl_plan.json
```

### 3. Compare Baseline FIFO Plan vs. IE-2 RSL-Based Plan (and Export CSV)
```bash
python evaluate_plan.py --fifo path/to/fifo_plan.json --rsl path/to/rsl_plan.json --out comparison.csv
```

If `rsl_true_days` is stored in a separate instance file rather than embedded in the plan, pass `--truth`:
```bash
python evaluate_plan.py --fifo fifo_plan.json --rsl rsl_plan.json --truth instance_2025-05-05_small.json --out comparison.csv
```

---

## Expected Plan JSON Format

```json
{
  "plan_name": "IE2_RSL_Plan",
  "date": "2025-05-05",
  "routes": [
    {
      "vehicle_id": "V1",
      "distance_km": 44.5,
      "stops": [
        {
          "store_id": "U01",
          "arrival_time": "07:10",
          "latest_time": "10:00",
          "allocations": [
            {
              "lot_id": "L-MID",
              "crates": 15,
              "rsl_true_days": 5.01
            }
          ]
        }
      ]
    }
  ]
}
```

---

## IE-3 M2 Deliverables Included

- `evaluate_plan.py` — Reusable KPI evaluation & FIFO vs. RSL comparison script
- `ENG499_IE3_KPI_Evaluator.xlsx` — Spreadsheet mirror of the KPI evaluator & comparison dashboard
- `ENG499_M2_Presentation_Outline.docx` — 20-minute M2 presentation outline & IE-3 slides
- `eng499_m2_gantt_chart.pdf` — Updated 10-week project Gantt chart (Weeks 5–6 / M2)
- `eng499_m2_meeting_minutes.pdf` — Weeks 5–6 coordination meeting minutes
- `README.md` — Documentation for the team repository