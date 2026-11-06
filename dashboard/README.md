# dashboard/ – Digital Twin dashboard (CSE-3)

* `app.py` – Week 2 screen (reads the data files directly, kept as is)
* `dashboard_v2.py` – Part B dashboard, talks to the backend API

## Run
```bash
python -m uvicorn backend.main:app --port 8000        # terminal 1 (repository root)
python -m streamlit run dashboard/dashboard_v2.py     # terminal 2
```
Different API address: `COLDCHAIN_API=http://host:8000`.

## What it shows
* **Overview** – list of shipment units with alert flags, counts (ok / warning / critical), alerts per store
* **Shipment unit** – time slider (replay) + stage buttons; current location, last temperature, RSL, alerts;
  journey timeline, temperature history (4 °C and 8 °C lines, blind spots, "now" marker), RSL checkpoints, event log
