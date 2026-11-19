#!/usr/bin/env python3
"""
ENG499 Multidisciplinary Project - Cold Chain Digital Twin
Role: IE-3 (Evaluation & Project Coordination Lead)
File: evaluate_plan.py

Description:
    Scores any distribution plan (FIFO baseline or IE-2 RSL-based optimization plan)
    against ground-truth remaining shelf life (`rsl_true_days`) and computes the 5 core KPIs:
      1. Waste (spoiled crates, waste rate %, waste cost in TRY)
      2. Crates delivered below minimum RSL threshold (< 2.0 days)
      3. Total distance (km)
      4. Total logistics & waste cost (TRY)
      5. Lateness (late stops, late delivery rate %, total lateness in minutes)

    Also supports direct side-by-side comparison between a baseline FIFO plan
    and an RSL-based plan, exporting a comparison table to CSV and Markdown.

Usage:
    # Score a single plan:
    python evaluate_plan.py --plan path/to/rsl_plan.json

    # Compare FIFO baseline vs. IE-2 RSL-based plan:
    python evaluate_plan.py --fifo path/to/fifo_plan.json --rsl path/to/rsl_plan.json --out comparison.csv

    # Run built-in 6-store sanity-check demo (no external files required):
    python evaluate_plan.py --demo
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Project Cost & Service Parameters (Agreed with IE-1 & IE-2)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "min_rsl_days": 2.0,          # Minimum acceptable RSL at delivery (days)
    "fixed_vehicle_cost": 1500.0, # TRY per dispatched vehicle-day
    "cost_per_km": 14.0,          # TRY per kilometer
    "waste_cost_per_crate": 480.0,# TRY per spoiled crate (product value proxy)
    "expected_shelf_days": 2.0,   # Expected retail dwell time before sale (days)
}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def parse_time_to_minutes(value: Any) -> Optional[float]:
    """Convert 'HH:MM' string or numeric minutes into minutes from midnight."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        parts = text.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) > 2 else 0
        return hours * 60.0 + minutes + seconds / 60.0
    return float(text)


def load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_lot_rsl_lookup(
    plan: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Build a mapping {lot_id: rsl_true_days} from either an external ground-truth
    dictionary or embedded 'lots' / 'inventory' sections inside the plan JSON.
    """
    lookup: Dict[str, float] = {}

    sources = []
    if ground_truth:
        sources.append(ground_truth)
    sources.append(plan)

    for src in sources:
        # Format A: {"lots": [{"lot_id": "L1", "rsl_true_days": 4.2}, ...]}
        for key in ("lots", "inventory", "rsl_ground_truth"):
            items = src.get(key)
            if isinstance(items, list):
                for item in items:
                    lid = item.get("lot_id") or item.get("id")
                    rsl = (
                        item.get("rsl_true_days")
                        if item.get("rsl_true_days") is not None
                        else item.get("true_rsl_days")
                    )
                    if lid is not None and rsl is not None:
                        lookup[str(lid)] = float(rsl)
            elif isinstance(items, dict):
                for lid, val in items.items():
                    if isinstance(val, (int, float)):
                        lookup[str(lid)] = float(val)
                    elif isinstance(val, dict):
                        rsl = val.get("rsl_true_days", val.get("true_rsl_days"))
                        if rsl is not None:
                            lookup[str(lid)] = float(rsl)
    return lookup


# ---------------------------------------------------------------------------
# Core KPI Evaluation Function
# ---------------------------------------------------------------------------
def evaluate_plan(
    plan: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single distribution plan against ground-truth `rsl_true_days`.

    Expected plan JSON structure (flexible field names supported):
    {
      "plan_name": "RSL_Optimized",
      "date": "2025-05-05",
      "routes": [
        {
          "vehicle_id": "V1",
          "distance_km": 42.5,
          "stops": [
            {
              "store_id": "U01",
              "arrival_time": "07:45",        # or minutes from midnight (465)
              "latest_time": "10:00",         # or time_window_end (600)
              "allocations": [
                {"lot_id": "LOT-01", "crates": 15, "rsl_true_days": 5.01}
              ]
            }
          ]
        }
      ]
    }
    """
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    min_rsl = cfg["min_rsl_days"]
    fixed_cost = cfg["fixed_vehicle_cost"]
    cost_per_km = cfg["cost_per_km"]
    waste_unit_cost = cfg["waste_cost_per_crate"]
    shelf_days = cfg["expected_shelf_days"]

    lot_rsl_map = build_lot_rsl_lookup(plan, ground_truth)

    routes = plan.get("routes", [])
    active_vehicles = 0
    total_distance_km = 0.0
    total_stops = 0
    late_stops = 0
    total_lateness_min = 0.0

    total_delivered_crates = 0.0
    crates_below_min_rsl = 0.0
    waste_crates = 0.0

    for route in routes:
        stops = route.get("stops", [])
        route_dist = float(route.get("distance_km", route.get("distance", 0.0)) or 0.0)
        is_used = bool(stops) or route_dist > 0.0

        if is_used:
            active_vehicles += 1
            total_distance_km += route_dist

        for stop in stops:
            total_stops += 1

            # --- Lateness check ---
            arrival = parse_time_to_minutes(
                stop.get("arrival_time", stop.get("arrival_min"))
            )
            latest = parse_time_to_minutes(
                stop.get(
                    "latest_time",
                    stop.get("time_window_end", stop.get("due_time")),
                )
            )
            if arrival is not None and latest is not None and arrival > latest:
                late_stops += 1
                total_lateness_min += arrival - latest

            # --- Allocation, Min-RSL, and Waste check ---
            allocations = stop.get("allocations", stop.get("deliveries", []))
            # Support flat stop format if allocations list is omitted
            if not allocations and "crates" in stop:
                allocations = [stop]

            for alloc in allocations:
                crates = float(alloc.get("crates", alloc.get("quantity", 0.0)) or 0.0)
                if crates <= 0:
                    continue

                lot_id = alloc.get("lot_id")
                rsl_true = alloc.get("rsl_true_days", alloc.get("true_rsl_days"))
                if rsl_true is None and lot_id is not None:
                    rsl_true = lot_rsl_map.get(str(lot_id))

                if rsl_true is None:
                    raise ValueError(
                        f"Missing rsl_true_days for lot '{lot_id}' at store '{stop.get('store_id')}'."
                    )
                rsl_true = float(rsl_true)

                total_delivered_crates += crates

                # KPI: Crates delivered below minimum acceptable RSL
                if rsl_true < min_rsl:
                    crates_below_min_rsl += crates

                # KPI: Spoilage / Waste based on ground-truth RSL
                # - rsl_true <= 0: already spoiled at delivery (100% waste)
                # - 0 < rsl_true < expected_shelf_days: proportional shelf spoilage
                # - rsl_true >= expected_shelf_days: 0% waste
                if rsl_true <= 0.0:
                    waste_crates += crates
                elif rsl_true < shelf_days:
                    spoilage_fraction = 1.0 - (rsl_true / shelf_days)
                    waste_crates += crates * spoilage_fraction

    # --- Derived KPIs ---
    waste_rate_pct = (
        (waste_crates / total_delivered_crates) * 100.0
        if total_delivered_crates > 0
        else 0.0
    )
    pct_below_min_rsl = (
        (crates_below_min_rsl / total_delivered_crates) * 100.0
        if total_delivered_crates > 0
        else 0.0
    )
    late_delivery_rate_pct = (
        (late_stops / total_stops) * 100.0 if total_stops > 0 else 0.0
    )

    fixed_vehicle_cost_try = active_vehicles * fixed_cost
    variable_transport_cost_try = total_distance_km * cost_per_km
    transport_cost_try = fixed_vehicle_cost_try + variable_transport_cost_try
    waste_cost_try = waste_crates * waste_unit_cost
    total_cost_try = transport_cost_try + waste_cost_try

    return {
        "plan_name": plan.get("plan_name", "Unnamed Plan"),
        "date": plan.get("date", "N/A"),
        "active_vehicles": active_vehicles,
        "total_stops": total_stops,
        "total_delivered_crates": round(total_delivered_crates, 2),
        "waste_crates": round(waste_crates, 2),
        "waste_rate_pct": round(waste_rate_pct, 2),
        "waste_cost_try": round(waste_cost_try, 2),
        "crates_below_min_rsl": round(crates_below_min_rsl, 2),
        "pct_below_min_rsl": round(pct_below_min_rsl, 2),
        "total_distance_km": round(total_distance_km, 2),
        "transport_cost_try": round(transport_cost_try, 2),
        "total_cost_try": round(total_cost_try, 2),
        "late_stops": late_stops,
        "late_delivery_rate_pct": round(late_delivery_rate_pct, 2),
        "total_lateness_min": round(total_lateness_min, 2),
    }


# ---------------------------------------------------------------------------
# Comparison Table Builder (FIFO vs. RSL-Based Plan)
# ---------------------------------------------------------------------------
def build_comparison_rows(
    fifo_kpis: Dict[str, Any],
    rsl_kpis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    metrics = [
        ("Delivered Crates", "total_delivered_crates", "crates"),
        ("Waste (Crates)", "waste_crates", "crates"),
        ("Waste Rate (%)", "waste_rate_pct", "%"),
        ("Waste Cost (TRY)", "waste_cost_try", "TRY"),
        ("Crates Below Min RSL (<2.0 d)", "crates_below_min_rsl", "crates"),
        ("Below Min RSL Rate (%)", "pct_below_min_rsl", "%"),
        ("Total Distance (km)", "total_distance_km", "km"),
        ("Transport Cost (TRY)", "transport_cost_try", "TRY"),
        ("Total Cost (Transport + Waste, TRY)", "total_cost_try", "TRY"),
        ("Late Stops", "late_stops", "stops"),
        ("Late Delivery Rate (%)", "late_delivery_rate_pct", "%"),
        ("Total Lateness (min)", "total_lateness_min", "min"),
    ]

    rows = []
    for label, key, unit in metrics:
        f_val = float(fifo_kpis[key])
        r_val = float(rsl_kpis[key])
        delta = round(r_val - f_val, 2)
        if f_val != 0:
            pct_change = f"{((r_val - f_val) / f_val) * 100.0:+.1f}%"
        else:
            pct_change = "0.0%" if r_val == 0 else "N/A"
        rows.append(
            {
                "KPI": label,
                "Unit": unit,
                "FIFO_Baseline": f_val,
                "RSL_Optimized": r_val,
                "Delta (RSL - FIFO)": delta,
                "Change (%)": pct_change,
            }
        )
    return rows


def format_markdown_table(rows: List[Dict[str, Any]]) -> str:
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def save_comparison_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Built-in Sanity-Check Demo (6-Store Small Instance Format)
# ---------------------------------------------------------------------------
def run_demo() -> None:
    """Self-contained sanity check mirroring a 6-store small instance (78 crates)."""
    fifo_demo = {
        "plan_name": "FIFO_Baseline_Demo",
        "date": "2025-05-05",
        "routes": [
            {
                "vehicle_id": "V1",
                "distance_km": 48.0,
                "stops": [
                    {
                        "store_id": "U01",
                        "arrival_time": "07:15",
                        "latest_time": "10:00",
                        "allocations": [{"lot_id": "L-OLD", "crates": 15, "rsl_true_days": 1.4}],
                    },
                    {
                        "store_id": "U02",
                        "arrival_time": "08:05",
                        "latest_time": "10:00",
                        "allocations": [{"lot_id": "L-OLD", "crates": 13, "rsl_true_days": 1.4}],
                    },
                    {
                        "store_id": "U03",
                        "arrival_time": "09:10",
                        "latest_time": "10:00",
                        "allocations": [{"lot_id": "L-MID", "crates": 12, "rsl_true_days": 5.01}],
                    },
                    {
                        "store_id": "U04",
                        "arrival_time": "10:18",
                        "latest_time": "10:00",
                        "allocations": [{"lot_id": "L-MID", "crates": 14, "rsl_true_days": 5.01}],
                    },
                    {
                        "store_id": "U05",
                        "arrival_time": "10:55",
                        "latest_time": "11:00",
                        "allocations": [{"lot_id": "L-FRESH", "crates": 12, "rsl_true_days": 7.03}],
                    },
                    {
                        "store_id": "U06",
                        "arrival_time": "11:30",
                        "latest_time": "12:00",
                        "allocations": [{"lot_id": "L-FRESH", "crates": 12, "rsl_true_days": 7.03}],
                    },
                ],
            }
        ],
    }

    rsl_demo = {
        "plan_name": "IE2_RSL_Plan_Demo",
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
                        "allocations": [{"lot_id": "L-MID", "crates": 15, "rsl_true_days": 5.01}],
                    },
                    {
                        "store_id": "U02",
                        "arrival_time": "07:55",
                        "latest_time": "10:00",
                        "allocations": [{"lot_id": "L-MID", "crates": 13, "rsl_true_days": 5.01}],
                    },
                    {
                        "store_id": "U03",
                        "arrival_time": "08:40",
                        "latest_time": "10:00",
                        "allocations": [{"lot_id": "L-MID", "crates": 12, "rsl_true_days": 5.01}],
                    },
                    {
                        "store_id": "U04",
                        "arrival_time": "09:35",
                        "latest_time": "10:00",
                        "allocations": [{"lot_id": "L-FRESH", "crates": 14, "rsl_true_days": 7.03}],
                    },
                    {
                        "store_id": "U05",
                        "arrival_time": "10:15",
                        "latest_time": "11:00",
                        "allocations": [{"lot_id": "L-FRESH", "crates": 12, "rsl_true_days": 7.03}],
                    },
                    {
                        "store_id": "U06",
                        "arrival_time": "10:50",
                        "latest_time": "12:00",
                        "allocations": [{"lot_id": "L-FRESH", "crates": 12, "rsl_true_days": 7.03}],
                    },
                ],
            }
        ],
    }

    fifo_res = evaluate_plan(fifo_demo)
    rsl_res = evaluate_plan(rsl_demo)
    rows = build_comparison_rows(fifo_res, rsl_res)
    print("=== ENG499 IE-3 Evaluation Script — Sanity-Check Demo (6 Stores, 78 Crates) ===")
    print(format_markdown_table(rows))


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ENG499 IE-3 Plan Evaluation Script (KPI Scorer & FIFO vs. RSL Comparator)"
    )
    parser.add_argument("--plan", type=Path, help="Path to a single plan JSON file to score")
    parser.add_argument("--fifo", type=Path, help="Path to baseline FIFO plan JSON file")
    parser.add_argument("--rsl", type=Path, help="Path to IE-2 RSL-based plan JSON file")
    parser.add_argument(
        "--truth",
        type=Path,
        default=None,
        help="Optional external JSON file containing lot rsl_true_days",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output CSV path for comparison table",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run built-in 6-store sanity-check demo",
    )

    args = parser.parse_args()
    truth_data = load_json_file(args.truth) if args.truth else None

    if args.demo or (not args.plan and not (args.fifo and args.rsl)):
        run_demo()
        return

    if args.plan:
        plan_data = load_json_file(args.plan)
        result = evaluate_plan(plan_data, ground_truth=truth_data)
        print(json.dumps(result, indent=2))
        return

    if args.fifo and args.rsl:
        fifo_kpis = evaluate_plan(load_json_file(args.fifo), ground_truth=truth_data)
        rsl_kpis = evaluate_plan(load_json_file(args.rsl), ground_truth=truth_data)
        rows = build_comparison_rows(fifo_kpis, rsl_kpis)
        print(format_markdown_table(rows))
        if args.out:
            save_comparison_csv(rows, args.out)
            print(f"\nSaved comparison table to: {args.out}")


if __name__ == "__main__":
    main()
