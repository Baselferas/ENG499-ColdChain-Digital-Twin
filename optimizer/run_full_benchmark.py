import csv
import glob
import json
import math
import os
import time
from ortools.linear_solver import pywraplp


def hhmm_to_minutes(time_str: str) -> float:
    hours, minutes = map(int, time_str.split(":"))
    return hours * 60.0 + minutes


def minutes_to_hhmm(mins: float) -> str:
    total = int(round(mins))
    h = (total // 60) % 24
    m = total % 60
    return f"{h:02d}:{m:02d}"


def build_route_details(route_node_indices, arrival_times, nodes):
    stops = []
    route_str_parts = ["DEPOT"]
    for idx in route_node_indices[1:-1]:
        arr_min = arrival_times.get(idx, 0.0)
        arr_str = minutes_to_hhmm(arr_min)
        stops.append({
            "node": nodes[idx],
            "arrival_time": arr_str,
            "arrival_minutes": round(arr_min, 2),
        })
        route_str_parts.append(f"{nodes[idx]}({arr_str})")
    route_str_parts.append("DEPOT")
    return " -> ".join(route_str_parts), stops


def evaluate_single_route(route, veh, dist_mat, time_mat, tw_open, tw_close, service_time):
    """Checks time window feasibility and returns (feasible, total_cost, arrival_dict, route_nodes)."""
    if not route:
        return True, 0.0, {}, [0, 0]

    curr_time = hhmm_to_minutes(veh.get("earliest_departure", "05:00"))
    prev_node = 0
    total_dist = 0.0
    arr_times = {}
    route_nodes = [0]

    for s_idx in route:
        curr_node = s_idx + 1
        total_dist += dist_mat[prev_node][curr_node]
        arr = curr_time + time_mat[prev_node][curr_node]
        if arr > tw_close[s_idx] + 1e-6:
            return False, math.inf, {}, []
        start_service = max(arr, tw_open[s_idx])
        arr_times[curr_node] = start_service
        curr_time = start_service + service_time[s_idx]
        route_nodes.append(curr_node)
        prev_node = curr_node

    total_dist += dist_mat[prev_node][0]
    route_nodes.append(0)
    cost = veh["fixed_cost_try"] + veh["cost_per_km_try"] * total_dist
    return True, cost, arr_times, route_nodes


def two_opt_route(route, veh, dist_mat, time_mat, tw_open, tw_close, service_time):
    """Applies 2-opt local search on a single vehicle route while preserving time windows."""
    if len(route) < 3:
        return route
    best_route = list(route)
    _, best_cost, _, _ = evaluate_single_route(
        best_route, veh, dist_mat, time_mat, tw_open, tw_close, service_time
    )
    improved = True
    while improved:
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                candidate = best_route[:i] + best_route[i : j + 1][::-1] + best_route[j + 1 :]
                feas, cand_cost, _, _ = evaluate_single_route(
                    candidate, veh, dist_mat, time_mat, tw_open, tw_close, service_time
                )
                if feas and cand_cost + 1e-6 < best_cost:
                    best_route = candidate
                    best_cost = cand_cost
                    improved = True
    return best_route


def solve_vrptw_rsl_exact(data: dict, alpha: float = 1.0, time_limit_sec: int = 20):
    solver = pywraplp.Solver.CreateSolver("CBC")
    if not solver:
        return None

    solver.SetTimeLimit(time_limit_sec * 1000)

    nodes = data["nodes"]
    num_nodes = len(nodes)
    store_indices = list(range(1, num_nodes))

    stores = data["stores"]
    lots = data["depot_inventory"]
    vehicles = data["vehicles"]
    dist_mat = data["distance_km"]
    time_mat = data["travel_time_min"]

    min_rsl = data.get("product", {}).get("min_rsl_at_delivery_days", 2.0)

    num_stores = len(stores)
    num_lots = len(lots)
    num_vehicles = len(vehicles)

    service_time = [
        s["service_min_fixed"] + s["service_min_per_crate"] * s["order_crates"]
        for s in stores
    ]
    tw_open = [hhmm_to_minutes(s["tw_open"]) for s in stores]
    tw_close = [hhmm_to_minutes(s["tw_close"]) for s in stores]
    demands = [s["order_crates"] for s in stores]

    valid_lot = [
        1 if l.get("rsl_predicted_days", l.get("rsl_true_days", 0.0)) >= min_rsl else 0
        for l in lots
    ]
    rsl_vals = [
        l.get("rsl_predicted_days", l.get("rsl_true_days", 0.0)) for l in lots
    ]
    lot_caps = [l["crates"] for l in lots]

    max_tw_close = max(tw_close) if tw_close else 1440.0
    max_travel = max(max(row) for row in time_mat) if time_mat else 0.0
    max_service = max(service_time) if service_time else 0.0
    big_m = max_tw_close + max_travel + max_service + 60.0

    x = {
        (i, j, v): solver.BoolVar(f"x_{i}_{j}_{v}")
        for v in range(num_vehicles)
        for i in range(num_nodes)
        for j in range(num_nodes)
        if i != j
    }
    u = [solver.BoolVar(f"u_{v}") for v in range(num_vehicles)]
    z = {
        (s, v): solver.BoolVar(f"z_{s}_{v}")
        for s in range(num_stores)
        for v in range(num_vehicles)
    }
    y = {
        (s, l): solver.IntVar(
            0,
            min(demands[s], lot_caps[l]) if valid_lot[l] else 0,
            f"y_{s}_{l}",
        )
        for s in range(num_stores)
        for l in range(num_lots)
    }

    w = {}
    for v in range(num_vehicles):
        dep_min = hhmm_to_minutes(vehicles[v].get("earliest_departure", "05:00"))
        w[0, v] = solver.NumVar(dep_min, max_tw_close, f"w_0_{v}")
        for s in range(num_stores):
            node_idx = s + 1
            w[node_idx, v] = solver.NumVar(tw_open[s], tw_close[s], f"w_{node_idx}_{v}")

    for s in range(num_stores):
        solver.Add(sum(y[s, l] for l in range(num_lots)) == demands[s])

    for l in range(num_lots):
        solver.Add(sum(y[s, l] for s in range(num_stores)) <= lot_caps[l])
        for s in range(num_stores):
            solver.Add(y[s, l] <= demands[s] * valid_lot[l])

    for s in range(num_stores):
        solver.Add(sum(z[s, v] for v in range(num_vehicles)) == 1)
        solver.Add(
            sum(y[s, l] for l in range(num_lots))
            == demands[s] * sum(z[s, v] for v in range(num_vehicles))
        )

    for v in range(num_vehicles):
        cap_v = vehicles[v]["capacity_crates"]
        dep_min = hhmm_to_minutes(vehicles[v].get("earliest_departure", "05:00"))

        solver.Add(sum(x[0, j, v] for j in store_indices) == u[v])
        solver.Add(sum(x[i, 0, v] for i in store_indices) == u[v])

        for s in range(num_stores):
            curr = s + 1
            inflow = sum(x[i, curr, v] for i in range(num_nodes) if i != curr)
            outflow = sum(x[curr, j, v] for j in range(num_nodes) if j != curr)
            solver.Add(inflow == outflow)
            solver.Add(outflow == z[s, v])

        solver.Add(
            sum(demands[s] * z[s, v] for s in range(num_stores)) <= cap_v * u[v]
        )

        for s in range(num_stores):
            j_node = s + 1
            solver.Add(
                dep_min + time_mat[0][j_node] - big_m * (1 - x[0, j_node, v])
                <= w[j_node, v]
            )
            for s2 in range(num_stores):
                if s != s2:
                    i_node = s + 1
                    j2_node = s2 + 1
                    solver.Add(
                        w[i_node, v]
                        + service_time[s]
                        + time_mat[i_node][j2_node]
                        - big_m * (1 - x[i_node, j2_node, v])
                        <= w[j2_node, v]
                    )

    fixed_cost_expr = sum(
        vehicles[v]["fixed_cost_try"] * u[v] for v in range(num_vehicles)
    )
    travel_cost_expr = sum(
        vehicles[v]["cost_per_km_try"] * dist_mat[i][j] * x[i, j, v]
        for v in range(num_vehicles)
        for i in range(num_nodes)
        for j in range(num_nodes)
        if i != j
    )
    fifo_penalty_expr = alpha * sum(
        rsl_vals[l] * y[s, l]
        for s in range(num_stores)
        for l in range(num_lots)
    )

    solver.Minimize(fixed_cost_expr + travel_cost_expr + fifo_penalty_expr)

    start_time = time.time()
    status = solver.Solve()
    solve_time = time.time() - start_time

    status_map = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }

    has_sol = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    result = {
        "method": "EXACT_CBC",
        "instance_id": data.get("instance_id", "unknown"),
        "date": data.get("date"),
        "status": status_map.get(status, str(status)),
        "objective_value": solver.Objective().Value() if has_sol else None,
        "solve_time_sec": solve_time,
        "vehicles_used": int(round(sum(u[v].solution_value() for v in range(num_vehicles)))) if has_sol else 0,
        "routes": [],
        "allocations": [],
    }

    if has_sol:
        store_to_vehicle = {}
        for s in range(num_stores):
            for v in range(num_vehicles):
                if z[s, v].solution_value() > 0.5:
                    store_to_vehicle[stores[s]["store_id"]] = vehicles[v]["vehicle_id"]

        for v in range(num_vehicles):
            if u[v].solution_value() > 0.5:
                route_indices = [0]
                arr_times = {}
                curr = 0
                visited = {0}
                while True:
                    next_node = None
                    for j in range(num_nodes):
                        if (
                            j != curr
                            and j not in visited
                            and (curr, j, v) in x
                            and x[curr, j, v].solution_value() > 0.5
                        ):
                            next_node = j
                            break
                    if next_node is None or next_node == 0:
                        route_indices.append(0)
                        break
                    arr_times[next_node] = w[next_node, v].solution_value()
                    route_indices.append(next_node)
                    visited.add(next_node)
                    curr = next_node
                route_str, stops = build_route_details(route_indices, arr_times, nodes)
                result["routes"].append({
                    "vehicle_id": vehicles[v]["vehicle_id"],
                    "route": route_str,
                    "stops": stops,
                })

        for s in range(num_stores):
            sid = stores[s]["store_id"]
            for l in range(num_lots):
                qty = int(round(y[s, l].solution_value()))
                if qty > 0:
                    result["allocations"].append({
                        "store_id": sid,
                        "vehicle_id": store_to_vehicle.get(sid),
                        "lot_id": lots[l]["lot_id"],
                        "crates": qty,
                        "rsl_pred": rsl_vals[l],
                    })

    return result


def allocate_lots_fifo(stores, lots, min_rsl, alpha, store_to_vehicle):
    remaining_lot_cap = {
        idx: (
            l["crates"]
            if l.get("rsl_predicted_days", l.get("rsl_true_days", 0.0)) >= min_rsl
            else 0
        )
        for idx, l in enumerate(lots)
    }
    valid_lot_order = sorted(
        [idx for idx, l in enumerate(lots) if remaining_lot_cap[idx] > 0],
        key=lambda idx: lots[idx].get(
            "rsl_predicted_days", lots[idx].get("rsl_true_days", 0.0)
        ),
    )

    allocations = []
    fifo_penalty = 0.0
    for store in stores:
        need = store["order_crates"]
        sid = store["store_id"]
        for l_idx in valid_lot_order:
            if need <= 0:
                break
            take = min(need, remaining_lot_cap[l_idx])
            if take > 0:
                remaining_lot_cap[l_idx] -= take
                need -= take
                rsl_val = lots[l_idx].get(
                    "rsl_predicted_days", lots[l_idx].get("rsl_true_days", 0.0)
                )
                fifo_penalty += alpha * rsl_val * take
                allocations.append({
                    "store_id": sid,
                    "vehicle_id": store_to_vehicle.get(sid),
                    "lot_id": lots[l_idx]["lot_id"],
                    "crates": take,
                    "rsl_pred": rsl_val,
                })
    return allocations, fifo_penalty


def solve_vrptw_rsl_heuristic_improved(data: dict, alpha: float = 1.0):
    """Improved Heuristic: TW-feasible Insertion + Route Consolidation (Merge) + 2-Opt Local Search."""
    start_time = time.time()

    nodes = data["nodes"]
    stores = data["stores"]
    lots = data["depot_inventory"]
    vehicles = data["vehicles"]
    dist_mat = data["distance_km"]
    time_mat = data["travel_time_min"]
    min_rsl = data.get("product", {}).get("min_rsl_at_delivery_days", 2.0)

    num_stores = len(stores)
    service_time = [
        s["service_min_fixed"] + s["service_min_per_crate"] * s["order_crates"]
        for s in stores
    ]
    tw_open = [hhmm_to_minutes(s["tw_open"]) for s in stores]
    tw_close = [hhmm_to_minutes(s["tw_close"]) for s in stores]
    demands = [s["order_crates"] for s in stores]

    # Sort vehicles largest capacity first so fewer vehicles are opened
    veh_indices_sorted = sorted(
        range(len(vehicles)),
        key=lambda idx: (-vehicles[idx]["capacity_crates"], vehicles[idx]["fixed_cost_try"]),
    )

    store_order = sorted(
        range(num_stores), key=lambda s: (tw_close[s], tw_open[s], -demands[s])
    )
    vehicle_routes = [[] for _ in vehicles]
    vehicle_loads = [0 for _ in vehicles]

    # Step 1: Time-window feasible cheapest insertion
    for s_idx in store_order:
        best_v = None
        best_pos = None
        best_delta = math.inf

        for v_idx in veh_indices_sorted:
            veh = vehicles[v_idx]
            if vehicle_loads[v_idx] + demands[s_idx] > veh["capacity_crates"]:
                continue
            route = vehicle_routes[v_idx]
            _, old_cost, _, _ = evaluate_single_route(
                route, veh, dist_mat, time_mat, tw_open, tw_close, service_time
            )
            for pos in range(len(route) + 1):
                candidate = route[:pos] + [s_idx] + route[pos:]
                feas, new_cost, _, _ = evaluate_single_route(
                    candidate, veh, dist_mat, time_mat, tw_open, tw_close, service_time
                )
                if feas:
                    delta = new_cost - old_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_v = v_idx
                        best_pos = pos

        if best_v is not None:
            vehicle_routes[best_v].insert(best_pos, s_idx)
            vehicle_loads[best_v] += demands[s_idx]

    # Step 2: Route consolidation (eliminate redundant vehicles by merging routes)
    merged = True
    while merged:
        merged = False
        active_vs = [v for v in veh_indices_sorted if len(vehicle_routes[v]) > 0]
        # Try emptying smaller-load vehicles into other active vehicles
        active_vs_by_load = sorted(active_vs, key=lambda v: vehicle_loads[v])
        for src_v in active_vs_by_load:
            src_stores = list(vehicle_routes[src_v])
            if not src_stores:
                continue
            temp_routes = [list(r) for r in vehicle_routes]
            temp_loads = list(vehicle_loads)
            temp_routes[src_v] = []
            temp_loads[src_v] = 0

            all_inserted = True
            for s_idx in src_stores:
                best_dst = None
                best_pos = None
                best_delta = math.inf
                for dst_v in active_vs:
                    if dst_v == src_v or len(temp_routes[dst_v]) == 0:
                        continue
                    veh = vehicles[dst_v]
                    if temp_loads[dst_v] + demands[s_idx] > veh["capacity_crates"]:
                        continue
                    _, old_c, _, _ = evaluate_single_route(
                        temp_routes[dst_v], veh, dist_mat, time_mat, tw_open, tw_close, service_time
                    )
                    for pos in range(len(temp_routes[dst_v]) + 1):
                        cand = temp_routes[dst_v][:pos] + [s_idx] + temp_routes[dst_v][pos:]
                        feas, new_c, _, _ = evaluate_single_route(
                            cand, veh, dist_mat, time_mat, tw_open, tw_close, service_time
                        )
                        if feas and (new_c - old_c) < best_delta:
                            best_delta = new_c - old_c
                            best_dst = dst_v
                            best_pos = pos
                if best_dst is None:
                    all_inserted = False
                    break
                temp_routes[best_dst].insert(best_pos, s_idx)
                temp_loads[best_dst] += demands[s_idx]

            if all_inserted:
                vehicle_routes = temp_routes
                vehicle_loads = temp_loads
                merged = True
                break

    # Step 3: 2-opt local search on every active vehicle route
    for v_idx in range(len(vehicles)):
        if len(vehicle_routes[v_idx]) >= 3:
            vehicle_routes[v_idx] = two_opt_route(
                vehicle_routes[v_idx],
                vehicles[v_idx],
                dist_mat,
                time_mat,
                tw_open,
                tw_close,
                service_time,
            )

    # Build final schedule & allocations
    store_to_vehicle = {}
    total_route_cost = 0.0
    routes_out = []
    vehicles_used = 0

    for v_idx, veh in enumerate(vehicles):
        route = vehicle_routes[v_idx]
        if not route:
            continue
        vehicles_used += 1
        _, r_cost, arr_times, route_nodes = evaluate_single_route(
            route, veh, dist_mat, time_mat, tw_open, tw_close, service_time
        )
        total_route_cost += r_cost
        for s_idx in route:
            store_to_vehicle[stores[s_idx]["store_id"]] = veh["vehicle_id"]

        route_str, stops = build_route_details(route_nodes, arr_times, nodes)
        routes_out.append({
            "vehicle_id": veh["vehicle_id"],
            "route": route_str,
            "stops": stops,
        })

    allocations, fifo_penalty = allocate_lots_fifo(
        stores, lots, min_rsl, alpha, store_to_vehicle
    )
    solve_time = time.time() - start_time
    total_obj = total_route_cost + fifo_penalty

    return {
        "method": "IMPROVED_HEURISTIC",
        "instance_id": data.get("instance_id", "unknown"),
        "date": data.get("date"),
        "status": "FEASIBLE",
        "objective_value": total_obj,
        "solve_time_sec": solve_time,
        "vehicles_used": vehicles_used,
        "routes": routes_out,
        "allocations": allocations,
    }


def export_plan(plan_dir: str, filename: str, exact_res, heur_res):
    # Choose exact if optimal & available, otherwise improved heuristic
    chosen = (
        exact_res
        if (exact_res is not None and exact_res["status"] == "OPTIMAL")
        else heur_res
    )
    if chosen is None:
        return
    out_path = os.path.join(plan_dir, f"plan_{os.path.splitext(filename)[0]}.json")
    payload = {
        "instance_id": chosen["instance_id"],
        "date": chosen["date"],
        "selected_method": chosen["method"],
        "status": chosen["status"],
        "objective_value": round(chosen["objective_value"], 4)
        if chosen["objective_value"] is not None
        else None,
        "solve_time_sec": round(chosen["solve_time_sec"], 6),
        "vehicles_used": chosen["vehicles_used"],
        "routes": chosen["routes"],
        "allocations": chosen["allocations"],
        "benchmark_comparison": {
            "exact_status": exact_res["status"] if exact_res else None,
            "exact_objective": round(exact_res["objective_value"], 4)
            if (exact_res and exact_res["objective_value"] is not None)
            else None,
            "exact_time_sec": round(exact_res["solve_time_sec"], 6) if exact_res else None,
            "heuristic_objective": round(heur_res["objective_value"], 4)
            if (heur_res and heur_res["objective_value"] is not None)
            else None,
            "heuristic_time_sec": round(heur_res["solve_time_sec"], 6) if heur_res else None,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_full_benchmark():
    all_files = sorted(
        f
        for f in glob.glob("*.json")
        if f.endswith(("_small.json", "_medium.json", "_full.json"))
    )

    if not all_files:
        print("Hiç instance (*.json) dosyası bulunamadı.")
        return

    small_count = sum(1 for f in all_files if f.endswith("_small.json"))
    medium_count = sum(1 for f in all_files if f.endswith("_medium.json"))
    full_count = sum(1 for f in all_files if f.endswith("_full.json"))

    plan_dir = "plans"
    os.makedirs(plan_dir, exist_ok=True)

    print(
        f"Toplam {len(all_files)} dosya çözülüyor -> Small: {small_count} | Medium: {medium_count} | Full: {full_count}"
    )
    print("=" * 112)

    records = []

    for path in all_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Small ve Medium icin Exact (20 sn limit), Full icin ise cok buyuk oldugundan dogrudan Heuristic + 10 sn Exact denemesi
        time_lim = 10 if path.endswith("_full.json") else 20
        exact_res = solve_vrptw_rsl_exact(data, time_limit_sec=time_lim)
        heur_res = solve_vrptw_rsl_heuristic_improved(data)

        export_plan(plan_dir, path, exact_res, heur_res)

        exact_obj = exact_res["objective_value"] if exact_res else None
        heur_obj = heur_res["objective_value"]

        gap_pct = None
        if exact_obj is not None and heur_obj is not None and exact_obj > 0:
            gap_pct = ((heur_obj - exact_obj) / exact_obj) * 100.0

        e_obj_str = f"{exact_obj:.2f}" if exact_obj is not None else "N/A"
        h_obj_str = f"{heur_obj:.2f}" if heur_obj is not None else "N/A"
        gap_str = f"{gap_pct:.2f}%" if gap_pct is not None else "N/A"

        print(
            f"{path:<30} | Exact({exact_res['status']:<8}): {e_obj_str:>9} ({exact_res['solve_time_sec']:>6.2f}s, {exact_res['vehicles_used']}v) "
            f"| Heur: {h_obj_str:>9} ({heur_res['solve_time_sec']:>6.4f}s, {heur_res['vehicles_used']}v) | Gap: {gap_str:>7}"
        )

        records.append({
            "instance": path,
            "date": data.get("date", ""),
            "tier": "full" if path.endswith("_full.json") else ("medium" if path.endswith("_medium.json") else "small"),
            "exact_status": exact_res["status"] if exact_res else "ERR",
            "exact_obj": round(exact_obj, 2) if exact_obj is not None else "",
            "exact_time_sec": round(exact_res["solve_time_sec"], 4) if exact_res else "",
            "exact_vehicles": exact_res["vehicles_used"] if exact_res else "",
            "heur_obj": round(heur_obj, 2) if heur_obj is not None else "",
            "heur_time_sec": round(heur_res["solve_time_sec"], 4),
            "heur_vehicles": heur_res["vehicles_used"],
            "gap_pct": round(gap_pct, 2) if gap_pct is not None else "",
        })

    csv_path = "benchmark_results_v2.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "instance",
            "date",
            "tier",
            "exact_status",
            "exact_obj",
            "exact_time_sec",
            "exact_vehicles",
            "heur_obj",
            "heur_time_sec",
            "heur_vehicles",
            "gap_pct",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("=" * 112)
    print(f"Sonuç tablosu '{csv_path}' dosyasına ve tüm planlar '{plan_dir}/' klasörüne kaydedildi.")


if __name__ == "__main__":
    run_full_benchmark()