# GAZIANTEP UNIVERSITY
## Faculty of Engineering
### ENG499 Multidisciplinary Project (2026–2027 Fall)

**Project Title:** IoT-Based Digital Twin for Remaining Shelf-Life Prediction and Distribution Planning Optimization in the Cold Chain  
**Supervisor:** Assoc. Prof. Dr. Bülent Haznedar (Department of Computer Engineering)  
**Project Team:** Department of Computer Engineering (CSE) & Department of Industrial Engineering (IE)  

---

## 1. Project Description

Perishable products lose quality rapidly when exposed to temperature excursions in the cold chain; however, conventional distribution planning relies on static expiry dates and First-In-First-Out (FIFO) rules. 

This multidisciplinary project develops an integrated **Digital Twin prototype** that bridges real-time cold chain monitoring and distribution optimization following a First-Expired-First-Out (FEFO) approach:

* **IoT Simulation & Monitoring:** A virtual sensor simulator generates time-stamped temperature and humidity streams (including excursion scenarios across cold room, transport, and retail stages) and publishes data via MQTT to a time-series database.
* **Remaining Shelf-Life (RSL) Prediction:** Kinetic shelf-life models (Arrhenius / Q10) and machine learning algorithms (LSTM / Gradient Boosting) estimate the remaining days of shelf life based on the shipment's thermal history.
* **Distribution Optimization:** A Mixed-Integer Linear Programming (MILP) and heuristic solver addresses the shelf-life-aware allocation and Vehicle Routing Problem with Time Windows (VRPTW).
* **Digital Twin & Evaluation:** A centralized backend API and web dashboard monitor shipment states, generate RSL alerts, present the recommended dispatch plan, and evaluate FEFO against traditional FIFO planning in terms of product waste, routing cost, and service level.

---

## 2. Repository Directory Structure

All source code, simulated datasets, and project documentation are organized under six primary directories:

* `simulator/` : Virtual IoT sensor simulator, temperature-abuse/excursion scenarios, and MQTT data publisher (CSE-1).
* `rsl_model/` : Kinetic (Arrhenius/Q10) and machine learning models for Remaining Shelf-Life (RSL) prediction (CSE-2).
* `optimizer/` : Mathematical formulation, MILP solver, and heuristic algorithms for FEFO allocation and VRPTW (IE-2).
* `dashboard/` : Digital Twin state model, backend API, system integration, and live web dashboard (CSE-3).
* `data/` : Simulated sensor datasets, product kinetic parameters, and distribution scenario instances (IE-1, CSE-1).
* `docs/` : Project brief, literature reviews, process maps, KPI reports, Gantt chart, and weekly meeting minutes (IE-3 & Team).

---

## 3. Team Roles and Work Packages

| Work Package | Role Title | Assigned Student | Primary Responsibilities |
| :--- | :--- | :--- | :--- |
| **CSE-1** | IoT Simulation & Data Lead | *TBD* | Virtual sensor simulator, MQTT data pipeline, time-series database |
| **CSE-2** | RSL Modeling Lead | *TBD* | Kinetic (Arrhenius/Q10) and ML-based RSL prediction models |
| **CSE-3** | Digital Twin & Integration Lead | Basil Elcabiri | Backend API, digital twin state model, web dashboard, repository management |
| **IE-1** | Cold Chain Process & Data Lead | *TBD* | Cold chain process mapping, product kinetic parameters, scenario generation |
| **IE-2** | Optimization Modeling Lead | *TBD* | Shelf-life-aware allocation and VRPTW formulation (MILP & heuristics) |
| **IE-3** | Evaluation & Coordination Lead | *TBD* | KPI analysis (FIFO vs. FEFO), sensitivity analysis, project schedule & minutes |
