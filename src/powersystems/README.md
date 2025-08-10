# 🔌 Power Systems Simulation Overview

This folder contains **Python implementations of benchmark and custom power system models** used for the experiments in this research.
Each `.py` file provides a **class or function** to initialize the network, modify load conditions, and run **power flow analysis**.

---

## 📂 Available Systems

* **`ieee39.py`** — IEEE 39-bus New England system
* **`ieee118.py`** — IEEE 118-bus test system
* **`ieee145.py`** — IEEE 145-bus system
* **`ieee300.py`** — IEEE 300-bus system
* **`node34.py`** — 34-node distribution network
* **`randomsystem.py`** — Randomly generated test system for stress testing algorithms

Each file builds its network model using [`pandapower`](https://pandapower.readthedocs.io/en/latest/) and provides methods for:

* Loading predefined test cases
* Setting active/reactive load values
* Running power flow simulations
* Retrieving results such as bus voltages and angles

---

## 🧪 Workflow Overview

To run a simulation:

1. **Import** the desired case class from the file
2. **Initialize** the network
3. **Inspect** its default load values
4. **Modify** active and reactive powers as needed
5. **Run** the power flow analysis
6. **Access** and analyze results

---

## 📊 Example — IEEE 300-bus Power Flow

```python
import numpy as np
from powersystems.ieee300 import Case300PF

# Step 1: Initialize the case
case300 = Case300PF()

# Step 2: Diagnose the system (default loads & voltage profile)
case300._diagnose()

# Step 3: Create new load values
num_loads = len(case300.net.load)
p_vec = np.random.normal(case300.net.load["p_mw"].values, 1.0)  # Active power in MW
q_vec = np.random.normal(case300.net.load["q_mvar"].values, 1.0)  # Reactive power in MVar

# Step 4: Apply new loads
case300.set_loads(p_vec, q_vec)

# Step 5: Run power flow
print("Running power flow analysis with modified loads...")
results = case300.run_pf()

# Step 6: Output results
print("Bus voltage magnitudes (p.u.):")
print(results["vm_pu"])
print("Bus voltage angles (degrees):")
print(results["va_degree"])
```

---

## 🎯 Applications

* **Power flow studies** (steady-state analysis)
* **Load sensitivity analysis**
* **Impact assessment** of different injection scenarios
* **Benchmarking** optimization and machine learning algorithms for power systems

---

## ⚙️ Dependencies

* `numpy` ≥ 1.20
* `pandapower` ≥ 2.12

Install with:

```bash
pip install numpy pandapower
```

