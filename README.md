# Azure VM & Disk Right‑Sizing (IOPS‑aware)

This utility analyzes Azure VMs and their managed disks to help you **right‑size** compute and storage based on **IOPS utilization**. It:

- Reads the VM’s **Uncached IOPS cap** (from Resource SKUs) and the VM’s **“VM Uncached IOPS Consumed %”** metric.
- Suggests a **bigger or smaller VM SKU** (same family first) using configurable headroom rules.
- Pulls **per‑disk (OS + each data LUN)** metrics and gives **disk‑level suggestions** (e.g., reduce or increase provisioned IOPS / change SKU).
- Uses a **per‑LUN metrics query** (one call per LUN) because this was the most reliable across environments.

---

## Features

- ✅ VM right‑sizing using `UncachedDiskIOPS` capability per size & region  
- ✅ Per‑LUN **Data Disk IOPS Consumed %** with **MAX** aggregation over a configurable window  
- ✅ OS disk IOPS consumed %  
- ✅ Human‑readable recommendations with current caps (IOPS & SKU shown)

---

## Quick Start

### 1) Prerequisites

- **Python** 3.9+ (3.10/3.11 recommended)
- Azure permissions for read access to:
  - `Microsoft.Compute/virtualMachines`
  2) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip - `Microsoft.Compute/disks`
  - Azure Monitor metrics (Monitoring Reader is sufficient)
- Auth via **DefaultAzureCredential** (any of these work):
  - Logged in with `az login`  
  - Managed identity with proper RBAC  
  - Service principal via environment variables

###
