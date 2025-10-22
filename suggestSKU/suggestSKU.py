#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import warnings
from datetime import datetime, timedelta, timezone
import math
import re
from typing import Dict, Optional

import urllib3
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.monitor.query import MetricsQueryClient

# ───────────────────────── CONFIG ─────────────────────────
subscription_id = "<your sub id>"
resource_group_name = "< your resource grp name"

# Sizing bands
threshold_high = 80
threshold_low  = 30

# ----- METRICS WINDOW (CONFIGURABLE) -----
# Keep last 7 days, but allow you to change easily.
METRIC_LOOKBACK_MINUTES = 7 * 24 * 60   # last 7 days
# Use "PT5M" for fine grain; "PT1H" approximates portal "Automatic" for 7 days.
METRIC_GRANULARITY     = "PT5M"

# ───────────────────────── Bootstrap ──────────────────────
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

credential = DefaultAzureCredential()
compute_client = ComputeManagementClient(credential, subscription_id)
metrics_client = MetricsQueryClient(credential)

# ───────────────────────── Helpers: Metrics ───────────────
def get_week_metric(resource_id: str, metric_name: str):
    """
    Fetch the Maximum of a VM metric over the configured lookback window
    at the configured granularity.
    """
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=METRIC_LOOKBACK_MINUTES)
        resp = metrics_client.query_resource(
            resource_id,
            metric_names=[metric_name],
            timespan=(start_time, end_time),
            granularity=METRIC_GRANULARITY,
            aggregations=["Maximum"]
        )
        if not resp.metrics or not resp.metrics[0].timeseries:
            return None
        points = resp.metrics[0].timeseries[0].data
        return max([p.maximum for p in points if p.maximum is not None], default=None)
    except Exception as e:
        print(f" Error fetching metric {metric_name}: {e}")
        return None


def get_week_disk_metrics(vm_resource_id: str) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Single-path, per-LUN implementation (no retries, no 'LUN eq *', no OR filter).

    Returns:
      {
        "OS Disk IOPS Consumed Percentage":   { "OS": <max% or None> },
        "Data Disk IOPS Consumed Percentage": { "<lun>": <max% or None>, ... }
      }
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=METRIC_LOOKBACK_MINUTES)

    out: Dict[str, Dict[str, Optional[float]]] = {}

    # --- OS disk (no LUN dimension) ---
    os_map: Dict[str, Optional[float]] = {}
    try:
        resp_os = metrics_client.query_resource(
            vm_resource_id,
            metric_names=["OS Disk IOPS Consumed Percentage"],
            timespan=(start_time, end_time),
            granularity=METRIC_GRANULARITY,
            aggregations=["Maximum"],
            filter=None
        )
        if resp_os.metrics and resp_os.metrics[0].timeseries:
            ts = resp_os.metrics[0].timeseries[0]
            vals = [p.maximum for p in ts.data if getattr(p, "maximum", None) is not None]
            os_map["OS"] = max(vals) if vals else None
    except Exception as e:
        print(f" Error fetching OS disk metric: {e}")
    out["OS Disk IOPS Consumed Percentage"] = os_map

    # --- Data disks (strictly per LUN) ---
    # Read LUNs from the VM model
    data_map: Dict[str, Optional[float]] = {}
    try:
        rid = parse_resource_id(vm_resource_id)
        rg = rid.get("resourcegroups")
        vm_name = rid.get("virtualmachines")
        vm_obj = compute_client.virtual_machines.get(rg, vm_name)
        luns = [str(dd.lun) for dd in (vm_obj.storage_profile.data_disks or []) if dd.lun is not None]
    except Exception as e:
        print(f" Error reading VM data disks: {e}")
        luns = []

    # One query per LUN; assign result to that LUN key regardless of SDK metadata
    for lun in luns:
        try:
            resp = metrics_client.query_resource(
                vm_resource_id,
                metric_names=["Data Disk IOPS Consumed Percentage"],
                timespan=(start_time, end_time),
                granularity=METRIC_GRANULARITY,
                aggregations=["Maximum"],
                filter=f"LUN eq '{lun}'"
            )
            if not resp.metrics or not resp.metrics[0].timeseries:
                data_map[lun] = None
                continue

            # Take max over buckets (Maximum aggregation)
            ts = resp.metrics[0].timeseries[0]
            vals = [p.maximum for p in ts.data if getattr(p, "maximum", None) is not None]
            data_map[lun] = max(vals) if vals else None
        except Exception as e:
            print(f" Error fetching data-disk metric for LUN {lun}: {e}")
            data_map[lun] = None

    out["Data Disk IOPS Consumed Percentage"] = data_map
    return out
# ───────────────────────── Helpers: Disks ─────────────────
def parse_resource_id(resource_id: str) -> dict:
    """
    Minimal ARM ID parser into a dict of segments (lowercased keys).
    Example keys: 'subscriptions','resourcegroups','providers','microsoft.compute','disks'
    """
    parts = [p for p in resource_id.strip('/').split('/')]
    out = {}
    for i in range(0, len(parts), 2):
        k = parts[i].lower()
        v = parts[i+1] if i+1 < len(parts) else None
        out[k] = v
    return out


def build_vm_disk_map(vm) -> dict:
    """
    Build mapping:
      {
        "os": { "diskId","name","rg","sku","prov_iops" },
        "data": { "<lun>": { "diskId","name","rg","sku","prov_iops" }, ... }
      }
    """
    disk_map = {"os": None, "data": {}}

    # OS disk (no LUN)
    osd = getattr(vm.storage_profile, "os_disk", None)
    if osd and getattr(osd, "managed_disk", None) and osd.managed_disk.id:
        disk_map["os"] = {"diskId": osd.managed_disk.id, "name": osd.name}

    # Data disks with LUN
    for dd in (vm.storage_profile.data_disks or []):
        if getattr(dd, "lun", None) is not None and getattr(dd, "managed_disk", None) and dd.managed_disk.id:
            disk_map["data"][str(dd.lun)] = {"diskId": dd.managed_disk.id, "name": dd.name}

    # Enrich via Disks GET
    # OS
    if disk_map["os"]:
        rid = parse_resource_id(disk_map["os"]["diskId"])
        rg = rid.get("resourcegroups")
        name = rid.get("disks")
        try:
            d = compute_client.disks.get(rg, name)
            disk_map["os"]["rg"] = rg
            disk_map["os"]["sku"] = (d.sku.name if getattr(d, "sku", None) else "Unknown")
            disk_map["os"]["prov_iops"] = getattr(d, "disk_iops_read_write", None)
        except Exception:
            pass

    # Data
    for lun, info in list(disk_map["data"].items()):
        rid = parse_resource_id(info["diskId"])
        rg = rid.get("resourcegroups")
        name = rid.get("disks")
        try:
            d = compute_client.disks.get(rg, name)
            info["rg"] = rg
            info["sku"] = (d.sku.name if getattr(d, "sku", None) else "Unknown")
            info["prov_iops"] = getattr(d, "disk_iops_read_write", None)
        except Exception:
            pass

    return disk_map


def suggest_disk_action(disk_iops_pct, current_iops=None, is_premiumv2=False, disk_sku=None):
    """Human-friendly recommendation for a single disk series point."""
    if disk_iops_pct is None:
        return "No data available"

    if disk_iops_pct > threshold_high:
        if is_premiumv2 and current_iops:
            suggested_iops = int(current_iops * 1.2)
            return f"⚠️ High ({disk_iops_pct:.1f}%) - Increase provisioned IOPS to {suggested_iops} (1.2× current)"
        elif current_iops:
            suggested_iops = int(current_iops * 1.2)
            return f"⚠️ High ({disk_iops_pct:.1f}%) - Target IOPS: {suggested_iops} (1.2× current - {disk_sku or 'upgrade SKU'})"
        else:
            return f"⚠️ High ({disk_iops_pct:.1f}%) - Consider upgrading to a higher SKU"

    if disk_iops_pct < threshold_low:
        if is_premiumv2 and current_iops:
            suggested_iops = int(current_iops * 0.8)
            return f"✓ Low ({disk_iops_pct:.1f}%) - Reduce provisioned IOPS to {suggested_iops} (0.8× current)"
        elif current_iops:
            suggested_iops = int(current_iops * 0.8)
            return f"✓ Low ({disk_iops_pct:.1f}%) - Target IOPS: {suggested_iops} (0.8× current - {disk_sku or 'downgrade SKU'})"
        else:
            return f"✓ Low ({disk_iops_pct:.1f}%) - Consider downgrading to a lower SKU"

    return f"✓ Optimal ({disk_iops_pct:.1f}%) - Keep current configuration"

# ───────────────────────── Helpers: VM SKU ────────────────
# Resource SKUs → build a per-region map of {size_name: UncachedDiskIOPS}
_iops_catalog_cache: Dict[str, Dict[str, int]] = {}
_size_info_cache: Dict[str, Dict[str, dict]] = {}

def get_family_key(size_name: str) -> str:
    """
    Normalize to 'family-with-version' so comparisons stay within the same lineage.
    Examples:
      Standard_D2as_v5 -> Standard_Das_v5
      Standard_E8ds_v4 -> Standard_Eds_v4
    """
    m = re.match(r'^(Standard_[A-Za-z]+)(\d+)(.*)$', size_name)
    if m:
        return f"{m.group(1)}{m.group(3)}"
    return ''.join([c for c in size_name if not c.isdigit()])


def get_vm_iops_catalog(location: str) -> Dict[str, int]:
    """
    Query Resource SKUs for the region and extract 'UncachedDiskIOPS' capability for VM sizes.
    Keeps the max if duplicates exist (zones/variants).
    """
    if location in _iops_catalog_cache:
        return _iops_catalog_cache[location]

    catalog: Dict[str, int] = {}
    for sku in compute_client.resource_skus.list(filter=f"location eq '{location}'"):
        if getattr(sku, "resource_type", "").lower() != "virtualmachines":
            continue
        caps = {c.name: c.value for c in (sku.capabilities or [])}
        uncached = caps.get("UncachedDiskIOPS")
        if uncached and str(uncached).isdigit():
            val = int(uncached)
            if sku.name not in catalog or val > catalog[sku.name]:
                catalog[sku.name] = val
    _iops_catalog_cache[location] = catalog
    return catalog


def get_size_info_map(location: str) -> Dict[str, dict]:
    """
    Build a display map with vCPU and RAM per size (for pretty printing).
    Sourced from VirtualMachineSizes (doesn't include IOPS caps).
    """
    if location in _size_info_cache:
        return _size_info_cache[location]

    info = {}
    try:
        sizes = list(compute_client.virtual_machine_sizes.list(location=location))
        for s in sizes:
            info[s.name] = {
                "cores": s.number_of_cores,
                "mem_gb": round((s.memory_in_mb or 0) / 1024.0, 1)
            }
    except Exception:
        pass

    _size_info_cache[location] = info
    return info


def suggest_vm_sku(vm, current_iops_pct: float, allow_cross_family: bool = True):
    """
    VM SKU recommendation using UncachedDiskIOPS caps:
      • Low (≤threshold_low): target = observed * 1.2
        → suggest smallest lower same-family SKU with cap ≥ target; else note 'lowest in family'.
      • High (≥threshold_high): target = current_max * 1.1
        → suggest smallest SKU meeting target (same-family first; else cross-family).
      • Else: keep current.
    """
    try:
        if current_iops_pct is None:
            return "Keep current VM size (insufficient data)"

        location = vm.location
        current_name = vm.hardware_profile.vm_size

        iops_catalog = get_vm_iops_catalog(location)
        size_info = get_size_info_map(location)

        if current_name not in iops_catalog:
            return "No current IOPS info available"

        current_max_iops = iops_catalog[current_name]
        observed_iops = current_max_iops * (current_iops_pct / 100.0)

        fam_key = get_family_key(current_name)
        same_family = {n: v for n, v in iops_catalog.items() if get_family_key(n) == fam_key}

        def fmt(name, iops):
            inf = size_info.get(name, {})
            label = f"{inf.get('cores','?')} vCPU, {inf.get('mem_gb','?')} GB" if inf else "vCPU/GB n/a"
            return f"{name} ({label}, uncached IOPS {iops})"

        # High utilization
        if current_iops_pct >= threshold_high:
            target = int(math.ceil(current_max_iops * 1.10))
            sf_higher = {n: v for n, v in same_family.items() if v >= target}
            if sf_higher:
                pick = min(sf_higher.items(), key=lambda kv: kv[1])
                return (f"⬆️ High uncached IOPS ({current_iops_pct:.1f}%). "
                        f"Current max={current_max_iops}, target≈{target}. Suggest (same family): {fmt(*pick)}")

            if allow_cross_family:
                cross = {n: v for n, v in iops_catalog.items() if v >= target}
                if cross:
                    pick = min(cross.items(), key=lambda kv: kv[1])
                    return (f"⬆️ High uncached IOPS ({current_iops_pct:.1f}%). "
                            f"Current max={current_max_iops}, target≈{target}. "
                            f"No same-family SKU; suggest cross-family: {fmt(*pick)}")

            return (f"⚠️ High uncached IOPS ({current_iops_pct:.1f}%). "
                    f"Target≈{target}, but no SKU in region meets it.")

        # Low utilization
        if current_iops_pct <= threshold_low:
            target = int(math.ceil(observed_iops * 1.20))
            lower_family = {n: v for n, v in same_family.items() if v < current_max_iops}
            candidates = {n: v for n, v in lower_family.items() if v >= target}
            if candidates:
                pick = min(candidates.items(), key=lambda kv: kv[1])
                return (f"⬇️ Low uncached IOPS ({current_iops_pct:.1f}%). "
                        f"Observed≈{int(observed_iops)}, target≈{target}. Suggest (same family): {fmt(*pick)}")

            if not lower_family:
                return (f"ℹ️ Low uncached IOPS ({current_iops_pct:.1f}%). "
                        f"Observed≈{int(observed_iops)}, target≈{target}. "
                        f"Already at lowest IOPS in this family; consider other families.")

            # Lower SKUs exist but none meet ≥ target
            closest = max(lower_family.items(), key=lambda kv: kv[1])
            msg = (f"ℹ️ Low uncached IOPS ({current_iops_pct:.1f}%). "
                   f"Observed≈{int(observed_iops)}, target≈{target}. "
                   f"No lower same-family SKU can provide ≥ target. Closest lower is {fmt(*closest)}.")
            if allow_cross_family:
                cross_lower = {n: v for n, v in iops_catalog.items() if v < current_max_iops and v >= target}
                if cross_lower:
                    xpick = min(cross_lower.items(), key=lambda kv: kv[1])
                    msg += f" Consider cross-family: {fmt(*xpick)}"
                else:
                    msg += " No cross-family lower SKU in region meets target either."
            return msg

        # Middle band
        return f"✓ Optimal ({current_iops_pct:.1f}%) — keep {current_name} (max {current_max_iops} uncached IOPS)"

    except Exception as e:
        return f"Error suggesting VM SKU: {e}"

# ───────────────────────── Main ───────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*90}")
    print(f"Analyzing VMs in {resource_group_name}...")
    print(f"{'='*90}\n")

    for vm in compute_client.virtual_machines.list(resource_group_name):
        vm_resource_id = vm.id
        vm_name = vm.name

        print(f"📊 VM: {vm_name}")
        print(f"   Size: {vm.hardware_profile.vm_size}")

        # VM-level uncached IOPS cap usage
        vm_iops_pct = get_week_metric(vm_resource_id, "VM Uncached IOPS Consumed Percentage")
        print(f"   VM Uncached IOPS %: {vm_iops_pct if vm_iops_pct is not None else 'No data'}")

        if vm_iops_pct is not None:
            vm_action = suggest_vm_sku(vm, vm_iops_pct, allow_cross_family=True)
            print(f"   VM Suggestion: {vm_action}")

        # Disk map (OS + each data LUN) for correct SKU / provisioned IOPS
        disk_map = build_vm_disk_map(vm)

        # Disk metrics (configurable window + granularity)
        vm_disk_stats = get_week_disk_metrics(vm_resource_id)
        print("   [debug] LUN series:", vm_disk_stats.get("Data Disk IOPS Consumed Percentage", {}))
        days_label = int(METRIC_LOOKBACK_MINUTES / 60 / 24)
        print(f"\n   📁 Disk Metrics (Last {days_label} days @ {METRIC_GRANULARITY}):")

        # OS Disk
        os_series = vm_disk_stats.get("OS Disk IOPS Consumed Percentage", {})
        os_val = os_series.get("OS")
        if os_val is not None:
            print(f"   OS Disk IOPS Consumed Percentage:")
            print(f"     OS: {os_val:.2f}%")
            os_info = disk_map.get("os")
            if os_info:
                is_pv2 = bool(os_info.get("sku", "").lower().endswith("v2"))
                suggestion = suggest_disk_action(os_val, os_info.get("prov_iops"),
                                                 is_premiumv2=is_pv2, disk_sku=os_info.get("sku"))
                print(f"     {suggestion}")
                if os_info.get("prov_iops"):
                    iops_label = "Provisioned IOPS" if is_pv2 else "Max IOPS"
                    print(f"     Current {iops_label}: {os_info['prov_iops']} ({os_info.get('sku')})")
        else:
            print("   OS Disk IOPS Consumed Percentage: No data")

        # Data Disks
        print(f"   Data Disk IOPS Consumed Percentage:")
        data_series = vm_disk_stats.get("Data Disk IOPS Consumed Percentage", {})
        # Print each known LUN from the VM definition, even if the metric is missing
        for lun_str, info in sorted(disk_map.get("data", {}).items(), key=lambda kv: int(kv[0])):
            val = data_series.get(lun_str)
            if val is not None:
                print(f"     LUN {lun_str}: {val:.2f}%")
                is_pv2 = bool(info.get("sku", "").lower().endswith("v2"))
                suggestion = suggest_disk_action(val, info.get("prov_iops"),
                                                 is_premiumv2=is_pv2, disk_sku=info.get("sku"))
                print(f"     {suggestion}")
                if info.get("prov_iops"):
                    iops_label = "Provisioned IOPS" if is_pv2 else "Max IOPS"
                    print(f"     Current {iops_label}: {info['prov_iops']} ({info.get('sku')})")
            else:
                print(f"     LUN {lun_str}: No metric in window")

        print(f"\n{'-'*90}\n")