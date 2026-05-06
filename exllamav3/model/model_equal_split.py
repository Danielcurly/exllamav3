"""
Equal Autosplit: pre-calculate balanced module distribution across GPUs.

This module implements a purely arithmetic VRAM estimation that allows finding
the maximum viable context length for a given multi-GPU configuration without
trial-and-error loading.  All calculations operate on model metadata already in
memory (tensor sizes from safetensors index, dimensions from config.json, cached
module properties) — no I/O, no VRAM allocation, no forward passes.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# 2.A — Static weight + cache bytes for a single module
# ---------------------------------------------------------------------------

def calc_module_static(module) -> int:
    """Calculate static bytes (weights + cache) for a module."""
    stc = module.config.stc
    total_bytes = 0
    
    # Identify all keys related to this module
    search_keys = [module.key]
    if hasattr(module, "alt_key") and module.alt_key:
        if isinstance(module.alt_key, str):
            search_keys.append(module.alt_key)
        else:
            search_keys.extend(module.alt_key)

    # For precise index matching, we split the key to identify numeric indices
    # e.g., "model.layers.1" -> ["model", "layers", "1"] (index 1 at pos 2)
    module_key_parts = module.key.split(".")
    indices = [(i, p) for i, p in enumerate(module_key_parts) if p.isdigit()]

    # Collect unique keys from STC matching our criteria
    matched_keys = set()
    for sk in search_keys:
        # Prefix matching via trie to get candidates
        if sk in stc.tensor_file_map:
            matched_keys.add(sk)
        
        prefix = sk if sk.endswith(".") else sk + "."
        candidates = stc.get_tensor_file_map_trie().keys(prefix)
        
        for k in candidates:
            # Precise Filter: ensure all numeric indices in the key match the module's indices
            # This prevents "layers.1." from matching "layers.10."
            k_parts = k.split(".")
            match = True
            for i, val in indices:
                if i >= len(k_parts) or k_parts[i] != val:
                    match = False
                    break
            if match:
                matched_keys.add(k)

    # Sum up sizes of all unique matched tensors
    for k in matched_keys:
        total_bytes += stc.get_tensor_size(k)

    # Recursive check for Cache and Special Recurrent States
    for m in module:
        # Account for KV cache layers if attached
        if hasattr(m, "cache_layers"):
            for cl in m.cache_layers:
                total_bytes += cl.storage_size()
                total_bytes += cl.overhead_size()
        
        # Special case: GatedDeltaNet recurrent state memory
        if m.get_name() == "GatedDeltaNet":
            conv_size = 1 * m.fdim_qkv * m.conv_kernel_size * 2
            rec_size = 1 * m.num_v_heads * m.k_head_dim * m.v_head_dim * 4
            total_bytes += conv_size + rec_size
            
    return total_bytes


# ---------------------------------------------------------------------------
# 2.B — Peak transient overhead during a module's forward pass
# ---------------------------------------------------------------------------

def calc_module_peak_overhead(module, max_chunk_size: int, hidden_size: int) -> int:
    """Calculate peak transient bytes during a module's forward pass.

    The peak occurs when the following coexist:
      1. Activations (hidden state)
      2. Reconstruction buffer for the largest Linear (ephemeral)
      3. MLP/Attn intermediates (gate, up, activation, cache dequant placeholders)

    Scaling rules (from TPAllocator, model_tp_alloc.py L110-120):
      - overhead_to_split:  PER-TOKEN — multiply by max_chunk_size
      - overhead_per_device: FIXED per-device cost — do NOT multiply
      - recons_temp:         FIXED size (2 × in × out for largest Linear)

    Within a component (e.g. Attention), buffers coexist → SUM.
    Across components (e.g. Attention vs MLP), sequential → MAX.

    WARNING — Cache dequant overhead is already inside
    Attention.make_tp_allocation().overhead_to_split (attn.py L1097-1098).
    Do NOT add it separately.
    """

    # 1. Activations: the dummy_state tensor traveling between layers
    activation_bytes = 1 * max_chunk_size * hidden_size * 2  # fp16

    # 2. Collect ALL TPAllocations for this module via ExLlamaV3's own recursion
    try:
        tpa_list = module.make_tp_allocation({})
        # 3. Peak component overhead (MAX across sequential components)
        peak_comp_overhead = 0
        for tpa in tpa_list:
            comp_overhead = (
                tpa.overhead_per_device
                + (tpa.overhead_to_split * max_chunk_size)
            )
            peak_comp_overhead = max(peak_comp_overhead, comp_overhead)
    except Exception:
        # For unknown modules, sum the max overhead of children + core buffers
        peak_comp_overhead = 0
        for m in module.modules:
            # Recursive peak estimation for children (minus their own activation floor)
            try:
                child_peak = calc_module_peak_overhead(m, max_chunk_size, hidden_size) - activation_bytes
                peak_comp_overhead = max(peak_comp_overhead, child_peak)
            except:
                pass
        
        # If it's a GatedDeltaNet, add its core transient buffers (fused op intermediates)
        if module.get_name() == "GatedDeltaNet":
            # Mixed QKV, Z, Beta, G buffers are active during fused op
            peak_comp_overhead += 2 * max_chunk_size * hidden_size * 2 # Conservative 2x activations

    # Real peak: activations + max overhead of any sequential component
    return activation_bytes + peak_comp_overhead


# ---------------------------------------------------------------------------
# 3 — Equal distribution algorithm
# ---------------------------------------------------------------------------

def compute_equal_device_map(
    model,                          # Model (constructed, NOT loaded)
    cache,                          # Cache (MUST be constructed and attached before calling)
    proportions: list[int],         # e.g. [10, 10, 8, 8, 8, 8, 8, 8]
    max_chunk_size: int,            # e.g. 2048
    max_output_size: int = 32,      # max output tokens (logits layer uses min(chunk, this))
    max_output_factor: int = 1,     # extra copies of output tensor for logits upcasting
) -> tuple[dict[int, int | str], list[str]]:
    """Pre-calculate device_map: dict[int, int|str] = {module_idx: device_idx|'cpu'}.

    Returns (device_map, report_lines).
    Only aborts if loading is physically impossible (real overflow).
    """

    # --- STEP 1: Identify active GPUs ---
    active_gpus = [i for i, p in enumerate(proportions) if p > 0]
    num_gpus = len(active_gpus)
    assert num_gpus >= 1, "At least one GPU must have proportion > 0"

    vram_total = {
        i: torch.cuda.get_device_properties(i).total_memory
        for i in active_gpus
    }

    # Measure actually available VRAM on each GPU.
    # mem_get_info() reports free memory at the NVIDIA driver level,
    # which already accounts for the CUDA context and any other processes.
    # touch_device() forces context creation on a fresh start.
    from ..util.memory import touch_device
    vram_available = {}
    cuda_overhead = {}
    for i in active_gpus:
        touch_device(i)
        free, total = torch.cuda.mem_get_info(i)
        vram_available[i] = free
        cuda_overhead[i] = total - free

    # Normalized proportions
    active_proportions = [proportions[i] for i in active_gpus]
    total_prop = sum(active_proportions)

    # --- STEP 2: Calculate per-module costs and group them ---
    import re
    hidden_size = model.config.hidden_size
    stc = model.config.stc

    # 2.A — Map every tensor to the deepest module that "owns" it
    # This ensures zero double-counting across the hierarchy.
    tensor_to_module = {} # tensor_key -> module_idx
    all_tensor_keys = list(stc.tensor_file_map.keys())
    
    # We sort modules by key length (longest first) to find the deepest match easily
    sorted_modules = sorted(enumerate(model.modules), key=lambda x: len(x[1].key), reverse=True)
    
    # Pre-build prefixes to avoid string concat in loop
    module_prefixes = [(idx, m.key + "." if m.key else "") for idx, m in sorted_modules]
    
    # Optimization: use trie if possible, but for now exact match + prefix match
    for t_key in all_tensor_keys:
        found = False
        # 1. Exact match
        for idx, m in sorted_modules:
            if t_key == m.key:
                tensor_to_module[t_key] = idx
                found = True
                break
        if found: continue
        
        # 2. Prefix match (deepest first)
        for idx, prefix in module_prefixes:
            if t_key.startswith(prefix):
                tensor_to_module[t_key] = idx
                found = True
                break

    # 2.B — Define Balance Groups and aggregate costs
    # A group is usually a Layer. We group components by their structural prefix.
    def get_group_id(m_key):
        if not m_key: return "root"
        # Match typical layer patterns: model.layers.1, transformer.h.1, etc.
        # We also catch vision towers and other sub-structures
        match = re.search(r"(.*?\.(?:layers|h|blocks|vision_tower\.layers))\.(\d+)", m_key)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
        return m_key

    # Calculate static cost per module based on owned tensors
    module_owned_static = {idx: 0 for idx in range(len(model.modules))}
    for t_key, idx in tensor_to_module.items():
        module_owned_static[idx] += stc.get_tensor_size(t_key)

    # Account for cache and state memory in modules
    for idx, m in enumerate(model.modules):
        # KV cache
        for sub_m in m:
            if hasattr(sub_m, "cache_layers"):
                for cl in sub_m.cache_layers:
                    module_owned_static[idx] += cl.storage_size() + cl.overhead_size()
            if sub_m.get_name() == "GatedDeltaNet":
                module_owned_static[idx] += (1 * sub_m.fdim_qkv * sub_m.conv_kernel_size * 2) + \
                                            (1 * sub_m.num_v_heads * sub_m.k_head_dim * sub_m.v_head_dim * 4)

    # 2.C — Consolidate into Groups
    group_map = {} # group_id -> {static, peak, is_cpu, first_idx, module_indices}
    ordered_groups = [] # To maintain model sequence
    
    for idx, m in enumerate(model.modules):
        gid = get_group_id(m.key)
        if gid not in group_map:
            group_map[gid] = {
                'id': gid, 'static': 0, 'peak': 0, 
                'is_cpu': m.caps.get("prefer_cpu", False),
                'indices': []
            }
            ordered_groups.append(gid)
        
        g = group_map[gid]
        g['indices'].append(idx)
        g['static'] += module_owned_static[idx]
        
        # Peak is the max of any component in the group (they execute sequentially)
        m_peak = 0
        if m.caps.get("logits_output"):
            effective_output = min(max_chunk_size, max_output_size)
            m_peak = calc_module_peak_overhead(m, effective_output, hidden_size)
            output_dim = m.out_features if hasattr(m, 'out_features') else hidden_size
            m_peak += (max_output_factor - 1) * effective_output * output_dim * 2
        elif not g['is_cpu']:
            m_peak = calc_module_peak_overhead(m, max_chunk_size, hidden_size)
        
        g['peak'] = max(g['peak'], m_peak)

    # --- STEP 3: Precalculate Group budget per GPU ---
    gpu_groups = [group_map[gid] for gid in ordered_groups if not group_map[gid]['is_cpu']]
    num_gpu_groups = len(gpu_groups)

    # Calculate model load and global scaling
    group_costs_mib = [(g['static'] + g['peak']) / 1024**2 for g in gpu_groups]
    total_cost_mib = sum(group_costs_mib)
    total_active_vram_mib = sum(vram_total[i] for i in active_gpus) / 1024**2
    mib_to_ratio_factor = total_prop / total_active_vram_mib

    system_ratios = []
    for ptr, i in enumerate(active_gpus):
        system_mib = (vram_total[i] - vram_available[i]) / 1024**2
        system_ratios.append(system_mib * mib_to_ratio_factor)

    total_model_ratio = total_cost_mib * mib_to_ratio_factor
    total_system_ratio = sum(system_ratios)
    k = (total_system_ratio + total_model_ratio) / total_prop
    
    target_cost_mib = []
    for ptr in range(num_gpus):
        target = (k * active_proportions[ptr] - system_ratios[ptr]) / mib_to_ratio_factor
        target_cost_mib.append(max(0.001, target))
    
    # Re-normalize targets
    sum_targets = sum(target_cost_mib)
    target_cost_mib = [total_cost_mib * r / sum_targets for r in target_cost_mib]

    # Cumulative distribution
    cumulative_costs = []
    curr = 0
    for c in group_costs_mib:
        curr += c
        cumulative_costs.append(curr)
        
    cumulative_targets = []
    curr = 0
    for t in target_cost_mib:
        curr += t
        cumulative_targets.append(curr)
    
    def get_cum(idx): return cumulative_costs[idx] if idx >= 0 else 0
    
    group_budget = []
    last_idx = -1
    for gpu_idx in range(num_gpus - 1):
        target = cumulative_targets[gpu_idx]
        best_idx = last_idx
        min_diff = abs(get_cum(last_idx) - target)
        for i in range(last_idx + 1, num_gpu_groups):
            diff = abs(get_cum(i) - target)
            if diff <= min_diff:
                min_diff = diff
                best_idx = i
            else: break
        group_budget.append(best_idx - last_idx)
        last_idx = best_idx
    group_budget.append(num_gpu_groups - 1 - last_idx)

    # --- STEP 4: Assign Groups to GPUs ---
    device_map = {}
    gpu_static = {i: 0 for i in active_gpus}
    gpu_peak = {i: 0 for i in active_gpus}

    # CPU groups
    for gid in ordered_groups:
        g = group_map[gid]
        if g['is_cpu']:
            for idx in g['indices']:
                device_map[idx] = 'cpu'

    # GPU groups
    gpu_iter = iter(gpu_groups)
    for ptr in range(num_gpus):
        gpu_id = active_gpus[ptr]
        for _ in range(group_budget[ptr]):
            g = next(gpu_iter)
            for idx in g['indices']:
                device_map[idx] = gpu_id
            gpu_static[gpu_id] += g['static']
            gpu_peak[gpu_id] = max(gpu_peak[gpu_id], g['peak'])

    # --- STEP 5: Validation and report ---
    def gpu_total(gpu_id): return gpu_static[gpu_id] + gpu_peak[gpu_id]
    report = []
    overflow = False
    report.append("  === Equal Autosplit Distribution Table ===")
    for ptr, i in enumerate(active_gpus):
        model_total = gpu_total(i)
        smi_total = model_total + cuda_overhead[i]
        pct = (smi_total / vram_total[i]) * 100
        status = "OK" if model_total <= vram_available[i] else "!! OVERFLOW !!"
        if model_total > vram_available[i]: overflow = True
        report.append(
            f"  GPU {i}: {smi_total/(1024**2):,.0f} MiB / {vram_total[i]/(1024**2):,.0f} MiB ({pct:.1f}%) [{status}]"
            f"  [static={gpu_static[i]/(1024**2):,.0f} MiB, peak={gpu_peak[i]/(1024**2):,.0f} MiB, system={cuda_overhead[i]/(1024**2):,.0f} MiB]"
            f"  [{group_budget[ptr]} groups]"
        )

    if overflow:
        raise RuntimeError("Equal autosplit: configuration exceeds available VRAM.\n" + "\n".join(report))

    # --- Debug Logging ---
    print(f"=== Equal Autosplit Group Debug ===")
    print(f"Total Groups: {len(ordered_groups)} (GPU: {num_gpu_groups})")
    print(f"GPU Budget (Groups): {group_budget}")
    summary = {}
    for idx, dev in device_map.items(): summary[dev] = summary.get(dev, 0) + 1
    print(f"Device Map Summary (Total modules per device): {summary}")

    return device_map, report
