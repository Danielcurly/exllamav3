"""
Equal autosplit: place the layers of a layer-split model so that every GPU keeps the same amount of
free VRAM, instead of filling one device after another.

The plan is computed up front from metadata (tensor sizes in the safetensors index, the cache layers
attached to the model, each module's TP allocation for transient memory), then enforced by the loader.
After loading, the plan is checked against what the loader measured (allocated memory and the largest
transient of every device), and an optional stress test runs a worst-case forward at the end of the
cache. Both checks fail with EqualSplitHeadroomError when a device ends up with less free VRAM than the
safety margin, so an over-tight split fails at load time rather than on the first long request.

The margin is set per device (one value for all, or one per GPU). The planner subtracts it from what each
device can give, so the objective below maximizes the VRAM left over beyond every device's own margin.

Objective ("water-filling"): find the largest margin M such that the layers, kept in order and split into
contiguous runs, fit on the devices with

    static(device) + max(peak of its layers) <= usable(device) - M

static = weights + cache + recurrent state, peak = the largest transient of any module on the device
(modules run one after another, so their transients don't add up). The greedy fill that packs as many
layers as fit on each device in turn is exact for this feasibility test, and a binary search over M
gives the split that maximizes the free VRAM of the fullest device.
"""

from __future__ import annotations

import re
import torch

from ..constants import PAGE_SIZE

MiB = 1024 ** 2


class EqualSplitHeadroomError(RuntimeError):
    """
    A device ends up (or would end up) with less free VRAM than the safety margin.

    deficit: {device: bytes missing to reach the margin}
    kv_bytes_per_token: total K/V cache bytes per token of context over all devices (0 if unknown), so a
        caller can work out how much context to drop: freeing x tokens raises the equalized margin by
        about x * kv_bytes_per_token / num_devices
    num_devices: devices in the split
    report: lines of the per-device table
    """

    def __init__(
        self,
        message: str,
        deficit: dict[int, int],
        kv_bytes_per_token: int = 0,
        num_devices: int = 1,
        report: list[str] | None = None,
    ):
        super().__init__(message)
        self.deficit = deficit
        self.kv_bytes_per_token = kv_bytes_per_token
        self.num_devices = num_devices
        self.report = report or []

    def suggested_context(self, current: int, granularity: int = PAGE_SIZE) -> int:
        """
        Context length that should clear the largest deficit, rounded down to the granularity
        (at least one step below the current length)
        """
        if not self.kv_bytes_per_token:
            return current - granularity
        need = max(self.deficit.values(), default = 0) * self.num_devices
        drop = -(-need // self.kv_bytes_per_token)
        drop = -(-drop // granularity) * granularity
        return max(granularity, current - max(drop, granularity))


# ---------------------------------------------------------------------------
# Planner core (pure Python, no torch: unit-testable on CPU)
# ---------------------------------------------------------------------------

def split_feasible(
    statics: list[int],
    peaks: list[int],
    usable: list[int],
    margin: int,
) -> list[int] | None:
    """
    Greedy contiguous fill: each group goes on the current device while static + max(peak) stays within
    usable - margin, otherwise on the next device that can take it. Returns the device position of every
    group, or None if the groups don't fit. A device that can't hold even the next group is skipped.
    """
    n = len(usable)
    d = 0
    s = 0
    p = 0
    assignment = []
    for gs, gp in zip(statics, peaks):
        while d < n and s + gs + max(p, gp) > usable[d] - margin:
            d += 1
            s = 0
            p = 0
        if d >= n:
            return None
        s += gs
        p = max(p, gp)
        assignment.append(d)
    return assignment


def plan_split(
    statics: list[int],
    peaks: list[int],
    usable: list[int],
) -> tuple[list[int], int]:
    """
    Split that maximizes the smallest free margin over all devices.

    :param statics:
        Resident bytes of each group, in model order
    :param peaks:
        Largest transient of each group
    :param usable:
        Bytes each device can give to the model (free VRAM now, scaled by its proportion)
    :return:
        (device position of every group, achieved margin in bytes). The margin can be negative: the model
        then does not fit and the value is the shortfall of the fullest device.
    """
    assert len(statics) == len(peaks)
    assert len(usable) >= 1
    if not statics:
        return [], min(usable)

    # Any margin this low fits everything on the first device; the largest usable bounds it from above
    lo = -(sum(statics) + max(peaks) + 1)
    hi = max(usable) + 1
    assert split_feasible(statics, peaks, usable, lo) is not None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if split_feasible(statics, peaks, usable, mid) is not None:
            lo = mid
        else:
            hi = mid
    assignment = split_feasible(statics, peaks, usable, lo)

    # The search margin is a bound; report the margin actually left on the fullest used device
    n = len(usable)
    load_s = [0] * n
    load_p = [0] * n
    for d, gs, gp in zip(assignment, statics, peaks):
        load_s[d] += gs
        load_p[d] = max(load_p[d], gp)
    used = set(assignment)
    achieved = min(usable[d] - load_s[d] - load_p[d] for d in used)
    return assignment, achieved


def max_context_tokens(
    fixed_statics: list[int],
    kv_per_token: list[float],
    peaks: list[int],
    usable: list[int],
    margin: int,
    max_tokens: int,
    granularity: int = PAGE_SIZE,
) -> tuple[int, int]:
    """
    Largest context (a multiple of granularity, at most max_tokens) whose split still leaves margin bytes
    free on every device. The K/V cache grows linearly with the context, so group statics are
    fixed + kv_per_token * tokens. Returns (tokens, margin achieved at that size); tokens is 0 if not even
    one granularity step fits.
    """
    def margin_at(steps):
        t = steps * granularity
        statics = [int(f + k * t) for f, k in zip(fixed_statics, kv_per_token)]
        return plan_split(statics, peaks, usable)[1]

    lo, hi = 0, max_tokens // granularity
    if margin_at(hi) >= margin:
        return hi * granularity, margin_at(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if margin_at(mid) >= margin:
            lo = mid
        else:
            hi = mid
    if lo == 0:
        return 0, margin_at(1)
    return lo * granularity, margin_at(lo)


def margin_bytes(margin_mb, proportions: list[float], devices: list[int]) -> list[int]:
    """
    Safety margin in bytes for each active device. margin_mb is one number (MB, applies to every device)
    or a list indexed by GPU like the proportions (one value per GPU).
    """
    if isinstance(margin_mb, (int, float)):
        return [int(margin_mb * MiB)] * len(devices)
    margin_mb = list(margin_mb)
    assert len(margin_mb) >= len(proportions), \
        f"equal_split_margin_mb lists {len(margin_mb)} values, needs one per GPU ({len(proportions)})"
    return [int(margin_mb[i] * MiB) for i in devices]


def usable_bytes(free: list[int], total: list[int], proportions: list[float]) -> list[int]:
    """
    Bytes each device can give to the model. Proportions are relative to the largest one: the largest
    proportion uses all free VRAM, a device at half of it may fill up to half of its total VRAM (minus
    what it already holds), 0 excludes the device.
    """
    pmax = max(proportions)
    assert pmax > 0, "At least one device must have a proportion > 0"
    out = []
    for f, t, p in zip(free, total, proportions):
        if p <= 0:
            out.append(0)
        else:
            out.append(max(0, int(f - (1 - p / pmax) * t)))
    return out


# ---------------------------------------------------------------------------
# Model metadata -> groups
# ---------------------------------------------------------------------------

_group_re = re.compile(r"(.*?\.(?:layers|h|blocks))\.(\d+)")
_warned = set()


def _group_id(key: str) -> str:
    if not key:
        return "root"
    m = _group_re.search(key)
    return f"{m.group(1)}.{m.group(2)}" if m else key


def _warn_once(tag: str, msg: str):
    if tag not in _warned:
        _warned.add(tag)
        print(f" !! Equal autosplit: {msg}", flush = True)


def _module_overhead(module, tokens: int) -> int:
    """
    Largest transient of any component of the module (components run one after another), from the
    module's own TP allocation. A module whose allocation can't be computed before loading is split into
    its children; the fallback is logged, never silent.
    """
    from ..modules.quant.exl3 import AUTO_RECONSTRUCT_THRESHOLD
    try:
        tpa_list = module.make_tp_allocation({})
    except Exception as e:
        if not module.modules:
            _warn_once(
                f"leaf:{type(module).__name__}",
                f"no transient estimate for {type(module).__name__} ({module.key}): {e!r}; "
                f"covered by the measured check after loading"
            )
            return 0
        _warn_once(
            f"split:{type(module).__name__}",
            f"{type(module).__name__} ({module.key}) has no TP allocation before loading ({e!r}), "
            f"estimating from its submodules"
        )
        return max((_module_overhead(m, tokens) for m in module.modules), default = 0)
    peak = 0
    for tpa in tpa_list:
        ch = max(1, tpa.channels_to_split)
        o = tpa.overhead(tokens, ch, ch)
        # Quantized linears only reconstruct the full weight above this many rows
        if tokens <= AUTO_RECONSTRUCT_THRESHOLD:
            o -= tpa.recons_temp
        peak = max(peak, o)
    return peak


def collect_groups(
    model,
    max_chunk_size: int,
    max_output_size: int,
    max_output_factor: int,
) -> list[dict]:
    """
    Group the model's modules by layer, in model order, with per group:
      fixed: weights + recurrent state (bytes, independent of the context length)
      kv: K/V cache bytes at the size of the attached cache
      peak: largest transient of the group's modules
      cpu: group stays in system RAM
      indices: module indices
    """
    stc = model.config.stc
    hidden_size = model.config.hidden_size

    # Owner of every tensor: the deepest module whose key prefixes it (dict lookups on the key's
    # dotted prefixes, longest first)
    key_to_idx = {}
    for idx, m in enumerate(model.modules):
        for sub in m:
            if sub.key and sub.key not in key_to_idx:
                key_to_idx[sub.key] = idx
            for alt in ([sub.alt_key] if isinstance(getattr(sub, "alt_key", None), str)
                        else (getattr(sub, "alt_key", None) or [])):
                if alt and alt not in key_to_idx:
                    key_to_idx[alt] = idx
    weights = [0] * len(model.modules)
    unowned = 0
    for t_key in stc.tensor_file_map.keys():
        parts = t_key.split(".")
        for n in range(len(parts), 0, -1):
            idx = key_to_idx.get(".".join(parts[:n]))
            if idx is not None:
                weights[idx] += stc.get_tensor_size(t_key)
                break
        else:
            unowned += 1
    if unowned:
        _warn_once("unowned", f"{unowned} tensor(s) in the model files belong to other components "
                              f"(vision, MTP) and are not part of this split")

    groups = []
    by_id = {}
    for idx, m in enumerate(model.modules):
        gid = _group_id(m.key)
        g = by_id.get(gid)
        if g is None:
            g = {"id": gid, "fixed": 0, "kv": 0, "peak": 0, "cpu": bool(m.caps.get("prefer_cpu")), "indices": []}
            by_id[gid] = g
            groups.append(g)
        g["indices"].append(idx)
        g["fixed"] += weights[idx]
        for sub in m:
            for cl in getattr(sub, "cache_layers", []):
                g["kv"] += int(cl.storage_size())
            for rl in getattr(sub, "recurrent_layers", []):
                g["fixed"] += int(rl.storage_size())
        if g["cpu"]:
            continue
        if m.caps.get("logits_output"):
            tokens = min(max_chunk_size, max_output_size)
            out_features = getattr(m, "out_features", hidden_size)
            peak = _module_overhead(m, tokens) + max_output_factor * tokens * out_features * 4
        else:
            tokens = max_chunk_size
            peak = _module_overhead(m, tokens)
        # Hidden state in and out of the module
        peak += 2 * tokens * hidden_size * 2
        g["peak"] = max(g["peak"], peak)
    return groups


def device_state(devices: list[int]) -> tuple[list[int], list[int]]:
    """
    (bytes available, total bytes) per device: driver-level free memory plus what PyTorch holds reserved
    but unallocated. Creates the CUDA context first, so its cost is already deducted.
    """
    from ..util.memory import touch_device, device_mem_info
    free, total = [], []
    for i in devices:
        touch_device(i)
        f, t = device_mem_info(i)
        free.append(f + torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i))
        total.append(t)
    return free, total


# ---------------------------------------------------------------------------
# Model-level entry points
# ---------------------------------------------------------------------------

def _active(proportions: list[float]) -> list[int]:
    devices = [i for i, p in enumerate(proportions) if p > 0]
    assert devices, "At least one device must have a proportion > 0"
    assert max(devices) < torch.cuda.device_count(), \
        f"equal_split_proportions lists {len(proportions)} devices, only {torch.cuda.device_count()} visible"
    return devices


def compute_equal_device_map(
    model,
    proportions: list[float],
    margin_mb,
    max_chunk_size: int,
    max_output_size: int,
    max_output_factor: int,
) -> tuple[dict[int, int | str], dict]:
    """
    Plan the split. margin_mb: one number for every device or one per GPU. Returns
    (device_map {module_idx: device | 'cpu'}, plan). Raises EqualSplitHeadroomError when no split leaves
    every device its margin.
    """
    devices = _active(proportions)
    margin = margin_bytes(margin_mb, proportions, devices)
    free, total = device_state(devices)
    usable = usable_bytes(free, total, [proportions[i] for i in devices])
    budget = [u - m for u, m in zip(usable, margin)]
    baseline = [torch.cuda.memory_allocated(i) for i in devices]

    groups = collect_groups(model, max_chunk_size, max_output_size, max_output_factor)
    gpu_groups = [g for g in groups if not g["cpu"]]
    statics = [g["fixed"] + g["kv"] for g in gpu_groups]
    peaks = [g["peak"] for g in gpu_groups]
    assignment, achieved = plan_split(statics, peaks, budget)

    device_map = {}
    for g in groups:
        if g["cpu"]:
            for idx in g["indices"]:
                device_map[idx] = "cpu"
    n = len(devices)
    est_static = [0] * n
    est_peak = [0] * n
    num_groups = [0] * n
    kv_dev = [0] * n
    for g, d, s in zip(gpu_groups, assignment, statics):
        for idx in g["indices"]:
            device_map[idx] = devices[d]
        est_static[d] += s
        est_peak[d] = max(est_peak[d], g["peak"])
        num_groups[d] += 1
        kv_dev[d] += g["kv"]

    cache = _attached_cache(model)
    kv_per_token = sum(g["kv"] for g in gpu_groups) // cache.max_num_tokens if cache else 0

    plan = {
        "devices": devices,
        "proportions": list(proportions),
        "total": total,
        "usable": usable,
        "baseline_allocated": baseline,
        "est_static": est_static,
        "est_peak": est_peak,
        "num_groups": num_groups,
        "kv_per_token": kv_per_token,
        "context": cache.max_num_tokens if cache else 0,
        "margin": margin,
        "achieved": achieved,
    }

    report = plan_report(plan)
    if achieved < 0:
        deficit = {devices[d]: margin[d] - (usable[d] - est_static[d] - est_peak[d])
                   for d in range(n) if usable[d] - est_static[d] - est_peak[d] < margin[d]}
        raise EqualSplitHeadroomError(
            f"Equal autosplit: no split leaves every GPU its margin "
            f"(best: {-achieved // MiB} MiB short on the fullest)\n" + "\n".join(report),
            deficit, kv_per_token, n, report
        )
    return device_map, plan


def plan_report(plan: dict) -> list[str]:
    lines = [
        f" -- Equal autosplit plan: context {plan['context']:,} tokens, "
        f"spare beyond each GPU's margin {plan['achieved'] // MiB} MiB (fullest GPU)",
        f"    {'GPU':>5}{'layers':>8}{'static':>10}{'peak':>10}{'usable':>10}{'free':>10}{'margin':>10}   (MiB)",
    ]
    for d, i in enumerate(plan["devices"]):
        s, p, u = plan["est_static"][d], plan["est_peak"][d], plan["usable"][d]
        lines.append(f"    {i:>5}{plan['num_groups'][d]:>8}{s // MiB:>10,}{p // MiB:>10,}{u // MiB:>10,}"
                     f"{(u - s - p) // MiB:>10,}{plan['margin'][d] // MiB:>10,}")
    return lines


def _attached_cache(model):
    for ref in model.cache_weakrefs.values():
        c = ref()
        if c is not None:
            return c
    return None


def max_context(
    model,
    cache,
    proportions: list[float],
    margin_mb,
    max_chunk_size: int,
    max_output_size: int,
    max_output_factor: int,
) -> tuple[int, dict]:
    """
    Largest cache size (multiple of the page size, at most the size of the given cache) that the equal
    split can place while leaving every GPU its margin (margin_mb: one number or one per GPU). Call before loading the model, with the
    draft model (if any) already loaded so its VRAM is accounted. The cache's K/V storage scales linearly
    with its size; the attached cache only provides the per-token cost.
    """
    devices = _active(proportions)
    margin = margin_bytes(margin_mb, proportions, devices)
    free, total = device_state(devices)
    usable = usable_bytes(free, total, [proportions[i] for i in devices])
    budget = [u - m for u, m in zip(usable, margin)]
    groups = [g for g in collect_groups(model, max_chunk_size, max_output_size, max_output_factor)
              if not g["cpu"]]
    n0 = cache.max_num_tokens
    tokens, achieved = max_context_tokens(
        [g["fixed"] for g in groups],
        [g["kv"] / n0 for g in groups],
        [g["peak"] for g in groups],
        budget,
        0,
        n0,
    )
    info = {
        "devices": devices,
        "usable": usable,
        "margin": margin,
        "kv_per_token": sum(g["kv"] for g in groups) // n0,
        "achieved": achieved,
    }
    return tokens, info


def check_after_load(model, plan: dict, max_transient: dict[int, int]) -> list[str]:
    """
    Compare the plan with what loading measured: on every device, the VRAM still available for transients
    (driver-free plus PyTorch's reserved-but-unallocated pool) minus the largest transient measured there
    must stay above the margin. Raises EqualSplitHeadroomError otherwise.
    """
    from ..util.memory import device_mem_info
    margin = plan["margin"]
    lines = [
        f" -- Equal autosplit after load (MiB): estimated vs measured",
        f"    {'GPU':>5}{'est.static':>12}{'loaded':>10}{'est.peak':>10}{'meas.peak':>11}{'headroom':>10}{'margin':>9}",
    ]
    deficit = {}
    for d, i in enumerate(plan["devices"]):
        free, _ = device_mem_info(i)
        alloc = torch.cuda.memory_allocated(i)
        reusable = free + torch.cuda.memory_reserved(i) - alloc
        transient = max_transient.get(i, 0)
        headroom = reusable - transient
        loaded = alloc - plan["baseline_allocated"][d]
        flag = "" if headroom >= margin[d] else "  << below margin"
        lines.append(f"    {i:>5}{plan['est_static'][d] // MiB:>12,}{loaded // MiB:>10,}"
                     f"{plan['est_peak'][d] // MiB:>10,}{transient // MiB:>11,}{headroom // MiB:>10,}"
                     f"{margin[d] // MiB:>9,}{flag}")
        if headroom < margin[d]:
            deficit[i] = margin[d] - headroom
    if deficit:
        raise EqualSplitHeadroomError(
            "Equal autosplit: after loading, GPU(s) " + ", ".join(str(i) for i in deficit) +
            " have less than their margin left beyond their largest measured transient\n" +
            "\n".join(lines),
            deficit, plan["kv_per_token"], len(plan["devices"]), lines
        )
    return lines


@torch.inference_mode()
def stress_test(
    model,
    cache,
    devices: list[int],
    margin: list[int],
    max_chunk_size: int,
    max_q_len: int = 1,
    kv_per_token: int = 0,
    extra_params: dict | None = None,
) -> list[str]:
    """
    Worst-case forward at the far end of the cache: a full chunk of prompt ending max_q_len tokens before
    the end of the cache, then a step of max_q_len tokens (a draft verification or plain decode) up to the
    end. Measures the peak allocation on every device on top of what is resident, and requires the margin
    to remain. Writes garbage into the cache and borrows a recurrent-state slot: run it before a generator
    uses the cache (like Model.warmup). Raises EqualSplitHeadroomError on failure, including a CUDA OoM.
    extra_params are added to every forward (e.g. a draft model's draft_verifier_params, so the target
    exports the hidden states the drafter reads, as it does when serving).
    """
    from ..util.memory import device_mem_info, free_mem

    cap = cache.max_num_tokens
    q = max(1, min(max_q_len, cache.max_history + 1))
    chunk = min(max_chunk_size, cap - q)
    assert chunk > 0, "Cache too small for the stress test"
    p1 = cap - q - chunk
    vocab = model.config.vocab_size
    recurrent = len(model.get_recurrent_layers()) > 0

    for i in devices:
        torch.cuda.synchronize(i)
        torch.cuda.reset_peak_memory_stats(i)
    before = {i: torch.cuda.memory_allocated(i) for i in devices}

    states = None
    oom = None
    try:
        if recurrent:
            states = [cache.get_test_state(p1)]
        steps = [(chunk, p1, True), (q, p1 + chunk, False)]
        for n, past, prefill_like in steps:
            params = {
                "attn_mode": "flash_attn",
                "cache": cache,
                "batch_shape": (1, cap),
                "past_len": past,
            }
            if states is not None:
                params["recurrent_states"] = states
                if n > 1 and not prefill_like:
                    params["recurrent_history"] = True
            if prefill_like:
                params["last_tokens_only"] = 1
            if extra_params:
                params.update(extra_params)
            ids = torch.randint(0, vocab, (1, n), dtype = torch.long)
            model.forward(ids, params)
            states = params.get("recurrent_states")
        for i in devices:
            torch.cuda.synchronize(i)
    except Exception as e:
        if e.__class__.__name__ == "OutOfMemoryError" or "out of memory" in str(e):
            oom = e
        else:
            raise
    finally:
        if states:
            for r in states:
                cache.release_state(r)

    lines = [
        f" -- Equal autosplit stress test: {chunk} prompt tokens at {p1:,} + {q} token step at "
        f"{p1 + chunk:,} (cache {cap:,})",
        f"    {'GPU':>5}{'peak':>10}{'headroom':>10}{'margin':>9}   (MiB)",
    ]
    deficit = {}
    for d, i in enumerate(devices):
        peak = max(0, torch.cuda.max_memory_allocated(i) - before[i])
        free, _ = device_mem_info(i)
        headroom = free + torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i) - peak
        if oom is not None:
            headroom = min(headroom, 0)
        flag = "" if headroom >= margin[d] else "  << below margin"
        lines.append(f"    {i:>5}{peak // MiB:>10,}{headroom // MiB:>10,}{margin[d] // MiB:>9,}{flag}")
        if headroom < margin[d]:
            deficit[i] = margin[d] - headroom
    if oom is not None:
        lines.append(f"    out of memory: {str(oom).splitlines()[0][:200]}")
    free_mem()
    if deficit:
        raise EqualSplitHeadroomError(
            "Equal autosplit: stress test at the end of the cache left GPU(s) " +
            ", ".join(str(i) for i in deficit) + " below their margin\n" + "\n".join(lines),
            deficit, kv_per_token, len(devices), lines
        )
    return lines
