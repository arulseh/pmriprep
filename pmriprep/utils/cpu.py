"""Detect CPU limits for an automatic local execution budget."""

import os
from pathlib import Path, PurePosixPath


def _cgroup_cpu_limits(proc: Path = Path("/proc/self")) -> list[int]:
    """Read CPU quotas from visible cgroup ancestors.

    Parameters
    ----------
    proc : pathlib.Path
        Process directory containing cgroup membership and mount information.

    Returns
    -------
    list of int
        Finite CPU quotas rounded down to whole workers, with a minimum of one.
        Missing, unreadable and unlimited quotas contribute no limit.
    """
    try:
        memberships = [line.split(":", 2) for line in (proc / "cgroup").read_text().splitlines()]
        mounts = (proc / "mountinfo").read_text().splitlines()
    except OSError:
        return []

    limits = []
    for mount in mounts:
        fields = mount.split()
        try:
            separator = fields.index("-")
            filesystem = fields[separator + 1]
            controllers = fields[separator + 3].split(",")
            if filesystem != "cgroup2" and not (filesystem == "cgroup" and "cpu" in controllers):
                continue
            root = PurePosixPath(fields[3])
            mountpoint = Path(fields[4])
        except (ValueError, IndexError):
            continue
        for membership in memberships:
            if len(membership) != 3:
                continue
            _, controller, location = membership
            if not ((filesystem == "cgroup2" and not controller)
                    or (filesystem == "cgroup" and "cpu" in controller.split(","))):
                continue
            try:
                relative = PurePosixPath(location).relative_to(root)
            except ValueError:
                # A cgroup namespace may expose only its mounted root.
                relative = PurePosixPath(".")
            if ".." in relative.parts:
                relative = PurePosixPath(".")
            current = mountpoint / relative
            while True:
                try:
                    if filesystem == "cgroup2":
                        quota, period = (current / "cpu.max").read_text().split()
                    else:
                        quota = (current / "cpu.cfs_quota_us").read_text().strip()
                        period = (current / "cpu.cfs_period_us").read_text().strip()
                    quota_value, period_value = int(quota), int(period)
                    if quota_value > 0 and period_value > 0:
                        limits.append(max(1, quota_value // period_value))
                except (OSError, ValueError):
                    pass
                if current == mountpoint:
                    break
                current = current.parent
    return limits


def automatic_cpu_budget() -> tuple[int, str]:
    """Resolve a capped local CPU budget and describe the detected limits.

    Returns
    -------
    count : int
        At most eight workers, bounded by logical CPU count, affinity, Slurm's
        per-task allocation and visible container CPU quotas when available.
    source : str
        Description suitable for the startup log.

    Notes
    -----
    CPU availability does not indicate a user's fair share on an unrestricted
    shared host. The default cap bounds that case; explicit budgets take
    precedence in the CLI resolver.
    """
    limits = {"default cap": 8, "logical CPUs": os.cpu_count() or 1}
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
            if affinity:
                limits["CPU affinity"] = affinity
        except OSError:
            pass
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocation is not None:
        try:
            slots = int(allocation)
            if slots > 0:
                limits["Slurm CPUs per task"] = slots
        except ValueError:
            pass
    quotas = _cgroup_cpu_limits()
    if quotas:
        limits["cgroup CPU quota"] = min(quotas)
    description = ", ".join(f"{name}={value}" for name, value in limits.items())
    return min(limits.values()), f"automatic: {description}"
