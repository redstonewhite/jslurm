from __future__ import annotations

import getpass
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Iterable

DELIM = "\x1f"
GPU_FEATURE_RE = re.compile(
    r"^(?:"
    r"gh[0-9]{1,4}[a-z0-9_+-]*|"
    r"[ahvl][0-9]{1,4}[a-z0-9_+-]*|"
    r"rtx[0-9]{4}[a-z0-9_+-]*|"
    r"gtx[0-9]{4}[a-z0-9_+-]*|"
    r"[0-9]{4}[a-z0-9_+-]*|"
    r"mi[0-9]{2,4}[a-z0-9_+-]*"
    r")$",
    re.IGNORECASE,
)


class SlurmError(RuntimeError):
    """Raised when a Slurm command cannot be executed successfully."""


@dataclass(frozen=True)
class Job:
    job_id: str
    name: str
    state: str
    elapsed: str
    time_limit: str
    nodes: str
    location: str
    partition: str
    gres: str
    cpus: str
    memory: str
    start_time: str
    submit_time: str
    user: str
    priority: str
    feature: str
    sched_nodes: str = "-"


@dataclass(frozen=True)
class Node:
    name: str
    state: str
    partitions: list[str]
    cpus_total: int | None
    cpus_alloc: int | None
    mem_total_mb: int | None
    mem_alloc_mb: int | None
    mem_free_os_mb: int | None
    gres: str
    gres_used: str
    cfg_tres: dict[str, int]
    alloc_tres: dict[str, int]
    features: str

    @property
    def cpus_free(self) -> int | None:
        if self.cpus_total is None or self.cpus_alloc is None:
            return None
        return max(0, self.cpus_total - self.cpus_alloc)

    @property
    def mem_free_sched_mb(self) -> int | None:
        if self.mem_total_mb is None or self.mem_alloc_mb is None:
            return None
        return max(0, self.mem_total_mb - self.mem_alloc_mb)

    @property
    def gpu_total(self) -> int | None:
        return gpu_total(self.cfg_tres, self.gres)

    @property
    def gpu_alloc(self) -> int | None:
        return gpu_alloc(self.alloc_tres, self.gres_used)

    @property
    def gpu_free(self) -> int | None:
        total = self.gpu_total
        alloc = self.gpu_alloc
        if total is None:
            return None
        if alloc is None:
            alloc = 0
        return max(0, total - alloc)

    @property
    def gpu_types(self) -> dict[str, tuple[int, int]]:
        return gpu_type_free_total(self.cfg_tres, self.alloc_tres, self.gres, self.gres_used)

    @property
    def is_allocatable_now(self) -> bool:
        state = normalize_state(self.state)
        bad_markers = {
            "DOWN",
            "DRAIN",
            "DRAINED",
            "DRAINING",
            "FAIL",
            "FAILING",
            "MAINT",
            "NO_RESPOND",
            "NOT_RESPONDING",
            "POWER_DOWN",
            "POWERED_DOWN",
            "REBOOT",
            "RESERVED",
            "UNKNOWN",
        }
        return not any(marker in state for marker in bad_markers)


@dataclass(frozen=True)
class PartitionRow:
    partition: str
    available: str
    time_limit: str
    nodes: str
    state: str
    nodelist: str
    gres: str


@dataclass(frozen=True)
class HistoryJob:
    job_id: str
    name: str
    partition: str
    state: str
    elapsed: str
    cpus: str
    alloc_tres: str
    exit_code: str


def current_user() -> str:
    return os.environ.get("USER") or getpass.getuser()


def run_slurm(args: list[str], timeout: int = 20) -> str:
    try:
        proc = subprocess.run(
            args,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise SlurmError(f"`{args[0]}` was not found. Make sure Slurm commands are in PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise SlurmError(f"`{' '.join(args)}` timed out.") from exc
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        raise SlurmError(f"`{' '.join(args)}` failed: {message}")
    return proc.stdout


def normalize_empty(value: str) -> str:
    value = value.strip()
    if value in {"", "(null)", "N/A", "n/a", "None", "none", "Unknown", "INVALID"}:
        return "-"
    return value


def split_fields(line: str, expected: int) -> list[str]:
    fields = line.rstrip("\n").split(DELIM)
    if len(fields) < expected:
        fields.extend([""] * (expected - len(fields)))
    return fields[:expected]


def squeue_format_field(name: str, width: int = 120, suffix: str = DELIM) -> str:
    return f"{name}:{width}{suffix}"


def squeue_jobs(
    *,
    user: str | None = None,
    all_users: bool = False,
    states: Iterable[str] | None = None,
    partition: str | None = None,
) -> list[Job]:
    fields = [
        "JobID",
        "Name",
        "State",
        "TimeUsed",
        "TimeLimit",
        "NumNodes",
        "ReasonList",
        "Partition",
        "tres-alloc",
        "NumCPUs",
        "MinMemory",
        "StartTime",
        "SubmitTime",
        "UserName",
        "PriorityLong",
        "Feature",
    ]
    format_spec = ",".join(
        squeue_format_field(field, suffix=DELIM if idx < len(fields) - 1 else "")
        for idx, field in enumerate(fields)
    )
    args = ["squeue", "-h", "-O", format_spec]
    if all_users:
        args.append("-a")
    else:
        args.extend(["-u", user or current_user()])
    if states:
        args.extend(["-t", ",".join(states)])
    if partition:
        args.extend(["-p", partition])

    output = run_slurm(args)
    jobs: list[Job] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        values = [normalize_empty(value.strip()) for value in split_fields(line, len(fields))]
        jobs.append(Job(*values))
    return jobs


def parse_key_value_line(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        parts = shlex.split(line, posix=True)
    except ValueError:
        parts = line.split()
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key] = value
    return result


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value in {"N/A", "(null)", "Unknown"}:
        return None
    match = re.match(r"^-?\d+", value)
    if not match:
        return None
    return int(match.group(0))


def parse_list(value: str | None) -> list[str]:
    if not value or value in {"N/A", "(null)"}:
        return []
    return [item for item in value.split(",") if item]


def parse_tres(value: str | None) -> dict[str, int]:
    result: dict[str, int] = {}
    if not value or value in {"N/A", "(null)"}:
        return result
    for item in value.split(","):
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        parsed = parse_int(raw)
        if parsed is not None:
            result[key] = parsed
    return result


def feature_tokens(feature: str | None) -> list[str]:
    if not feature or feature in {"-", "(null)", "N/A"}:
        return []
    return [
        token.lower()
        for token in re.split(r"[^A-Za-z0-9_+-]+", feature)
        if token and token not in {"*", "null", "none"}
    ]


def gpu_model_from_feature(feature: str | None) -> str | None:
    tokens = feature_tokens(feature)
    matches = [token for token in tokens if GPU_FEATURE_RE.match(token)]
    if len(matches) == 1:
        return matches[0]
    return None


def gpu_model_from_node(node: Node) -> str | None:
    typed = [gpu_type for gpu_type in node.gpu_types if gpu_type != "gpu"]
    if len(typed) == 1:
        return typed[0]
    return gpu_model_from_feature(node.features)


def split_hostlist(value: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in value:
        if char == "[":
            depth += 1
        elif char == "]" and depth > 0:
            depth -= 1
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def expand_range_token(token: str) -> list[str]:
    if "-" not in token:
        return [token]
    raw_range, _, raw_step = token.partition(":")
    start, end = raw_range.split("-", 1)
    if not start.isdigit() or not end.isdigit():
        return [token]
    step = parse_int(raw_step) if raw_step else 1
    if step is None or step <= 0:
        step = 1
    width = max(len(start), len(end))
    first = int(start)
    last = int(end)
    if first <= last:
        return [str(value).zfill(width) for value in range(first, last + 1, step)]
    return [str(value).zfill(width) for value in range(first, last - 1, -step)]


def expand_host_component(component: str, limit: int) -> list[str]:
    if "[" not in component:
        return [component]
    start = component.find("[")
    depth = 0
    end = -1
    for idx, char in enumerate(component[start:], start=start):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end = idx
                break
    if end < 0:
        return [component]
    prefix = component[:start]
    body = component[start + 1 : end]
    suffix = component[end + 1 :]
    expanded: list[str] = []
    for token in split_hostlist(body):
        for value in expand_range_token(token):
            for tail in expand_host_component(suffix, limit):
                expanded.append(prefix + value + tail)
                if len(expanded) >= limit:
                    return expanded
    return expanded


def expand_nodelist(nodelist: str | None, limit: int = 512) -> list[str]:
    if not nodelist or nodelist in {"-", "(null)", "N/A"}:
        return []
    if nodelist.startswith("(") and nodelist.endswith(")"):
        return []
    nodes: list[str] = []
    for component in split_hostlist(nodelist):
        for node in expand_host_component(component, max(1, limit - len(nodes))):
            nodes.append(node)
            if len(nodes) >= limit:
                return nodes
    return nodes


def parse_nodes(output: str) -> list[Node]:
    nodes: list[Node] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        data = parse_key_value_line(line)
        name = data.get("NodeName")
        if not name:
            continue
        partitions = parse_list(data.get("Partitions"))
        nodes.append(
            Node(
                name=name,
                state=data.get("State", "-"),
                partitions=partitions,
                cpus_total=parse_int(data.get("CPUTot")),
                cpus_alloc=parse_int(data.get("CPUAlloc")),
                mem_total_mb=parse_int(data.get("RealMemory")),
                mem_alloc_mb=parse_int(data.get("AllocMem")),
                mem_free_os_mb=parse_int(data.get("FreeMem")),
                gres=normalize_empty(data.get("Gres", "-")),
                gres_used=normalize_empty(data.get("GresUsed", "-")),
                cfg_tres=parse_tres(data.get("CfgTRES")),
                alloc_tres=parse_tres(data.get("AllocTRES")),
                features=normalize_empty(data.get("ActiveFeatures") or data.get("AvailableFeatures") or "-"),
            )
        )
    return nodes


def scontrol_nodes() -> list[Node]:
    return parse_nodes(run_slurm(["scontrol", "show", "node", "-o"], timeout=30))


def sinfo_partitions(partition: str | None = None) -> list[PartitionRow]:
    fields = ["%P", "%a", "%l", "%D", "%t", "%N", "%G"]
    args = ["sinfo", "-h", "-o", DELIM.join(fields)]
    if partition:
        args.extend(["-p", partition])
    output = run_slurm(args)
    rows: list[PartitionRow] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        values = [normalize_empty(value) for value in split_fields(line, len(fields))]
        rows.append(PartitionRow(*values))
    return rows


def sacct_history(*, user: str | None, days: int) -> list[HistoryJob]:
    fields = [
        "JobIDRaw",
        "JobName%50",
        "Partition",
        "State",
        "Elapsed",
        "AllocCPUS",
        "AllocTRES%100",
        "ExitCode",
    ]
    args = [
        "sacct",
        "-n",
        "-P",
        "-X",
        "-S",
        f"now-{days}days",
        "-o",
        ",".join(fields),
    ]
    if user:
        args.extend(["-u", user])
    output = run_slurm(args, timeout=30)
    rows: list[HistoryJob] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        values = [normalize_empty(value) for value in line.rstrip("\n").split("|")]
        if len(values) < len(fields):
            values.extend(["-"] * (len(fields) - len(values)))
        rows.append(HistoryJob(*values[: len(fields)]))
    return rows


def normalize_state(state: str) -> str:
    return state.upper().replace("*", "").replace("+", ",")


def gpu_total(cfg_tres: dict[str, int], gres: str) -> int | None:
    typed = [value for key, value in cfg_tres.items() if key.startswith("gres/gpu:")]
    if typed:
        return sum(typed)
    if "gres/gpu" in cfg_tres:
        return cfg_tres["gres/gpu"]
    return gpu_total_from_gres(gres)


def gpu_alloc(alloc_tres: dict[str, int], gres_used: str = "-") -> int | None:
    typed = [value for key, value in alloc_tres.items() if key.startswith("gres/gpu:")]
    if typed:
        return sum(typed)
    if "gres/gpu" in alloc_tres:
        return alloc_tres["gres/gpu"]
    used = gpu_total_from_gres(gres_used)
    if used is not None:
        return used
    return 0


def gpu_total_from_gres(gres: str) -> int | None:
    if not gres or gres == "-":
        return None
    total = 0
    found = False
    for item in gres.split(","):
        item = re.sub(r"\([^)]*\)", "", item.strip())
        if not item.startswith("gpu"):
            continue
        parts = item.split(":")
        count: int | None = None
        for part in reversed(parts[1:]):
            match = re.match(r"^(\d+)", part)
            if match:
                count = int(match.group(1))
                break
        if count is None and parts[0] == "gpu":
            count = 1
        if count is not None:
            found = True
            total += count
    return total if found else None


def gpu_type_free_total(
    cfg_tres: dict[str, int], alloc_tres: dict[str, int], gres: str, gres_used: str = "-"
) -> dict[str, tuple[int, int]]:
    typed_totals: dict[str, int] = {}
    for key, total in cfg_tres.items():
        if key.startswith("gres/gpu:"):
            typed_totals[key.removeprefix("gres/gpu:")] = total

    if not typed_totals:
        typed_totals = gpu_types_from_gres(gres)

    result: dict[str, tuple[int, int]] = {}
    for gpu_type, total in typed_totals.items():
        alloc = alloc_tres.get(f"gres/gpu:{gpu_type}")
        if alloc is None and len(typed_totals) == 1:
            alloc = alloc_tres.get("gres/gpu")
        if alloc is None:
            used_by_type = gpu_types_from_gres(gres_used)
            alloc = used_by_type.get(gpu_type, 0)
        result[gpu_type] = (max(0, total - alloc), total)
    return result


def gpu_types_from_gres(gres: str) -> dict[str, int]:
    if not gres or gres == "-":
        return {}
    result: dict[str, int] = {}
    for item in gres.split(","):
        item = re.sub(r"\([^)]*\)", "", item.strip())
        if not item.startswith("gpu"):
            continue
        parts = item.split(":")
        if len(parts) == 2:
            count = parse_int(parts[1])
            if count is not None:
                result["gpu"] = result.get("gpu", 0) + count
            continue
        if len(parts) >= 3:
            gpu_type = parts[1] or "gpu"
            count = 1
            for part in reversed(parts[2:]):
                parsed = parse_int(part)
                if parsed is not None:
                    count = parsed
                    break
            result[gpu_type] = result.get(gpu_type, 0) + count
    return {key: value for key, value in result.items() if value is not None}
