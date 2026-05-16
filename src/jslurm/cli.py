from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Callable, Iterable

from . import __version__
from .formatting import (
    Column,
    color_for_state,
    colorize,
    format_mb,
    format_ratio,
    join_short,
    print_table,
    should_color,
    split_csv,
)
from .slurm import (
    HistoryJob,
    Job,
    Node,
    SlurmError,
    current_user,
    sacct_history,
    scontrol_nodes,
    sinfo_partitions,
    squeue_jobs,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def add_common_output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="输出 JSON，便于 jq/脚本继续处理。")
    parser.add_argument("--no-color", action="store_true", help="禁用 ANSI 颜色。")
    parser.add_argument("--watch", type=positive_int, metavar="SEC", help="每 SEC 秒刷新一次。")


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def run_watch(args: argparse.Namespace, render: Callable[[], None]) -> int:
    if not getattr(args, "watch", None):
        render()
        return 0
    try:
        while True:
            print("\x1b[2J\x1b[H", end="")
            render()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 130


def job_gpu_summary(tres_or_gres: str) -> str:
    if not tres_or_gres or tres_or_gres == "-":
        return "-"
    typed: list[str] = []
    generic: str | None = None
    for item in tres_or_gres.split(","):
        item = item.strip()
        if item.startswith("gres/gpu:") and "=" in item:
            key, value = item.split("=", 1)
            typed.append(f"{key.removeprefix('gres/gpu:')}={value}")
        elif item.startswith("gres/gpu="):
            generic = item.split("=", 1)[1]
    if typed:
        return ",".join(typed)
    if generic is not None:
        return f"gpu={generic}"
    if tres_or_gres.startswith("gpu"):
        return tres_or_gres
    return "-"


def job_to_row(job: Job, *, color: bool, long: bool) -> dict[str, str]:
    state = colorize(job.state, color_for_state(job.state), color)
    row = {
        "job_id": job.job_id,
        "name": job.name,
        "state": state,
        "partition": job.partition,
        "location": job.location,
        "gpu": job_gpu_summary(job.gres),
        "cpus": job.cpus,
        "memory": job.memory,
        "nodes": job.nodes,
        "elapsed": job.elapsed,
        "limit": job.time_limit,
        "start": job.start_time,
    }
    if long:
        row.update({"submit": job.submit_time, "user": job.user, "priority": job.priority})
    return row


def add_queue_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-u", "--user", default=current_user(), help="查询指定用户，默认当前用户。")
    parser.add_argument("-a", "--all", action="store_true", help="显示所有用户的作业。")
    parser.add_argument("-p", "--partition", help="只显示指定 partition。")
    parser.add_argument(
        "-t",
        "--states",
        help="Slurm 状态过滤，例如 R,PD 或 RUNNING,PENDING。",
    )
    parser.add_argument("-l", "--long", action="store_true", help="显示提交时间、用户和优先级。")
    add_common_output_flags(parser)


def command_queue(args: argparse.Namespace) -> int:
    states = split_csv(args.states)

    def fetch() -> list[Job]:
        return squeue_jobs(user=args.user, all_users=args.all, states=states, partition=args.partition)

    def render_once() -> None:
        jobs = fetch()
        if args.json:
            print_json([asdict(job) for job in jobs])
            return
        color = should_color(args.no_color)
        rows = [job_to_row(job, color=color, long=args.long) for job in jobs]
        if not rows:
            who = "所有用户" if args.all else args.user
            print(f"没有找到 {who} 的 Slurm 作业。")
            return
        columns = [
            Column("job_id", "JOBID", min_width=6, max_width=14),
            Column("name", "NAME", min_width=8, max_width=24),
            Column("state", "STATE", min_width=7, max_width=13),
            Column("partition", "PART", min_width=5, max_width=12),
            Column("nodes", "N", min_width=1, max_width=4, align="right"),
            Column("location", "NODE/REASON", min_width=10, max_width=34),
            Column("gpu", "GPU", min_width=3, max_width=20),
            Column("cpus", "CPU", min_width=3, max_width=5, align="right"),
            Column("memory", "MEM", min_width=3, max_width=8, align="right"),
            Column("elapsed", "TIME", min_width=4, max_width=12, align="right"),
            Column("limit", "LIMIT", min_width=5, max_width=12, align="right"),
            Column("start", "START", min_width=5, max_width=19),
        ]
        if args.long:
            columns.extend(
                [
                    Column("submit", "SUBMIT", min_width=6, max_width=19),
                    Column("user", "USER", min_width=4, max_width=14),
                    Column("priority", "PRIO", min_width=4, max_width=10, align="right"),
                ]
            )
        print_table(rows, columns)

    return run_watch(args, render_once)


def node_gpu_type_summary(node: Node) -> str:
    typed = node.gpu_types
    if typed:
        return ", ".join(f"{name} {free}/{total}" for name, (free, total) in sorted(typed.items()))
    total = node.gpu_total
    free = node.gpu_free
    if total is None:
        return "-"
    return f"gpu {format_ratio(free, total)}"


def node_to_dict(node: Node) -> dict[str, Any]:
    data = asdict(node)
    data.update(
        {
            "cpus_free": node.cpus_free,
            "mem_free_sched_mb": node.mem_free_sched_mb,
            "gpu_total": node.gpu_total,
            "gpu_alloc": node.gpu_alloc,
            "gpu_free": node.gpu_free,
            "gpu_types": {
                key: {"free": value[0], "total": value[1]}
                for key, value in node.gpu_types.items()
            },
            "allocatable_now": node.is_allocatable_now,
        }
    )
    return data


def node_to_row(node: Node, *, color: bool) -> dict[str, str]:
    state = colorize(node.state, color_for_state(node.state), color)
    return {
        "name": node.name,
        "partition": ",".join(node.partitions) or "-",
        "gpu": format_ratio(node.gpu_free, node.gpu_total),
        "gpu_type": node_gpu_type_summary(node),
        "cpu": format_ratio(node.cpus_free, node.cpus_total),
        "mem": f"{format_mb(node.mem_free_sched_mb)}/{format_mb(node.mem_total_mb)}",
        "free_mem": format_mb(node.mem_free_os_mb),
        "state": state,
        "features": node.features,
    }


def filter_nodes(
    nodes: Iterable[Node],
    *,
    partition: str | None = None,
    gpu_type: str | None = None,
    min_gpus: int = 1,
    available_only: bool = False,
    gpu_only: bool = False,
    states: list[str] | None = None,
) -> list[Node]:
    result: list[Node] = []
    wanted_states = [state.upper() for state in states or []]
    for node in nodes:
        if partition and partition not in node.partitions:
            continue
        if wanted_states and not any(state in node.state.upper() for state in wanted_states):
            continue
        if gpu_type:
            if gpu_type not in node.gpu_types and gpu_type not in node.gres:
                continue
        if gpu_only and (node.gpu_total or 0) <= 0:
            continue
        if available_only:
            if not node.is_allocatable_now:
                continue
            if (node.gpu_free or 0) < min_gpus:
                continue
            if gpu_type and (node.gpu_types.get(gpu_type, (0, 0))[0] < min_gpus):
                continue
        result.append(node)
    return sorted(result, key=lambda item: (",".join(item.partitions), item.name))


def add_node_filter_args(parser: argparse.ArgumentParser, *, include_avail_defaults: bool = False) -> None:
    parser.add_argument("-p", "--partition", help="只显示指定 partition 的节点。")
    parser.add_argument("-g", "--gpu-type", help="只显示包含指定 GPU 类型的节点，例如 a100。")
    parser.add_argument("--min-gpus", type=int, default=1, help="可用 GPU 数量下限，默认 1。")
    parser.add_argument(
        "--states",
        help="状态过滤，例如 IDLE,MIXED,DRAIN。",
    )
    if include_avail_defaults:
        parser.add_argument("--all", action="store_true", help="显示所有 GPU 节点，而不只显示立即可用节点。")
    else:
        parser.add_argument("--available", action="store_true", help="只显示当前可调度且有空闲 GPU 的节点。")
    add_common_output_flags(parser)


def command_avail(args: argparse.Namespace) -> int:
    states = split_csv(args.states)

    def render_once() -> None:
        nodes = filter_nodes(
            scontrol_nodes(),
            partition=args.partition,
            gpu_type=args.gpu_type,
            min_gpus=args.min_gpus,
            available_only=not args.all,
            gpu_only=True,
            states=states,
        )
        if args.json:
            print_json([node_to_dict(node) for node in nodes])
            return
        if not nodes:
            print("没有找到满足条件的立即可用 GPU 资源。")
            return
        color = should_color(args.no_color)
        rows = [node_to_row(node, color=color) for node in nodes]
        print_table(
            rows,
            [
                Column("name", "NODE", min_width=5, max_width=18),
                Column("partition", "PART", min_width=5, max_width=18),
                Column("gpu", "GPU", min_width=5, max_width=8, align="right"),
                Column("gpu_type", "GPU TYPE", min_width=8, max_width=28),
                Column("cpu", "CPU", min_width=5, max_width=9, align="right"),
                Column("mem", "MEM", min_width=7, max_width=13, align="right"),
                Column("free_mem", "OSFREE", min_width=6, max_width=8, align="right"),
                Column("state", "STATE", min_width=5, max_width=16),
                Column("features", "FEATURES", min_width=5, max_width=26),
            ],
        )

    return run_watch(args, render_once)


def command_nodes(args: argparse.Namespace) -> int:
    states = split_csv(args.states)

    def render_once() -> None:
        nodes = filter_nodes(
            scontrol_nodes(),
            partition=args.partition,
            gpu_type=args.gpu_type,
            min_gpus=args.min_gpus,
            available_only=args.available,
            gpu_only=False,
            states=states,
        )
        if args.json:
            print_json([node_to_dict(node) for node in nodes])
            return
        if not nodes:
            print("没有找到满足条件的节点。")
            return
        color = should_color(args.no_color)
        rows = [node_to_row(node, color=color) for node in nodes]
        print_table(
            rows,
            [
                Column("name", "NODE", min_width=5, max_width=18),
                Column("partition", "PART", min_width=5, max_width=20),
                Column("gpu", "GPU", min_width=5, max_width=8, align="right"),
                Column("gpu_type", "GPU TYPE", min_width=8, max_width=28),
                Column("cpu", "CPU", min_width=5, max_width=9, align="right"),
                Column("mem", "MEM", min_width=7, max_width=13, align="right"),
                Column("free_mem", "OSFREE", min_width=6, max_width=8, align="right"),
                Column("state", "STATE", min_width=5, max_width=16),
                Column("features", "FEATURES", min_width=5, max_width=32),
            ],
        )

    return run_watch(args, render_once)


def command_partitions(args: argparse.Namespace) -> int:
    def render_once() -> None:
        rows = sinfo_partitions(args.partition)
        if args.json:
            print_json([asdict(row) for row in rows])
            return
        if not rows:
            print("没有找到 partition 信息。")
            return
        color = should_color(args.no_color)
        table_rows = []
        for row in rows:
            table_rows.append(
                {
                    "partition": row.partition,
                    "available": row.available,
                    "limit": row.time_limit,
                    "nodes": row.nodes,
                    "state": colorize(row.state, color_for_state(row.state), color),
                    "gres": row.gres,
                    "nodelist": row.nodelist,
                }
            )
        print_table(
            table_rows,
            [
                Column("partition", "PART", min_width=5, max_width=18),
                Column("available", "UP", min_width=2, max_width=4),
                Column("limit", "LIMIT", min_width=5, max_width=12),
                Column("nodes", "NODES", min_width=5, max_width=6, align="right"),
                Column("state", "STATE", min_width=5, max_width=12),
                Column("gres", "GRES", min_width=4, max_width=24),
                Column("nodelist", "NODELIST", min_width=8, max_width=44),
            ],
        )

    return run_watch(args, render_once)


def command_why(args: argparse.Namespace) -> int:
    def render_once() -> None:
        jobs = squeue_jobs(
            user=args.user,
            all_users=args.all,
            states=["PD", "PENDING"],
            partition=args.partition,
        )
        grouped: dict[str, list[Job]] = defaultdict(list)
        for job in jobs:
            grouped[job.location].append(job)
        data = [
            {
                "reason": reason,
                "count": len(items),
                "jobs": [item.job_id for item in items],
                "examples": [item.name for item in items[:3]],
                "starts": [item.start_time for item in items if item.start_time != "-"],
            }
            for reason, items in grouped.items()
        ]
        data.sort(key=lambda item: (-item["count"], item["reason"]))
        if args.json:
            print_json(data)
            return
        if not data:
            who = "所有用户" if args.all else args.user
            print(f"没有找到 {who} 的 pending 作业。")
            return
        rows = []
        for item in data:
            starts = [start for start in item["starts"] if start and start != "N/A"]
            rows.append(
                {
                    "reason": item["reason"],
                    "count": item["count"],
                    "jobs": join_short(item["jobs"], 5),
                    "examples": join_short(item["examples"], 3),
                    "start": min(starts) if starts else "-",
                }
            )
        print_table(
            rows,
            [
                Column("reason", "REASON", min_width=8, max_width=26),
                Column("count", "COUNT", min_width=5, max_width=6, align="right"),
                Column("jobs", "JOBS", min_width=6, max_width=32),
                Column("examples", "EXAMPLES", min_width=8, max_width=32),
                Column("start", "EARLIEST START", min_width=8, max_width=19),
            ],
        )

    return run_watch(args, render_once)


def add_why_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-u", "--user", default=current_user(), help="查询指定用户，默认当前用户。")
    parser.add_argument("-a", "--all", action="store_true", help="显示所有用户 pending 作业的原因。")
    parser.add_argument("-p", "--partition", help="只显示指定 partition。")
    add_common_output_flags(parser)


def history_to_row(job: HistoryJob, *, color: bool) -> dict[str, str]:
    return {
        "job_id": job.job_id,
        "name": job.name,
        "partition": job.partition,
        "state": colorize(job.state, color_for_state(job.state), color),
        "elapsed": job.elapsed,
        "cpus": job.cpus,
        "tres": job.alloc_tres,
        "exit": job.exit_code,
    }


def command_history(args: argparse.Namespace) -> int:
    def render_once() -> None:
        rows = sacct_history(user=args.user, days=args.days)
        if args.json:
            print_json([asdict(row) for row in rows])
            return
        if not rows:
            print("没有找到近期 sacct 记录。")
            return
        color = should_color(args.no_color)
        table_rows = [history_to_row(row, color=color) for row in rows]
        print_table(
            table_rows,
            [
                Column("job_id", "JOBID", min_width=6, max_width=14),
                Column("name", "NAME", min_width=8, max_width=24),
                Column("partition", "PART", min_width=5, max_width=12),
                Column("state", "STATE", min_width=7, max_width=18),
                Column("elapsed", "ELAPSED", min_width=7, max_width=12, align="right"),
                Column("cpus", "CPU", min_width=3, max_width=5, align="right"),
                Column("tres", "ALLOC TRES", min_width=8, max_width=36),
                Column("exit", "EXIT", min_width=4, max_width=8),
            ],
        )

    return run_watch(args, render_once)


def add_history_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-u", "--user", default=current_user(), help="查询指定用户，默认当前用户。")
    parser.add_argument("--days", type=positive_int, default=7, help="查询最近 N 天，默认 7。")
    add_common_output_flags(parser)


def build_parser(prog: str = "jslurm") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Small CLI helpers for Slurm.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    queue = subparsers.add_parser("queue", aliases=["q"], help="显示当前用户或指定用户的作业。")
    add_queue_args(queue)
    queue.set_defaults(func=command_queue)

    avail = subparsers.add_parser("avail", aliases=["a"], help="显示立即可用的 GPU 节点资源。")
    add_node_filter_args(avail, include_avail_defaults=True)
    avail.set_defaults(func=command_avail)

    nodes = subparsers.add_parser("nodes", aliases=["n"], help="显示节点资源概览。")
    add_node_filter_args(nodes)
    nodes.set_defaults(func=command_nodes)

    partitions = subparsers.add_parser("part", aliases=["p"], help="显示 partition/sinfo 概览。")
    partitions.add_argument("-p", "--partition", help="只显示指定 partition。")
    add_common_output_flags(partitions)
    partitions.set_defaults(func=command_partitions)

    why = subparsers.add_parser("why", aliases=["w"], help="汇总 pending 作业原因。")
    add_why_args(why)
    why.set_defaults(func=command_why)

    history = subparsers.add_parser("hist", aliases=["h"], help="显示近期 sacct 历史。")
    add_history_args(history)
    history.set_defaults(func=command_history)

    return parser


def run_with_errors(func: Callable[[argparse.Namespace], int], args: argparse.Namespace) -> int:
    try:
        return func(args)
    except SlurmError as exc:
        print(f"jslurm: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_with_errors(args.func, args)


def main_queue(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jqueue", description="显示 Slurm 作业队列。")
    add_queue_args(parser)
    args = parser.parse_args(argv)
    return run_with_errors(command_queue, args)


def main_avail(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="javail", description="显示立即可用的 GPU 资源。")
    add_node_filter_args(parser, include_avail_defaults=True)
    args = parser.parse_args(argv)
    return run_with_errors(command_avail, args)


def main_nodes(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jnodes", description="显示 Slurm 节点资源概览。")
    add_node_filter_args(parser)
    args = parser.parse_args(argv)
    return run_with_errors(command_nodes, args)


def main_partitions(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jpart", description="显示 Slurm partition 概览。")
    parser.add_argument("-p", "--partition", help="只显示指定 partition。")
    add_common_output_flags(parser)
    args = parser.parse_args(argv)
    return run_with_errors(command_partitions, args)


def main_why(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jwhy", description="汇总 Slurm pending 作业原因。")
    add_why_args(parser)
    args = parser.parse_args(argv)
    return run_with_errors(command_why, args)


def main_history(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jhist", description="显示近期 Slurm accounting 历史。")
    add_history_args(parser)
    args = parser.parse_args(argv)
    return run_with_errors(command_history, args)


if __name__ == "__main__":
    raise SystemExit(main())
