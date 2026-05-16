from __future__ import annotations

import re
import shutil
import sys
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True)
class Column:
    key: str
    title: str
    min_width: int = 3
    max_width: int | None = None
    align: str = "left"


def strip_ansi(value: str) -> str:
    return ANSI_RE.sub("", value)


def visible_len(value: str) -> int:
    return len(strip_ansi(value))


def should_color(no_color: bool = False) -> bool:
    return not no_color and sys.stdout.isatty()


def colorize(value: str, color: str | None, enabled: bool) -> str:
    if not enabled or not color:
        return value
    colors = {
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "gray": "90",
        "bold": "1",
    }
    code = colors.get(color)
    if code is None:
        return value
    return f"\x1b[{code}m{value}\x1b[0m"


def color_for_state(state: str) -> str | None:
    upper = state.upper()
    if upper.startswith(("RUNNING", "COMPLETING")):
        return "green"
    if upper.startswith(("PENDING", "CONFIGURING", "RESIZING")):
        return "yellow"
    if upper.startswith(("COMPLETED",)):
        return "green"
    if upper.startswith(("FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE")):
        return "red"
    if upper.startswith(("CANCELLED", "PREEMPTED", "REVOKED")):
        return "magenta"
    return None


def format_mb(value: int | None) -> str:
    if value is None:
        return "-"
    if value < 0:
        value = 0
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.1f}T"
    if value >= 1024:
        return f"{value / 1024:.0f}G"
    return f"{value}M"


def format_ratio(free: int | None, total: int | None) -> str:
    left = "-" if free is None else str(max(0, free))
    right = "-" if total is None else str(max(0, total))
    return f"{left}/{right}"


def truncate(value: str, width: int) -> str:
    if visible_len(value) <= width:
        return value
    plain = strip_ansi(value)
    if width <= 1:
        return plain[:width]
    return plain[: width - 1] + "…"


def pad(value: str, width: int, align: str) -> str:
    value = truncate(value, width)
    gap = width - visible_len(value)
    if gap <= 0:
        return value
    if align == "right":
        return " " * gap + value
    return value + " " * gap


def table(rows: Sequence[Mapping[str, object]], columns: Sequence[Column]) -> str:
    if not columns:
        return ""

    widths: list[int] = []
    for column in columns:
        content_width = visible_len(column.title)
        for row in rows:
            content_width = max(content_width, visible_len(str(row.get(column.key, ""))))
        if column.max_width is not None:
            content_width = min(content_width, column.max_width)
        widths.append(max(column.min_width, content_width))

    term_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    total_width = sum(widths) + 2 * (len(widths) - 1)
    shrinkable = [i for i, column in enumerate(columns) if widths[i] > column.min_width]
    while total_width > term_width and shrinkable:
        idx = max(shrinkable, key=lambda i: widths[i] - columns[i].min_width)
        widths[idx] -= 1
        if widths[idx] <= columns[idx].min_width:
            shrinkable.remove(idx)
        total_width = sum(widths) + 2 * (len(widths) - 1)

    header = "  ".join(pad(column.title, widths[i], column.align) for i, column in enumerate(columns))
    rule = "  ".join("-" * widths[i] for i in range(len(columns)))
    lines = [header, rule]
    for row in rows:
        lines.append(
            "  ".join(
                pad(str(row.get(column.key, "")), widths[i], column.align)
                for i, column in enumerate(columns)
            )
        )
    return "\n".join(lines)


def print_table(rows: Sequence[Mapping[str, object]], columns: Sequence[Column]) -> None:
    print(table(rows, columns))


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def join_short(values: Iterable[str], limit: int = 3) -> str:
    items = list(values)
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f", +{len(items) - limit}"
