# jslurm

`jslurm` is a small set of `uv tool install`-friendly Slurm inspection commands. It turns `squeue`, `scontrol`, `sinfo`, and `sacct` output into compact tables or JSON for day-to-day job and GPU resource checks.

Requirements: Python 3.10+ and Slurm commands available on the login node.

## Installation

From this repository:

```bash
uv tool install .
```

Force reinstall during development:

```bash
uv tool install --force .
```

Try commands without installing:

```bash
uvx --from . jslurm --help
uvx --from . jqueue --help
```

## Commands

### jqueue

Show jobs for the current user, including job id, name, state, partition, `WHERE`, GPU, CPU, memory, elapsed time, time limit, and estimated start time. `WHERE` shows the allocated node for running jobs and the pending reason for waiting jobs.

```bash
jqueue
jqueue --long
jqueue -u alice
jqueue -t R,PD
jqueue -p gpu --watch 10
jqueue --json
```

### javail

Show currently available GPU node resources. By default it only shows GPU nodes with free GPUs and schedulable node states.

```bash
javail
javail -p gpu
javail -g a100 --min-gpus 2
javail --all
javail --json
```

Columns:

- `CPU`: free CPU / total CPU.
- `GPU`: free GPU / total GPU by type, for example `h200 3/4`.
- `MEM`: Slurm schedulable free memory / total memory.

JSON output still includes `mem_free_os_mb`, which comes from the `FreeMem` field in `scontrol show node`. That is the operating system's current free memory reported to Slurm and is useful for diagnosing real-time node memory pressure. For normal scheduling decisions, prefer the table's `MEM` column.

### jnodes

Show a node resource overview. You can filter by partition, GPU type, or node state.

```bash
jnodes
jnodes --available
jnodes -p gpu -g a100
jnodes --states IDLE,MIXED
```

### jpart

Show a partition overview based on `sinfo`.

```bash
jpart
jpart -p gpu
```

### jwhy

Summarize pending job reasons. This is useful for quickly checking whether jobs are waiting on priority, resources, dependencies, or partition limits.

```bash
jwhy
jwhy --all
jwhy -p gpu
```

### jhist

Show recent `sacct` history.

```bash
jhist
jhist --days 3
jhist -u alice --json
```

### jslurm

All commands are also available through one entry point:

```bash
jslurm queue
jslurm avail
jslurm nodes
jslurm part
jslurm why
jslurm hist
```

## Notes

- Commands are read-only by default. They do not cancel or modify jobs.
- Use `--json` for scripts and `jq`.
- Use `--watch SEC` to refresh at an interval.
- Tables use color when stdout is a terminal. Piped output disables color automatically, and `--no-color` disables it explicitly.
- Table start times omit the year, for example `05-16 14:32`; JSON keeps the original Slurm time string.
- `jqueue` gets the GPU column from `squeue -O tres-alloc`. Running jobs show allocated TRES; pending jobs show requested TRES. If the cluster only returns `gres/gpu=1`, `jqueue` first tries to infer the model from job features, for example `h200=1`. For running jobs without an explicit feature model, it may load node information once and infer the model from allocated nodes' `Gres`, `CfgTRES`, or `ActiveFeatures`.
- `javail` GPU counts mainly come from `CfgTRES` and `AllocTRES` in `scontrol show node -o`. If GPU allocation details are missing there, it falls back to parsing `Gres` and `GresUsed`.
