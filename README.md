# jslurm

`jslurm` 是一组可用 `uv tool install` 安装的 Slurm 查询工具，目标是把 `squeue`、`scontrol`、`sinfo`、`sacct` 的输出整理成更适合日常提交和排队判断的表格或 JSON。

要求：Python 3.10+，并且登录节点上可以直接调用 Slurm 命令。

## 安装

在本仓库目录运行：

```bash
uv tool install .
```

开发时想覆盖已有安装：

```bash
uv tool install --force .
```

也可以只在当前环境中试跑：

```bash
uvx --from . jslurm --help
uvx --from . jqueue --help
```

## 命令

### jqueue

显示当前用户的 job，包含 job id、名称、状态、partition、节点或 pending 原因、GPU、CPU、内存、运行时间、time limit 和预计开始时间。

```bash
jqueue
jqueue --long
jqueue -u alice
jqueue -t R,PD
jqueue -p gpu --watch 10
jqueue --json
```

### javail

显示当前立即可用的 GPU 节点资源。默认只显示有空闲 GPU 且节点状态可调度的 GPU 节点。

```bash
javail
javail -p gpu
javail -g a100 --min-gpus 2
javail --all
javail --json
```

列含义：

- `GPU`: 空闲 GPU / 总 GPU。
- `GPU TYPE`: 按 GPU 类型显示空闲量和总量。
- `CPU`: 空闲 CPU / 总 CPU。
- `MEM`: Slurm 视角下未分配内存 / 总内存。
- `OSFREE`: 节点上报的操作系统空闲内存，仅供参考。

### jnodes

显示所有节点资源概览，可按 partition、GPU 类型或状态过滤。

```bash
jnodes
jnodes --available
jnodes -p gpu -g a100
jnodes --states IDLE,MIXED
```

### jpart

显示 partition 概览，对应 `sinfo` 的分区、状态、节点数量、GRES 和 nodelist。

```bash
jpart
jpart -p gpu
```

### jwhy

汇总当前 pending 作业的等待原因，适合快速判断是优先级、资源不足还是 dependency。

```bash
jwhy
jwhy --all
jwhy -p gpu
```

### jhist

显示近期 `sacct` 历史记录。

```bash
jhist
jhist --days 3
jhist -u alice --json
```

### jslurm

所有命令也可以通过一个总入口调用：

```bash
jslurm queue
jslurm avail
jslurm nodes
jslurm part
jslurm why
jslurm hist
```

## 说明

- 所有命令默认只读，不会修改或取消作业。
- `--json` 可用于脚本和 `jq`。
- `--watch SEC` 会每隔指定秒数刷新一次。
- `jqueue` 的 GPU 列来自 `squeue -O tres-alloc`，运行中显示已分配 TRES，等待中显示请求 TRES。
- `javail` 的 GPU 数量主要来自 `scontrol show node -o` 中的 `CfgTRES` 和 `AllocTRES`；如果 GPU 分配信息缺失，会退回解析 `Gres` 和 `GresUsed`。
