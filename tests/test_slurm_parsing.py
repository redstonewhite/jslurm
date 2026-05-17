import unittest
from unittest import mock

from jslurm.cli import filter_nodes, format_start_time, job_gpu_summary
from jslurm.formatting import visible_len, wrap_text
import jslurm.slurm as slurm
from jslurm.slurm import (
    DELIM,
    Node,
    expand_nodelist,
    gpu_model_from_feature,
    gpu_type_free_total,
    gpu_total_from_gres,
    parse_nodes,
    parse_tres,
    squeue_jobs,
)


class SlurmParsingTests(unittest.TestCase):
    def test_parse_tres(self):
        self.assertEqual(
            parse_tres("cpu=64,mem=512000M,billing=64,gres/gpu=4,gres/gpu:a100=4"),
            {
                "cpu": 64,
                "mem": 512000,
                "billing": 64,
                "gres/gpu": 4,
                "gres/gpu:a100": 4,
            },
        )

    def test_squeue_jobs_parses_more_than_two_rows(self):
        rows = [
            [
                "101",
                "train-a",
                "RUNNING",
                "1:00",
                "12:00:00",
                "1",
                "p001",
                "general",
                "cpu=8,mem=100G,gres/gpu=1",
                "8",
                "100G",
                "2026-05-16T10:00:00",
                "2026-05-16T09:00:00",
                "alice",
                "100",
                "h200",
            ],
            [
                "102",
                "train-b",
                "RUNNING",
                "2:00",
                "12:00:00",
                "1",
                "p002",
                "general",
                "cpu=8,mem=100G,gres/gpu=1",
                "8",
                "100G",
                "2026-05-16T10:10:00",
                "2026-05-16T09:10:00",
                "alice",
                "90",
                "h200",
            ],
            [
                "103",
                "train-c",
                "PENDING",
                "0:00",
                "12:00:00",
                "1",
                "Resources",
                "general",
                "cpu=8,mem=100G,gres/gpu=1",
                "8",
                "100G",
                "2026-05-16T12:00:00",
                "2026-05-16T09:20:00",
                "alice",
                "80",
                "h200",
            ],
            [
                "104",
                "train-d",
                "PENDING",
                "0:00",
                "12:00:00",
                "1",
                "Priority",
                "general",
                "cpu=8,mem=100G,gres/gpu=1",
                "8",
                "100G",
                "2026-05-16T13:00:00",
                "2026-05-16T09:30:00",
                "alice",
                "70",
                "h200",
            ],
        ]
        output = "\n".join(DELIM.join(row) for row in rows) + "\n"
        with mock.patch.object(slurm, "run_slurm", return_value=output):
            jobs = squeue_jobs(user="alice")
        self.assertEqual([job.job_id for job in jobs], ["101", "102", "103", "104"])

    def test_gpu_total_from_gres(self):
        self.assertEqual(gpu_total_from_gres("gpu:a100:4(S:0-3)"), 4)
        self.assertEqual(gpu_total_from_gres("gpu:2"), 2)
        self.assertIsNone(gpu_total_from_gres("-"))

    def test_parse_nodes(self):
        nodes = parse_nodes(
            "NodeName=gpu01 State=MIXED CPUTot=64 CPUAlloc=16 RealMemory=512000 "
            "AllocMem=128000 FreeMem=300000 Gres=gpu:a100:4(S:0-3) "
            "CfgTRES=cpu=64,mem=512000M,billing=64,gres/gpu=4,gres/gpu:a100=4 "
            "AllocTRES=cpu=16,mem=128000M,gres/gpu=1,gres/gpu:a100=1 "
            "Partitions=gpu,debug ActiveFeatures=a100,ib\n"
        )
        self.assertEqual(len(nodes), 1)
        node = nodes[0]
        self.assertEqual(node.gpu_total, 4)
        self.assertEqual(node.gpu_alloc, 1)
        self.assertEqual(node.gpu_free, 3)
        self.assertEqual(node.cpus_free, 48)
        self.assertEqual(node.mem_free_sched_mb, 384000)

    def test_parse_nodes_uses_gres_used_as_alloc_fallback(self):
        nodes = parse_nodes(
            "NodeName=gpu02 State=MIXED CPUTot=64 CPUAlloc=16 RealMemory=512000 "
            "AllocMem=128000 FreeMem=300000 Gres=gpu:a100:4(S:0-3) "
            "GresUsed=gpu:a100:2(IDX:0-1) "
            "CfgTRES=cpu=64,mem=512000M,billing=64,gres/gpu=4,gres/gpu:a100=4 "
            "AllocTRES=cpu=16,mem=128000M "
            "Partitions=gpu ActiveFeatures=a100\n"
        )
        node = nodes[0]
        self.assertEqual(node.gpu_alloc, 2)
        self.assertEqual(node.gpu_free, 2)
        self.assertEqual(node.gpu_types, {"a100": (2, 4)})

    def test_filter_nodes_can_hide_busy_gpu_nodes(self):
        free_node = Node(
            name="p001",
            state="MIXED",
            partitions=["general"],
            cpus_total=64,
            cpus_alloc=16,
            mem_total_mb=755000,
            mem_alloc_mb=100000,
            mem_free_os_mb=20000,
            gres="gpu:h200:4",
            gres_used="-",
            cfg_tres={"gres/gpu": 4, "gres/gpu:h200": 4},
            alloc_tres={"gres/gpu": 1, "gres/gpu:h200": 1},
            features="hopper,h200,141g",
        )
        busy_node = Node(
            name="q001",
            state="MIXED",
            partitions=["general"],
            cpus_total=64,
            cpus_alloc=64,
            mem_total_mb=755000,
            mem_alloc_mb=700000,
            mem_free_os_mb=1000,
            gres="gpu:h200:4",
            gres_used="-",
            cfg_tres={"gres/gpu": 4, "gres/gpu:h200": 4},
            alloc_tres={"gres/gpu": 4, "gres/gpu:h200": 4},
            features="hopper,h200,141g",
        )
        all_nodes = filter_nodes([free_node, busy_node], gpu_only=True)
        free_nodes = filter_nodes([free_node, busy_node], gpu_only=True, available_only=True)
        self.assertEqual([node.name for node in all_nodes], ["p001", "q001"])
        self.assertEqual([node.name for node in free_nodes], ["p001"])

    def test_gpu_type_free_total(self):
        typed = gpu_type_free_total(
            {"gres/gpu": 6, "gres/gpu:a100": 4, "gres/gpu:v100": 2},
            {"gres/gpu:a100": 1, "gres/gpu:v100": 2},
            "-",
        )
        self.assertEqual(typed, {"a100": (3, 4), "v100": (0, 2)})

    def test_job_gpu_summary(self):
        self.assertEqual(job_gpu_summary("cpu=8,mem=64000M,gres/gpu:a100=2"), "a100=2")
        self.assertEqual(job_gpu_summary("cpu=4,gres/gpu=1"), "gpu=1")
        self.assertEqual(job_gpu_summary("cpu=4,gres/gpu=1", "h200"), "h200=1")
        self.assertEqual(job_gpu_summary("cpu=4,gres/gpu=2", "h200&ib"), "h200=2")
        self.assertEqual(job_gpu_summary("cpu=4,gres/gpu=1", "ib"), "gpu=1")
        self.assertEqual(job_gpu_summary("cpu=4,gres/gpu=1", "ib,nvlink"), "gpu=1")
        self.assertEqual(job_gpu_summary("cpu=4,gres/gpu=1", node_models=["h200"]), "h200=1")
        self.assertEqual(job_gpu_summary("cpu=4,gres/gpu=2", node_models=["h200", "a100"]), "gpu=2(a100/h200)")
        self.assertEqual(job_gpu_summary("cpu=4,mem=16000M"), "-")

    def test_gpu_model_from_feature_is_conservative(self):
        self.assertEqual(gpu_model_from_feature("h200"), "h200")
        self.assertEqual(gpu_model_from_feature("h200&ib"), "h200")
        self.assertIsNone(gpu_model_from_feature("ib"))
        self.assertIsNone(gpu_model_from_feature("general"))

    def test_expand_nodelist(self):
        self.assertEqual(expand_nodelist("p002"), ["p002"])
        self.assertEqual(expand_nodelist("p[001-003]"), ["p001", "p002", "p003"])
        self.assertEqual(expand_nodelist("rack[1-2]n[01-02]"), ["rack1n01", "rack1n02", "rack2n01", "rack2n02"])

    def test_format_start_time_omits_year(self):
        self.assertEqual(format_start_time("2026-05-16T14:32:09"), "05-16 14:32")
        self.assertEqual(format_start_time("2026-05-16 14:32:09"), "05-16 14:32")
        self.assertEqual(format_start_time("2026-05-16"), "05-16")
        self.assertEqual(format_start_time("-"), "-")

    def test_wrap_text_limits_visible_width(self):
        message = (
            "Hidden GPU nodes: 12 "
            "(p001, p002, p003, p004, p005, p006, p007, p008, +4). "
            "Use `javail --all` to show all."
        )
        wrapped = wrap_text(message, 36)
        self.assertTrue(all(visible_len(line) <= 36 for line in wrapped.splitlines()))


if __name__ == "__main__":
    unittest.main()
