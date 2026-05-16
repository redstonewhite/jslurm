import unittest

from jslurm.cli import format_start_time, job_gpu_summary
from jslurm.slurm import (
    expand_nodelist,
    gpu_model_from_feature,
    gpu_type_free_total,
    gpu_total_from_gres,
    parse_nodes,
    parse_tres,
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


if __name__ == "__main__":
    unittest.main()
