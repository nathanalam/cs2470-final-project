"""Canned Slurm/nvidia-smi output for S1_FAKE_SLURM=1 (UI development off-cluster).

Modeled on an AWS ParallelCluster setup: dynamic nodes named after instance types,
powered-down nodes reported as "idle~".
"""

from __future__ import annotations

SINFO = """\
cpu*|up|1-00:00:00|18|idle~|(null)|8|30000|cpu-dy-c5x2large-[3-20]
cpu*|up|1-00:00:00|2|mix|(null)|8|30000|cpu-dy-c5x2large-[1-2]
gpu|up|8:00:00|7|idle~|gpu:1|4|15000|gpu-dy-g5xlarge-[4-10]
gpu|up|8:00:00|2|alloc|gpu:1|4|15000|gpu-dy-g5xlarge-[1-2]
gpu|up|8:00:00|1|idle#|gpu:1|4|15000|gpu-dy-g5xlarge-3
gpu-large|up|4:00:00|3|idle~|gpu:l40s:1|8|60000|gpu-large-dy-g6e2xlarge-[2-4]
gpu-large|up|4:00:00|1|down~|gpu:l40s:1|8|60000|gpu-large-dy-g6e2xlarge-1
"""

SCONTROL_NODES = """\
NodeName=gpu-dy-g5xlarge-1 Arch=x86_64 CPUTot=4 AvailableFeatures=dynamic,g5.xlarge,gpu ActiveFeatures=dynamic,g5.xlarge,gpu Gres=gpu:1 RealMemory=15000 State=ALLOCATED+CLOUD Partitions=gpu
NodeName=gpu-large-dy-g6e2xlarge-2 Arch=x86_64 CPUTot=8 Gres=gpu:l40s:1 RealMemory=60000 State=IDLE+CLOUD+POWERED_DOWN Partitions=gpu-large
"""

SQUEUE = """\
4101|gpu|s1-gpu-1412|RUNNING|None|1:02:11|8:00:00|gpu-dy-g5xlarge-1|gres/gpu:1
4107|gpu-large|s1-gpu-large-1530|PENDING|BeginTime|0:00|4:00:00||gres/gpu:l40s:1
4110|cpu|jevbench-sweep|PENDING|Resources|0:00|1-00:00:00||N/A
"""

NVIDIA_SMI = "0, NVIDIA A10G, 6132, 23028, 41\n"


def respond(cmd: list[str]) -> str:
    prog = cmd[0]
    if prog == "sinfo":
        return SINFO
    if prog == "scontrol" and cmd[1:3] == ["show", "node"]:
        return SCONTROL_NODES
    if prog == "scontrol" and cmd[1:3] == ["show", "job"]:
        return (
            f"JobId={cmd[3]} JobName=s1-fake UserId=you(1000) JobState=RUNNING\n   (fake scontrol output)\n"
        )
    if prog == "squeue":
        return SQUEUE
    if prog == "scancel":
        return ""
    if "nvidia-smi" in cmd:
        return NVIDIA_SMI
    return ""
