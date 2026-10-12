from s1cli import fake, slurm


def test_state_bucket():
    assert slurm.state_bucket("idle") == "idle"
    assert slurm.state_bucket("idle~") == "cloud"
    assert slurm.state_bucket("idle#") == "booting"
    assert slurm.state_bucket("mix") == "busy"
    assert slurm.state_bucket("alloc") == "busy"
    assert slurm.state_bucket("down~") == "down"
    assert slurm.state_bucket("drain*") == "down"


def test_parse_sinfo_aggregates_partitions():
    parts = {p.name: p for p in slurm.parse_sinfo(fake.SINFO)}
    assert set(parts) == {"cpu", "gpu", "gpu-large"}
    cpu, gpu, big = parts["cpu"], parts["gpu"], parts["gpu-large"]
    assert cpu.default and not cpu.has_gpu
    assert cpu.nodes == 20 and cpu.states["cloud"] == 18 and cpu.states["busy"] == 2
    assert gpu.nodes == 10 and gpu.states["booting"] == 1
    assert gpu.gpu_types == {None: 1}
    assert big.gpu_types == {"l40s": 1} and big.vram_gb == 48
    assert big.states["down"] == 1
    assert gpu.timelimit == "8:00:00" and gpu.cpus == 4 and gpu.mem_mb == 15000


def test_refine_gpu_types_from_instance_name():
    parts = slurm.parse_sinfo(fake.SINFO)
    slurm.refine_gpu_types(parts, slurm.parse_scontrol_nodes(fake.SCONTROL_NODES))
    gpu = next(p for p in parts if p.name == "gpu")
    assert gpu.gpu_types == {"a10g": 1}
    assert gpu.vram_gb == 24


def test_parse_squeue():
    jobs = slurm.parse_squeue(fake.SQUEUE)
    assert [j.id for j in jobs] == ["4101", "4107", "4110"]
    assert jobs[0].running and jobs[0].is_s1
    assert not jobs[2].is_s1 and jobs[2].reason == "Resources"


def test_timelimit():
    assert slurm.parse_timelimit_minutes("1-00:00:00") == 1440
    assert slurm.parse_timelimit_minutes("4:00:00") == 240
    assert slurm.parse_timelimit_minutes("30:00") == 30
    assert slurm.parse_timelimit_minutes("infinite") is None


def test_srun_cmd():
    a = slurm.Alloc(partition="gpu", gpus=1, cpus=8, mem="32G", time="01:00:00", name="s1-gpu-1200")
    assert a.srun_cmd() == [
        "srun",
        "-p",
        "gpu",
        "--gres=gpu:1",
        "-c",
        "8",
        "--mem=32G",
        "-t",
        "01:00:00",
        "-J",
        "s1-gpu-1200",
        "--pty",
        "bash",
        "-l",
    ]
    a.gpu_type = "l40s"
    assert "--gres=gpu:l40s:1" in a.srun_cmd()
