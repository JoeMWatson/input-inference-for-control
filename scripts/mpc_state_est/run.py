import subprocess

seeds = 50
concurrent = 10
for r in range(seeds // concurrent):
    s1 = r * concurrent
    s2 = s1 + concurrent
    cmd = " & ".join(
        [f"python scripts/mpc_gate/mpc_quad.py {s}" for s in range(s1, s2)]
    )
    print(cmd)
    subprocess.run(cmd, shell=True)
