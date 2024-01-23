import itertools
import subprocess


hyperparams = {
    'gbi-iters': [25, 50, 100],
    'lr': [1e-3, 1e-2, 1e-1],
    'reg': [0.5, 1.0, 2.0],
    'num-samples': [1000]
}

combinations = list(itertools.product(*hyperparams.values()))

for combination in combinations:
    args = ["python", "infer_gbi.py"]
    for key, value in zip(hyperparams.keys(), combination):
        args.extend([f"--{key}", str(value)])
    
    args.append('--training')
    
    print("Running:", " ".join(args))
    subprocess.run(args)
