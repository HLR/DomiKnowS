import os
import gc

import subprocess

# Basic usage

if __name__ == "__main__":
    
    # All learning rate to use
    for lr in [1e-4, 1e-5, 1e-6]:
        # Number of epoch to run with
        for epoch in range(20):
            # Subset from 1 to 6 to reduce memory used
            for sub_round in range(6):
                # Training
                command = [
                    "python", "main.py",
                    "--train-size", "6000",
                    "--test-size", "1000",
                    "--epochs", "1",
                    "--subset", str(sub_round + 1),
                    "--load-epoch", str(epoch),
                    "--load_previous_save",
                    "--lr", str(lr)
                ]
                subprocess.run(command)
                # Test
                if sub_round == 5:
                    command = [
                        "python", "main.py",
                        "--train-size", "6000",
                        "--test-size", "1000",
                        "--epochs", "1",
                        "--subset", str(sub_round + 1),
                        "--load-epoch", str(epoch),
                        "--lr", str(lr),
                        "--load_previous_save",
                        "--eval-only"
                    ]
                    subprocess.run(command)
                
                # program1e-06_1_0__1_6000_
                # G_1
                # program1e-06_0__1_6000_G_1.pth