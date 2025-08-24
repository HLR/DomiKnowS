import logging
import subprocess

try:
    from monitor.constraint_monitor import ( # type: ignore
         enable_monitoring, start_new_epoch
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Initialize monitoring
if MONITORING_AVAILABLE:
    enable_monitoring(port=8080, slave_mode=False)  # Master mode with web server
    
# Basic usage
if __name__ == "__main__":
    
    # All learning rate to use
    for lr in [1e-4, 1e-5, 1e-6]:
        # Number of epoch to run with
        for epoch in range(20):
            # New epoch
            if MONITORING_AVAILABLE:
                start_new_epoch()
                logging.getLogger(__name__).info(f"Starting new epoch {epoch} with lr {lr}")
            else:
                logging.getLogger(__name__).info(f"Starting new epoch {epoch} with lr {lr} (monitoring disabled)")
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