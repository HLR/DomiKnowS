import subprocess

if __name__ == '__main__':

    for lr in [1e-5, 1e-6]:
        for epoch in [10]:
            for N in [1000]:
                model_count = 0
                for relation in range(4):
                    command = ["python", "main_rel.py", "--lr", str(lr), "--N", f"{N}", "--epoch", f"{epoch * (relation + 1)}", "--save_file", f"model_{model_count}_lr_{lr}_epoch_{N}.pth", "--max_relation", f"{relation}"]

                    # if relation != 0:
                    #     command.extend(["--load_save", f"model_{model_count - 1}_lr_{lr}_epoch_{N}.pth"])
                    subprocess.run(command)

                    command = ["python", "main_rel.py", "--lr", str(lr), "--evaluate", "--N", f"{N}", "--epoch", f"{epoch * (relation + 1)}", "--load_save", f"model_{model_count}_lr_{lr}_epoch_{N}.pth", "--max_relation", f"{relation}"]
                    subprocess.run(command)

                    model_count += 1