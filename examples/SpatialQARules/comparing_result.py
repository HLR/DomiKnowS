import pandas as pd
import argparse

def compare_result(first_file, second_file):
    first = pd.read_csv(first_file, index_col=0)
    second = pd.read_csv(second_file, index_col=0)
    file_names = ["compare_t_t.csv", "compare_t_f.csv", "compare_f_t.csv", "compare_f_f.csv"]
    result =[[] for _ in range(4)]
    for index, first_data in first.iterrows():
        f_label = 0 if first_data["argmax"] == first_data["label"] else 1
        s_label = 0 if second.loc[index, "argmax"] == second.loc[index, "label"] else 1
        cur_dict = first_data.to_dict()
        cur_dict["second_argmax"] = second.loc[index, "argmax"]
        result[f_label * 2 + s_label].append(cur_dict) # Need to plus value for second

    for i in range(4):
        df = pd.DataFrame(result[i])
        df.to_csv(file_names[i])
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", dest="first", type=str, default="result.csv")
    parser.add_argument("--second", dest="second", type=str, default="result.csv")
    args = parser.parse_args()
    compare_result(args.first, args.second)