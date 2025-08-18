import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('./')
sys.path.append('../')

import pickle, py7zr
from dataset import make_dataset, default_image_transform
import os.path as osp
from pathlib import Path

def preprocess_dataset(args,NUM_INSTANCES,CACHE_DIR):
    def build_dataset():
        ds = make_dataset(
            scenes_json      = osp.join("train", "scenes.json"),
            questions_json   = osp.join("train", "questions.json"),
            image_root       = osp.join("train", "images"),
            image_transform  = default_image_transform,
            vocab_json       = osp.join("train", "vocab.json"),
            output_vocab_json= osp.join("train", "output-vocab.json"),
            incl_scene       = True,
            incl_raw_scene   = True,
        )

        # Filter dataset for attribute only
        # ds.filter_relational_type()

        # Filter dataset for attribute only
        ds.filter_one_relation()


        # Filter dataset for attribute only
        # ds.filter_atmostlatleastlequal()
        return ds

    dataset = []
    if args.dummy:
        for idx in range(NUM_INSTANCES):
            cache_file = CACHE_DIR / f"instance_{idx}.pkl"

            if cache_file.exists():
                with cache_file.open("rb") as f:
                    instance = pickle.load(f)
                    dataset.append(instance)
                    print(f"re-loaded {cache_file}")
            else:
                ds = build_dataset()
                for idx_ in range(NUM_INSTANCES):
                    cache_file = CACHE_DIR / f"instance_{idx_}.pkl"
                    with cache_file.open("wb") as f:
                        pickle.dump(ds[idx_], f, protocol=pickle.HIGHEST_PROTOCOL)
                        print(f"cached to {cache_file}")
    else:
        cache_file = CACHE_DIR / f"existsL_dataset.pkl"
        if cache_file.exists():
            with cache_file.open("rb") as f:
                dataset = pickle.load(f)
                print(f"re-loaded {cache_file}")
        else:
            dataset = build_dataset()
            with cache_file.open("wb") as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"cached to {cache_file}")

    print("Dataset length:", len(dataset))
    if args.eval_only:
        dataset = [dataset[i] for i in range(len(dataset) - args.test_size, len(dataset))]
        #dataset = [dataset[i] for i in range(args.test_size)]
    else:
        print(len(dataset))
        # args.subset = 1,2,3,4,5,6
        if args.subset != -1:
            subset_size = args.train_size // 6
            dataset = [dataset[i] for i in range(subset_size * (args.subset - 1), subset_size * args.subset)]
        else:
            dataset = [dataset[i] for i in range(args.test_size)]
    return dataset

def preprocess_folders_and_files(dummy):
    if not dummy and not Path("train/vocab.json").exists():
        print(f"Extracting json files...")
        with py7zr.SevenZipFile(Path("train/output-vocab.7z"), mode="r") as z:
            z.extractall(path="train/")
        print(f"Extraction complete")

    CACHE_DIR = Path("dataset_cache")
    for directory in [CACHE_DIR, Path("models"), Path("cache")]:
        directory.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR