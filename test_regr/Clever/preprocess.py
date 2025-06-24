import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('./')
sys.path.append('../')

import pickle
from dataset import make_dataset, default_image_transform
import os.path as osp

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
        ds.filter_relational_type()
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
        dataset = [dataset[i] for i in range(args.train_size)]
    return dataset