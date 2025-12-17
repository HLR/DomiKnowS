import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('./')
sys.path.append('../')

import pickle, py7zr
from dataset import make_dataset, default_image_transform
import os.path as osp
from pathlib import Path

def preprocess_dataset(args, NUM_INSTANCES, CACHE_DIR, question_type='relation'):
    """
    Preprocess dataset with configurable question type filtering.
    
    Args:
        args: Command line arguments
        NUM_INSTANCES: Number of instances for dummy mode
        CACHE_DIR: Cache directory path
        question_type: One of 'relation', 'query', 'query_relation', 'exist'
    """
    def build_dataset(q_type='relation'):
        ds = make_dataset(
            scenes_json=osp.join("train", "scenes.json"),
            questions_json=osp.join("train", "questions.json"),
            image_root=osp.join("train", "images"),
            image_transform=default_image_transform,
            vocab_json=osp.join("train", "vocab.json"),
            output_vocab_json=osp.join("train", "output-vocab.json"),
            incl_scene=True,
            incl_raw_scene=True,
        )

        if q_type == 'relation':
            # Original: filter for one relation (exist/count questions)
            ds.filter_one_relation()
        elif q_type == 'query':
            # Query questions only (no relations)
            ds.filter_query_no_same()
        elif q_type == 'query_relation':
            # Query questions with up to 2 relations
            ds.filter_query_with_relations(max_relations=2)
        elif q_type == 'exist':
            # Existence questions without query
            ds.filter_relational_type()
        elif q_type == 'counting':
            # Counting questions
            ds.filter_atmostlatleastlequal()
        
        return ds

    dataset = []
    cache_suffix = f"_{question_type}" if question_type != 'relation' else ""
    
    if args.dummy:
        for idx in range(NUM_INSTANCES):
            cache_file = CACHE_DIR / f"instance_{idx}{cache_suffix}.pkl"

            if cache_file.exists():
                with cache_file.open("rb") as f:
                    instance = pickle.load(f)
                    dataset.append(instance)
                    print(f"re-loaded {cache_file}")
            else:
                ds = build_dataset(question_type)
                for idx_ in range(min(NUM_INSTANCES, len(ds))):
                    cache_file = CACHE_DIR / f"instance_{idx_}{cache_suffix}.pkl"
                    with cache_file.open("wb") as f:
                        pickle.dump(ds[idx_], f, protocol=pickle.HIGHEST_PROTOCOL)
                        print(f"cached to {cache_file}")
                dataset = [ds[i] for i in range(min(NUM_INSTANCES, len(ds)))]
                break
    else:
        cache_file = CACHE_DIR / f"iotaL_dataset{cache_suffix}.pkl"
        if cache_file.exists():
            with cache_file.open("rb") as f:
                dataset = pickle.load(f)
                print(f"re-loaded {cache_file}")
        else:
            dataset = build_dataset(question_type)
            with cache_file.open("wb") as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"cached to {cache_file}")

    print("Dataset length:", len(dataset))
    
    if args.eval_only:
        dataset = [dataset[i] for i in range(len(dataset) - args.test_size, len(dataset))]
    else:
        print(len(dataset))
        if args.subset != -1:
            subset_size = args.train_size // 6
            dataset = [dataset[i] for i in range(subset_size * (args.subset - 1), subset_size * args.subset)]
        else:
            dataset = [dataset[i] for i in range(args.train_size)]
    
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