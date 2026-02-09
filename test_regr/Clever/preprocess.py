import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pickle, py7zr
import os.path as osp
try:
    from .dataset import make_dataset, default_image_transform
except ImportError:
    from dataset import make_dataset, default_image_transform

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
            ds.filter_one_relation()
        elif q_type == 'query':
            ds.filter_query_no_same()
        elif q_type == 'query_relation':
            ds.filter_query_with_relations(max_relations=2)
        elif q_type == 'exist':
            ds.filter_relational_type()
        elif q_type == 'counting':
            ds.filter_atmostlatleastlequal()
        elif q_type == 'complex_relation':
            ds.filter_complex_relation()
        elif q_type == 'counting':
            ds.filter_counting()
        
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
                dataset_len = len(ds) if hasattr(ds, '__len__') else NUM_INSTANCES
                for idx_ in range(min(NUM_INSTANCES, dataset_len)):
                    cache_file_ = CACHE_DIR / f"instance_{idx_}{cache_suffix}.pkl"
                    with cache_file_.open("wb") as f:
                        pickle.dump(ds[idx_], f, protocol=pickle.HIGHEST_PROTOCOL)
                        print(f"cached to {cache_file_}")
                dataset = [ds[i] for i in range(min(NUM_INSTANCES, dataset_len))]
                break
    else:
        cache_file = CACHE_DIR / f"dataset{cache_suffix}.pkl"
        if cache_file.exists():
            with cache_file.open("rb") as f:
                dataset = pickle.load(f)
                print(f"re-loaded {cache_file}")
        else:
            ds = build_dataset(question_type)
            dataset = [ds[i] for i in range(len(ds))]
            with cache_file.open("wb") as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"cached to {cache_file}")

    print(f"Dataset length: {len(dataset)}")
    
    if args.eval_only:
        if args.test_size is not None and args.test_size < len(dataset):
            dataset = dataset[-args.test_size:]
    else:
        if args.train_size is not None:
            if args.subset != -1:
                subset_size = args.train_size // 6
                start_idx = subset_size * (args.subset - 1)
                end_idx = subset_size * args.subset
                dataset = dataset[start_idx:end_idx]
            else:
                dataset = dataset[:min(args.train_size, len(dataset))]
        # If train_size is None, use full dataset
    
    return dataset


def preprocess_folders_and_files(dummy):
    if not dummy and not Path("train/vocab.json").exists():
        print("Extracting json files...")
        with py7zr.SevenZipFile(Path("train/output-vocab.7z"), mode="r") as z:
            z.extractall(path="train/")
        print("Extraction complete")

    CACHE_DIR = Path("dataset_cache")
    for directory in [CACHE_DIR, Path("models"), Path("cache")]:
        directory.mkdir(exist_ok=True)
    
    return CACHE_DIR