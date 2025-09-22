from dataset import make_dataset, default_image_transform
import os.path as osp, os, random, pickle, json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
import pickle

load_dotenv()

MAIN_PATH = 'train'

class GPT4oDirectConverter:
    def __init__(self, model_name="gpt-4o-mini", temperature=0.1):
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=300,
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self.system_prompt = """
You will receive:
• a counting question (Q)
• a template with:
   - "type": equality / at_most / at_least
   - "N": the number to use
   - "answer": yes / no
Write ONE grammatically correct yes/no question using Q as base, inserting N, and respecting singular/plural. Return JSON like:
{"question": "...", "type": "...", "N": ..., "answer": "..."}
"""

    def _make_templates(self, ans: int):
        """Generate all possible templates and return one randomly."""
        N = ans
        templates = [
            {"type": "equality", "N": N, "answer": "yes"},
            {"type": "equality", "N": N + random.randint(1,3) if N != 0 else 2, "answer": "no"},
            {"type": "at_most", "N": N + random.randint(0,2) if N > 0 else 0, "answer": "yes"},
            {"type": "at_least", "N": max(N - random.randint(0,2), 0), "answer": "yes"},
            {"type": "at_least", "N": N + random.randint(1,3), "answer": "no"},
        ]
        if N > 0:
            templates.append({"type": "at_most", "N": max(N - random.randint(1,3), 0), "answer": "no"})
        return random.choice(templates)

    def generate_single_yesno_question(self, question: str, answer: str):
        """Generate a single yes/no question per instance."""
        ans = int(answer)
        template = self._make_templates(ans)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=json.dumps({"Q": question, **template})),
        ]
        while True:
            try:
                response = self.model.invoke(messages)
                qdict = json.loads(response.content)
                qdict["original_question"] = question
                qdict["original_answer"] = answer
                return qdict
            except Exception as e:
                print(f"Error: {e}")
                print(f"Response: {getattr(response, 'content', 'No content')}")

def build_or_load_sample():

    # -- no cache yet: build dataset and cache sample -----------------
    dataset = make_dataset(
        scenes_json      = osp.join(MAIN_PATH, "scenes.json"),
        questions_json   = osp.join(MAIN_PATH, "questions.json"),
        image_root       = osp.join(MAIN_PATH, "images"),
        image_transform  = default_image_transform,
        vocab_json       = osp.join(MAIN_PATH, "vocab.json"),
        output_vocab_json= osp.join(MAIN_PATH, "output-vocab.json"),
        incl_scene       = True,
        incl_raw_scene   = True,
    )
    print(f"Initial dataset size: {len(dataset)}")
    dataset.filter_atmostlatleastlequal()
    print(f"After filtering (conceptual questions only): {len(dataset)}")
    d=[]
    for i in dataset:
        d.append({"question_raw": i["question_raw"], "answer": i["answer"]})

    return dataset


def main():
    # -----------------------------------------------------------------
    # 1.  Get first-N filtered examples (cached if possible)
    # -----------------------------------------------------------------
    dataset = build_or_load_sample()

    # -----------------------------------------------------------------
    # 2.  Convert counting → yes/no using GPT-4o
    # -----------------------------------------------------------------
    converter = GPT4oDirectConverter()
    output_file = "counting_to_yesno_questions_gpt4o.pkl"
    results = {}
    errors = []
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded {len(results)} existing results from {output_file}")

    # Filter dataset to only process remaining items
    remaining_dataset = [d for d in dataset if (d["question_raw"], d["answer"]) not in results]
    print(f"Remaining items to process: {len(remaining_dataset)}")

    if not remaining_dataset:
        print("All items already processed.")
        return

    print("\nConverting counting questions to yes/no questions…")

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {}
        for d in remaining_dataset:
            future = executor.submit(converter.generate_single_yesno_question, d["question_raw"], d["answer"])
            futures[future] = d  # Map future to d for key later if needed

        processed = 0
        for future in tqdm(as_completed(futures), total=len(futures)):
            d = futures[future]  # Not strictly needed since we extract from qdict
            try:
                qdict = future.result()
                key = (qdict["original_question"], qdict["original_answer"])
                results[key] = qdict
                processed += 1
                if processed % 1000 == 0:
                    with open(output_file, "wb") as f:
                        pickle.dump(results, f)
                    print(f"Saved {len(results)} results after {processed} new items.")
            except Exception as e:
                errors.append((d["question_raw"], d["answer"], str(e)))

    # -----------------------------------------------------------------
    # 3.  Persist conversion results & error log
    # -----------------------------------------------------------------
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    print(f"Final save: {len(results)} results.")

    if errors:
        with open("conversion_errors.log", "w") as f:
            for err in errors:
                f.write(f"{err}\n")
        print(f"{len(errors)} errors logged to conversion_errors.log")

def main2():
    from dataset import make_dataset, default_image_transform
    import os.path as osp

    main_path = "train"
    dataset = make_dataset(
        scenes_json=osp.join(main_path, 'scenes.json'),
        questions_json=osp.join(main_path, 'questions.json'),
        # questions_json=osp.join(main_path, 'questions_filtered.json'),
        image_root=osp.join(main_path, 'images'),
        image_transform=default_image_transform,
        vocab_json=osp.join(main_path, 'vocab.json'),
        output_vocab_json=osp.join(main_path, 'output-vocab.json'),
        incl_scene=True,
        incl_raw_scene=True,
    )
    ## filter filter_program_size_raw
    print(len(dataset))  # 699989
    dataset.filter_atmostlatleastlequal()
    print("Only conceptual questions are kept")
    print(len(dataset))  # 71281

if __name__ == "__main__":
    main2()