"""The learned task concept is fitted on training episodes only and replaces every task-derived input."""
import json

import pytest

from ..dataset import PlanningExample, deterministic_split, with_predicted_task
from ..knowledge.task_classifier import TaskClassifier, accuracy, fit_task_classifier, predict_tasks

PHRASES = {
    "select_fruit": "pick the {} fruit from the table and put it on the plate",
    "friction_qa": "which surface has the highest friction press the button of the {} one",
    "open_drawer": "open the {} drawer of the cabinet",
}


def make_examples(count=12):
    colours = ["red", "green", "yellow", "small", "large", "left", "right", "top", "bottom", "old", "new", "blue"]
    return [PlanningExample(episode_id=f"{task}/example{i}", instruction=text.format(colours[i % len(colours)]),
                            operation_sequence=(), image_paths=(), segmented_image_paths=(), env_config_path=None,
                            task=task)
            for task, text in PHRASES.items() for i in range(count)]


def test_classifier_is_fitted_on_the_training_split_and_generalises():
    examples = make_examples()
    splits = deterministic_split(examples, seed=7)
    classifier = fit_task_classifier(splits["train"])
    assert classifier.train_episodes == {e.episode_id for e in splits["train"]}
    assert accuracy(classifier, splits["test"])["accuracy"] == 1.0
    classifier.assert_disjoint(splits["test"])
    with pytest.raises(ValueError):
        classifier.assert_disjoint(splits["train"][:1])


def test_classifier_round_trips_and_abstains_below_confidence(tmp_path):
    classifier = fit_task_classifier(make_examples(), epochs=50)
    classifier.save(tmp_path / "c.pt")
    loaded = TaskClassifier.load(tmp_path / "c.pt")
    assert loaded.classes == classifier.classes and loaded.train_episodes == classifier.train_episodes
    assert loaded.predict("open the red drawer") == "open_drawer"
    assert loaded.predict("open the red drawer", min_confidence=1.01) is None


def test_prediction_requires_a_classifier_and_a_disjoint_split(tmp_path):
    examples = make_examples()
    with pytest.raises(ValueError):
        predict_tasks(examples, None)
    classifier = fit_task_classifier(examples[:12], epochs=20)
    classifier.save(tmp_path / "c.pt")
    with pytest.raises(ValueError):
        predict_tasks(examples[:3], tmp_path / "c.pt")


def test_with_predicted_task_drops_the_episode_task_facts(tmp_path):
    config_dir = tmp_path / "ex" / "env_config"
    config_dir.mkdir(parents=True)
    config = config_dir / "c.json"
    config.write_text(json.dumps({"task": {"components": []}}))
    base = PlanningExample(
        episode_id="select_fruit/example0", instruction="pick the fruit", operation_sequence=(), image_paths=(),
        segmented_image_paths=(), env_config_path=config, task="select_fruit", task_skill_sequence=("pick", "place"),
        task_target_entity="apple", task_target_container="plate", planning_facts={"required_skill_sequence": ["pick"]})
    rebuilt = with_predicted_task(base, "select_toy")
    assert rebuilt.predicted_task == "select_toy" and rebuilt.task == "select_fruit"  # task kept only for scoring
    assert rebuilt.task_skill_sequence is None
    assert rebuilt.task_target_entity is None and rebuilt.task_target_container is None
    assert rebuilt.planning_facts != base.planning_facts
