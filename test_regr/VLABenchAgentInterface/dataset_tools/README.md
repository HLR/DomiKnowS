# GPU2 dataset maintenance tools

These are the versioned source copies of the code used for the GPU2 dataset repair.
They deliberately target the existing vigorous_easley container, /workspace checkout,
and /home/auszok/experiments/vlabench_dataset_fix_20261029 evidence directory.

Install the tools inside that container before running the commands in
../docs/VLABENCH_DATASETS_GPU2.md:

    cd /workspace
    /opt/dominows/venv/bin/python -m test_regr.VLABenchAgentInterface.dataset_tools.install_gpu2_tools

The installer copies only the listed Python tools and document template. It does
not start downloads, collection, or training. Changed previous tool copies are
preserved under pre_install_sources/.

Datasets, generated scene images/plans, evaluation manifests, checkpoints,
download caches, run logs, and generated audit JSON remain on GPU2 and are not
part of this source directory.

Tool roles:

- download_composite.py: resume the pinned official composite release and validate completeness.
- download_videos.py: optional parallel video-transfer helper; the main downloader also downloads videos.
- generate_planning.py: collect independent initial scenes and expert plan supervision.
- generate_supplemental.py: collect additional task families, retaining unsupported projections separately.
- finalize_datasets.py: validate/freeze independent corpora, check current task LCs, and gate composite use.
- refresh_dataset_report.py: regenerate the inventory and dataset guide from server evidence.
- audit_planner_lcs.py: diagnostic comparison of generic/task constraint acceptance.
- document_template.md: source template for the generated dataset guide.

The tools do not provide trained support for all 19 SkillLib methods.
