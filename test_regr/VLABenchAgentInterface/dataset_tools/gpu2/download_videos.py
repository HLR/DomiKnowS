from huggingface_hub import snapshot_download
snapshot_download("VLABench/vlabench_composite_ft_lerobot_video",repo_type="dataset",revision="8ceba7df14015ba00c4572f6df648750364caf18",local_dir="/home/auszok/DomiKnowS/test_regr/VLABenchAgentInterface/data/control_composite",allow_patterns=["videos/**"],max_workers=16)
print("VIDEO_DOWNLOAD_COMPLETE",flush=True)
