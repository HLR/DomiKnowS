To run:

```
bash download_checkpoints.sh
bash test_all.sh --cuda --log TimeOnly | tee test_out.txt
```

Logs will be stored in `test_out.txt` and the `all_logs/` folder.

Single model test:
```
bash download_checkpoints.sh
python test.py --model_name PrimalDual --checkpoint_path checkpoints/primaldual_10k.pth/model.pth
```

