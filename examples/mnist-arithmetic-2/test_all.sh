mkdir "all_logs"

function move_time_logs() {
	mv "logs" "all_logs/$1"
}

echo "python test.py --model_name Sampling --checkpoint_path 'checkpoints/samplingloss_500.pth'"
python test.py --model_name Sampling --checkpoint_path 'checkpoints/samplingloss_500.pth'
echo ""
move_time_logs "samplingloss_500"

echo "python test.py --model_name Sampling --checkpoint_path 'checkpoints/samplingloss_10k.pth'"
python test.py --model_name Sampling --checkpoint_path 'checkpoints/samplingloss_10k.pth'
echo ""
move_time_logs "samplingloss_10k"

echo "python test.py --model_name Semantic --checkpoint_path 'checkpoints/semanticloss_500.pth'"
python test.py --model_name Semantic --checkpoint_path 'checkpoints/semanticloss_500.pth'
echo ""
move_time_logs "semanticloss_500"

echo "python test.py --model_name Semantic --checkpoint_path 'checkpoints/semanticloss_10k.pth'"
python test.py --model_name Semantic --checkpoint_path 'checkpoints/semanticloss_10k.pth'
echo ""
move_time_logs "semanticloss_10k"

echo "python test.py --model_name Baseline --checkpoint_path 'checkpoints/baseline_500.pth'"
python test.py --model_name Baseline --checkpoint_path 'checkpoints/baseline_500.pth'
echo ""
move_time_logs "baseline_500"

echo "python test.py --model_name Baseline --checkpoint_path 'checkpoints/baseline_10k.pth'"
python test.py --model_name Baseline --checkpoint_path 'checkpoints/baseline_10k.pth'
echo ""
move_time_logs "baseline_10k"

echo "python test.py --model_name DigitLabel --checkpoint_path 'checkpoints/digitlbl_500.pth'"
python test.py --model_name DigitLabel --checkpoint_path 'checkpoints/digitlbl_500.pth'
echo ""
move_time_logs "digitlbl_500"

echo "python test.py --model_name DigitLabel --checkpoint_path 'checkpoints/digitlbl_10k.pth'"
python test.py --model_name DigitLabel --checkpoint_path 'checkpoints/digitlbl_10k.pth'
echo ""
move_time_logs "digitlbl_10k"

echo "python test.py --model_name Explicit --checkpoint_path 'checkpoints/explicit_500.pth'"
python test.py --model_name Explicit --checkpoint_path 'checkpoints/explicit_500.pth'
echo ""
move_time_logs "explicit_500"

echo "python test.py --model_name Explicit --checkpoint_path 'checkpoints/explicit_10k.pth'"
python test.py --model_name Explicit --checkpoint_path 'checkpoints/explicit_10k.pth'
echo ""
move_time_logs "explicit_10k"

echo "python test.py --model_name PrimalDual --checkpoint_path 'checkpoints/primaldual_500.pth'"
python test.py --model_name PrimalDual --checkpoint_path 'checkpoints/primaldual_500.pth'
echo ""
move_time_logs "primaldual_500"

echo "python test.py --model_name PrimalDual --checkpoint_path 'checkpoints/primaldual_10k.pth'"
python test.py --model_name PrimalDual --checkpoint_path 'checkpoints/primaldual_10k.pth'
echo ""
move_time_logs "primaldual_10k"

echo "python test.py --model_name Baseline --checkpoint_path 'checkpoints/baseline_500.pth' --ILP --no_fixedL"
python test.py --model_name Baseline --checkpoint_path 'checkpoints/baseline_500.pth' --ILP --no_fixedL
echo ""
move_time_logs "baseline_500_ILP"

echo "python test.py --model_name Baseline --checkpoint_path 'checkpoints/baseline_10k.pth' --ILP --no_fixedL"
python test.py --model_name Baseline --checkpoint_path 'checkpoints/baseline_10k.pth' --ILP --no_fixedL
echo ""
move_time_logs "baseline_10k_ILP"

echo "python test.py --model_name Explicit --checkpoint_path 'checkpoints/explicit_500.pth' --ILP --no_fixedL"
python test.py --model_name Explicit --checkpoint_path 'checkpoints/explicit_500.pth' --ILP --no_fixedL
echo ""
move_time_logs "explicit_500_ILP"

echo "python test.py --model_name Explicit --checkpoint_path 'checkpoints/explicit_10k.pth' --ILP --no_fixedL"
python test.py --model_name Explicit --checkpoint_path 'checkpoints/explicit_10k.pth' --ILP --no_fixedL
echo ""
move_time_logs "explicit_10k_ILP"
