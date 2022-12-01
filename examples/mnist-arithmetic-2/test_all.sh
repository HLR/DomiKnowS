mkdir "all_logs"

function move_time_logs() {
	mv "logs" "all_logs/$1"
}

function run_test() {
	echo "python test.py --model_name $1 --checkpoint_path \"checkpoints/$2.pth\" ${@:3}"
	python test.py --model_name $1 --checkpoint_path "checkpoints/$2.pth" ${@:3}
	echo ""
}

function run_test_ILP() {
	echo "python test.py --model_name $1 --checkpoint_path \"checkpoints/$2.pth\" --ILP --no_fixedL ${@:3}"
	python test.py --model_name $1 --checkpoint_path "checkpoints/$2.pth" --ILP --no_fixedL ${@:3}
	echo ""
}

function run_mini_train() {
	echo "python train.py --model_name $1 --num_train 100 ${@:2}"
	python train.py --model_name $1 --num_train 100 ${@:2}
	echo ""
}

# mini training
run_mini_train "Sampling" ${@:1}
move_time_logs "train_samplingloss_10k"

run_mini_train "Semantic" ${@:1}
move_time_logs "train_samplingloss_10k"

run_mini_train "Baseline" ${@:1}
move_time_logs "train_samplingloss_10k"

run_mini_train "DigitLabel" ${@:1}
move_time_logs "train_samplingloss_10k"

run_mini_train "Explicit" ${@:1}
move_time_logs "train_samplingloss_10k"

run_mini_train "PrimalDual" ${@:1}
move_time_logs "train_samplingloss_10k"

# regular tests
run_test "Sampling" "samplingloss_10k" ${@:1}
move_time_logs "samplingloss_10k"

run_test "Sampling" "samplingloss_500" ${@:1}
move_time_logs "samplingloss_500"

run_test "Semantic" "semanticloss_500" ${@:1}
move_time_logs "semanticloss_500"

run_test "Semantic" "semanticloss_10k" ${@:1}
move_time_logs "semanticloss_10k"

run_test "Baseline" "baseline_500" ${@:1}
move_time_logs "baseline_500"

run_test "Baseline" "baseline_10k" ${@:1}
move_time_logs "baseline_10k"

run_test "DigitLabel" "digitlbl_500" ${@:1}
move_time_logs "digitlbl_500"

run_test "DigitLabel" "digitlbl_10k" ${@:1}
move_time_logs "digitlbl_10k"

run_test "Explicit" "explicit_500" ${@:1}
move_time_logs "explicit_500"

run_test "Explicit" "explicit_10k" ${@:1}
move_time_logs "explicit_10k"

run_test "PrimalDual" "primaldual_500" ${@:1}
move_time_logs "primaldual_500"

run_test "PrimalDual" "primaldual_10k" ${@:1}
move_time_logs "primaldual_10k"

# ILP
run_test_ILP "Baseline" "baseline_500" ${@:1}
move_time_logs "baseline_500_ILP"

run_test_ILP "Baseline" "baseline_10k" ${@:1}
move_time_logs "baseline_10k_ILP"

run_test_ILP "Explicit" "explicit_500" ${@:1}
move_time_logs "explicit_500_ILP"

run_test_ILP "Explicit" "explicit_10k" ${@:1}
move_time_logs "explicit_10k_ILP"
