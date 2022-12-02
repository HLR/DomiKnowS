input="""INFO:regr.program.program: - loss:
INFO:regr.program.program:{'zerohandwriting': 0.044410474598407745, 'onehandwriting': 0.03320235759019852, 'twohandwriting': 0.1322178989648819, 'threehandwriting': 0.16238872706890106, 'fourhandwriting': 0.0793474093079567, 'fivehandwriting': 0.1254628598690033, 'sixhandwriting': 0.07529517263174057, 'sevenhandwriting': 0.11403413116931915, 'eighthandwriting': 0.167806476354599, 'ninehandwriting': 0.15762443840503693}
INFO:regr.program.program: - Constraint loss:
INFO:regr.program.program:None
INFO:regr.program.program: - metric:
INFO:regr.program.program: - - ILP
INFO:regr.program.program:{'zerohandwriting': {'P': 0.9629629629629629, 'R': 0.9533333333333334, 'F1': 0.9581239530988274, 'accuracy': 0.9916666666666667}, 'onehandwriting': {'P': 0.921996879875195, 'R': 0.985, 'F1': 0.9524576954069299, 'accuracy': 0.9901666666666666}, 'twohandwriting': {'P': 0.9612068965517241, 'R': 0.7433333333333333, 'F1': 0.8383458646616541, 'accuracy': 0.9713333333333334}, 'threehandwriting': {'P': 0.7032258064516129, 'R': 0.9083333333333333, 'F1': 0.7927272727272727, 'accuracy': 0.9525}, 'fourhandwriting': {'P': 0.8566978193146417, 'R': 0.9166666666666666, 'F1': 0.8856682769726247, 'accuracy': 0.9763333333333334}, 'fivehandwriting': {'P': 0.8213114754098361, 'R': 0.835, 'F1': 0.828099173553719, 'accuracy': 0.9653333333333334}, 'sixhandwriting': {'P': 0.9676840215439856, 'R': 0.8983333333333333, 'F1': 0.9317199654278305, 'accuracy': 0.9868333333333333}, 'sevenhandwriting': {'P': 0.7805907172995781, 'R': 0.925, 'F1': 0.8466819221967964, 'accuracy': 0.9665}, 'eighthandwriting': {'P': 0.8705179282868526, 'R': 0.7283333333333334, 'F1': 0.7931034482758621, 'accuracy': 0.962}, 'ninehandwriting': {'P': 0.9186507936507936, 'R': 0.7716666666666666, 'F1': 0.8387681159420289, 'accuracy': 0.9703333333333334}}
INFO:regr.program.program: - - argmax
INFO:regr.program.program:{'zerohandwriting': {'P': 0.9578207381370826, 'R': 0.9083333333333333, 'F1': 0.932420872540633, 'accuracy': 0.9868333333333333}, 'onehandwriting': {'P': 0.9380952380952381, 'R': 0.985, 'F1': 0.9609756097560974, 'accuracy': 0.992}, 'twohandwriting': {'P': 0.9790209790209791, 'R': 0.7, 'F1': 0.8163265306122449, 'accuracy': 0.9685}, 'threehandwriting': {'P': 0.6682867557715675, 'R': 0.9166666666666666, 'F1': 0.7730147575544624, 'accuracy': 0.9461666666666667}, 'fourhandwriting': {'P': 0.8846153846153846, 'R': 0.8816666666666667, 'F1': 0.8831385642737896, 'accuracy': 0.9766666666666667}, 'fivehandwriting': {'P': 0.7658610271903323, 'R': 0.845, 'F1': 0.8034865293185419, 'accuracy': 0.9586666666666667}, 'sixhandwriting': {'P': 0.974609375, 'R': 0.8316666666666667, 'F1': 0.8974820143884893, 'accuracy': 0.981}, 'sevenhandwriting': {'P': 0.7652777777777777, 'R': 0.9183333333333333, 'F1': 0.8348484848484848, 'accuracy': 0.9636666666666667}, 'eighthandwriting': {'P': 0.8624338624338624, 'R': 0.5433333333333333, 'F1': 0.6666666666666666, 'accuracy': 0.9456666666666667}, 'ninehandwriting': {'P': 0.8871794871794871, 'R': 0.5766666666666667, 'F1': 0.6989898989898989, 'accuracy': 0.9503333333333334}}
constraint accuracy:  78.8
Log file for dataNode is in: /home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/examples/MNIST_binary/logs/datanode.log
/opt/conda/lib/python3.7/site-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ../torch/csrc/utils/tensor_numpy.cpp:178.)
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
device is :  cuda:2
POI
ILP
SAM
SAM  and ILP
INFO:regr.program.program:Testing:
Testing:   0%|                                           | 0/20 [00:00<?, ?it/s]Log file for ilpOntSolver is in: /home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/examples/MNIST_binary/logs/ilpOntSolver.log
Log file for ilpOntSolverTime is in: /home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/examples/MNIST_binary/logs/ilpOntSolver.log
Testing: 100%|██████████████████████████████████| 20/20 [01:21<00:00,  4.08s/it]
INFO:regr.program.program: - loss:
INFO:regr.program.program:{'zerohandwriting': 0.044001005589962006, 'onehandwriting': 0.04704561084508896, 'twohandwriting': 0.10337456315755844, 'threehandwriting': 0.24713821709156036, 'fourhandwriting': 0.08190925419330597, 'fivehandwriting': 0.13234896957874298, 'sixhandwriting': 0.07551966607570648, 'sevenhandwriting': 0.11245449632406235, 'eighthandwriting': 0.16174361109733582, 'ninehandwriting': 0.16142304241657257}
INFO:regr.program.program: - Constraint loss:
INFO:regr.program.program:None
INFO:regr.program.program: - metric:
INFO:regr.program.program: - - ILP
INFO:regr.program.program:{'zerohandwriting': {'P': 0.9629005059021922, 'R': 0.9516666666666667, 'F1': 0.9572506286672254, 'accuracy': 0.9915}, 'onehandwriting': {'P': 0.9078341013824884, 'R': 0.985, 'F1': 0.9448441247002398, 'accuracy': 0.9885}, 'twohandwriting': {'P': 0.9259259259259259, 'R': 0.75, 'F1': 0.8287292817679557, 'accuracy': 0.969}, 'threehandwriting': {'P': 0.6428571428571429, 'R': 0.915, 'F1': 0.7551581843191197, 'accuracy': 0.9406666666666667}, 'fourhandwriting': {'P': 0.8450920245398773, 'R': 0.9183333333333333, 'F1': 0.8801916932907349, 'accuracy': 0.975}, 'fivehandwriting': {'P': 0.8324958123953099, 'R': 0.8283333333333334, 'F1': 0.8304093567251463, 'accuracy': 0.9661666666666666}, 'sixhandwriting': {'P': 0.9658886894075404, 'R': 0.8966666666666666, 'F1': 0.929991356957649, 'accuracy': 0.9865}, 'sevenhandwriting': {'P': 0.8139880952380952, 'R': 0.9116666666666666, 'F1': 0.860062893081761, 'accuracy': 0.9703333333333334}, 'eighthandwriting': {'P': 0.8898488120950324, 'R': 0.6866666666666666, 'F1': 0.77516462841016, 'accuracy': 0.9601666666666666}, 'ninehandwriting': {'P': 0.9305263157894736, 'R': 0.7366666666666667, 'F1': 0.8223255813953487, 'accuracy': 0.9681666666666666}}
INFO:regr.program.program: - - argmax
INFO:regr.program.program:{'zerohandwriting': {'P': 0.9589285714285715, 'R': 0.895, 'F1': 0.9258620689655173, 'accuracy': 0.9856666666666667}, 'onehandwriting': {'P': 0.9022900763358779, 'R': 0.985, 'F1': 0.9418326693227091, 'accuracy': 0.9878333333333333}, 'twohandwriting': {'P': 0.935871743486974, 'R': 0.7783333333333333, 'F1': 0.8498635122838946, 'accuracy': 0.9725}, 'threehandwriting': {'P': 0.5510805500982319, 'R': 0.935, 'F1': 0.6934487021013598, 'accuracy': 0.9173333333333333}, 'fourhandwriting': {'P': 0.85553772070626, 'R': 0.8883333333333333, 'F1': 0.8716271463614064, 'accuracy': 0.9738333333333333}, 'fivehandwriting': {'P': 0.7425742574257426, 'R': 0.875, 'F1': 0.8033664881407805, 'accuracy': 0.9571666666666667}, 'sixhandwriting': {'P': 0.9655172413793104, 'R': 0.84, 'F1': 0.8983957219251336, 'accuracy': 0.981}, 'sevenhandwriting': {'P': 0.7724719101123596, 'R': 0.9166666666666666, 'F1': 0.8384146341463415, 'accuracy': 0.9646666666666667}, 'eighthandwriting': {'P': 0.8532338308457711, 'R': 0.5716666666666667, 'F1': 0.6846307385229541, 'accuracy': 0.9473333333333334}, 'ninehandwriting': {'P': 0.8983516483516484, 'R': 0.545, 'F1': 0.6784232365145229, 'accuracy': 0.9483333333333334}}
constraint accuracy:  77.13333333333333"""
p = '[\d]+[.\d]+|[\d]*[.][\d]+|[\d]+'
import re
for i in input.split("\n"):
    node=list(re.findall(p, i))
    if len(node)==1:
        print("constraint:",node[0])
    if len(node)>10:
        print("f1",sum([float(node[j+2]) for j in range(0,len(node)-2,4)])/len(node)*4)