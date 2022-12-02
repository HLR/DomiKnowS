input="""INFO:regr.program.program:Testing:
Testing:   0%|                                           | 0/20 [00:00<?, ?it/s]Log file for ilpOntSolver is in: /home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/examples/MNIST_binary/logs/ilpOntSolver.log
Log file for ilpOntSolverTime is in: /home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/examples/MNIST_binary/logs/ilpOntSolver.log
Testing: 100%|██████████████████████████████████| 20/20 [01:18<00:00,  3.92s/it]
INFO:regr.program.program: - loss:
INFO:regr.program.program:{'zerohandwriting': 0.04310394078493118, 'onehandwriting': 0.029491741210222244, 'twohandwriting': 0.10278086364269257, 'threehandwriting': 0.20871055126190186, 'fourhandwriting': 0.07852588593959808, 'fivehandwriting': 0.11961068212985992, 'sixhandwriting': 0.09607396274805069, 'sevenhandwriting': 0.09617417305707932, 'eighthandwriting': 0.18352486193180084, 'ninehandwriting': 0.13722661137580872}
INFO:regr.program.program: - Constraint loss:
INFO:regr.program.program:None
INFO:regr.program.program: - metric:
INFO:regr.program.program: - - ILP
INFO:regr.program.program:{'zerohandwriting': {'P': 0.9525368248772504, 'R': 0.97, 'F1': 0.9611890999174235, 'accuracy': 0.9921666666666666}, 'onehandwriting': {'P': 0.937799043062201, 'R': 0.98, 'F1': 0.9584352078239609, 'accuracy': 0.9915}, 'twohandwriting': {'P': 0.8756756756756757, 'R': 0.81, 'F1': 0.8415584415584417, 'accuracy': 0.9695}, 'threehandwriting': {'P': 0.6543942992874109, 'R': 0.9183333333333333, 'F1': 0.7642163661581137, 'accuracy': 0.9433333333333334}, 'fourhandwriting': {'P': 0.8525345622119815, 'R': 0.925, 'F1': 0.8872901678657075, 'accuracy': 0.9765}, 'fivehandwriting': {'P': 0.855379188712522, 'R': 0.8083333333333333, 'F1': 0.831191088260497, 'accuracy': 0.9671666666666666}, 'sixhandwriting': {'P': 0.9775280898876404, 'R': 0.87, 'F1': 0.9206349206349206, 'accuracy': 0.985}, 'sevenhandwriting': {'P': 0.8637820512820513, 'R': 0.8983333333333333, 'F1': 0.8807189542483661, 'accuracy': 0.9756666666666667}, 'eighthandwriting': {'P': 0.8948545861297539, 'R': 0.6666666666666666, 'F1': 0.764087870105062, 'accuracy': 0.9588333333333333}, 'ninehandwriting': {'P': 0.8966789667896679, 'R': 0.81, 'F1': 0.851138353765324, 'accuracy': 0.9716666666666667}}
INFO:regr.program.program: - - argmax
INFO:regr.program.program:{'zerohandwriting': {'P': 0.9396984924623115, 'R': 0.935, 'F1': 0.93734335839599, 'accuracy': 0.9875}, 'onehandwriting': {'P': 0.95, 'R': 0.9816666666666667, 'F1': 0.9655737704918033, 'accuracy': 0.993}, 'twohandwriting': {'P': 0.8738898756660746, 'R': 0.82, 'F1': 0.8460877042132415, 'accuracy': 0.9701666666666666}, 'threehandwriting': {'P': 0.5972073039742213, 'R': 0.9266666666666666, 'F1': 0.7263226649248856, 'accuracy': 0.9301666666666667}, 'fourhandwriting': {'P': 0.8612903225806452, 'R': 0.89, 'F1': 0.8754098360655737, 'accuracy': 0.9746666666666667}, 'fivehandwriting': {'P': 0.7905511811023622, 'R': 0.8366666666666667, 'F1': 0.8129554655870446, 'accuracy': 0.9615}, 'sixhandwriting': {'P': 0.9832985386221295, 'R': 0.785, 'F1': 0.8730305838739574, 'accuracy': 0.9771666666666666}, 'sevenhandwriting': {'P': 0.8387596899224806, 'R': 0.9016666666666666, 'F1': 0.8690763052208835, 'accuracy': 0.9728333333333333}, 'eighthandwriting': {'P': 0.8567335243553008, 'R': 0.49833333333333335, 'F1': 0.6301369863013698, 'accuracy': 0.9415}, 'ninehandwriting': {'P': 0.825, 'R': 0.715, 'F1': 0.7660714285714285, 'accuracy': 0.9563333333333334}}
constraint accuracy:  78.48333333333333
Log file for dataNode is in: /home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/examples/MNIST_binary/logs/datanode.log
/opt/conda/lib/python3.7/site-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ../torch/csrc/utils/tensor_numpy.cpp:178.)
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
device is :  cuda:1
POI
ILP
SAM
SAM  and ILP
INFO:regr.program.program:Testing:
Testing:   0%|                                           | 0/20 [00:00<?, ?it/s]Log file for ilpOntSolver is in: /home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/examples/MNIST_binary/logs/ilpOntSolver.log
Log file for ilpOntSolverTime is in: /home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/examples/MNIST_binary/logs/ilpOntSolver.log
Testing: 100%|██████████████████████████████████| 20/20 [01:17<00:00,  3.86s/it]
INFO:regr.program.program: - loss:
INFO:regr.program.program:{'zerohandwriting': 0.04677654057741165, 'onehandwriting': 0.03292345255613327, 'twohandwriting': 0.09275341033935547, 'threehandwriting': 0.19354824721813202, 'fourhandwriting': 0.08048701286315918, 'fivehandwriting': 0.12034336477518082, 'sixhandwriting': 0.09766090661287308, 'sevenhandwriting': 0.12131760269403458, 'eighthandwriting': 0.16398939490318298, 'ninehandwriting': 0.1672818958759308}
INFO:regr.program.program: - Constraint loss:
INFO:regr.program.program:None
INFO:regr.program.program: - metric:
INFO:regr.program.program: - - ILP
INFO:regr.program.program:{'zerohandwriting': {'P': 0.9630252100840336, 'R': 0.955, 'F1': 0.9589958158995815, 'accuracy': 0.9918333333333333}, 'onehandwriting': {'P': 0.9305993690851735, 'R': 0.9833333333333333, 'F1': 0.9562398703403565, 'accuracy': 0.991}, 'twohandwriting': {'P': 0.9376218323586745, 'R': 0.8016666666666666, 'F1': 0.8643306379155434, 'accuracy': 0.9748333333333333}, 'threehandwriting': {'P': 0.6654589371980676, 'R': 0.9183333333333333, 'F1': 0.7717086834733894, 'accuracy': 0.9456666666666667}, 'fourhandwriting': {'P': 0.825, 'R': 0.935, 'F1': 0.8765625, 'accuracy': 0.9736666666666667}, 'fivehandwriting': {'P': 0.856637168141593, 'R': 0.8066666666666666, 'F1': 0.8309012875536481, 'accuracy': 0.9671666666666666}, 'sixhandwriting': {'P': 0.9776119402985075, 'R': 0.8733333333333333, 'F1': 0.9225352112676056, 'accuracy': 0.9853333333333333}, 'sevenhandwriting': {'P': 0.7780898876404494, 'R': 0.9233333333333333, 'F1': 0.8445121951219512, 'accuracy': 0.966}, 'eighthandwriting': {'P': 0.8879668049792531, 'R': 0.7133333333333334, 'F1': 0.7911275415896488, 'accuracy': 0.9623333333333334}, 'ninehandwriting': {'P': 0.9362637362637363, 'R': 0.71, 'F1': 0.8075829383886256, 'accuracy': 0.9661666666666666}}
INFO:regr.program.program: - - argmax
INFO:regr.program.program:{'zerohandwriting': {'P': 0.9575221238938053, 'R': 0.9016666666666666, 'F1': 0.9287553648068669, 'accuracy': 0.9861666666666666}, 'onehandwriting': {'P': 0.9395866454689984, 'R': 0.985, 'F1': 0.9617575264442637, 'accuracy': 0.9921666666666666}, 'twohandwriting': {'P': 0.9411764705882353, 'R': 0.8, 'F1': 0.8648648648648648, 'accuracy': 0.975}, 'threehandwriting': {'P': 0.6125827814569537, 'R': 0.925, 'F1': 0.7370517928286852, 'accuracy': 0.934}, 'fourhandwriting': {'P': 0.8592, 'R': 0.895, 'F1': 0.876734693877551, 'accuracy': 0.9748333333333333}, 'fivehandwriting': {'P': 0.7990275526742301, 'R': 0.8216666666666667, 'F1': 0.810188989317995, 'accuracy': 0.9615}, 'sixhandwriting': {'P': 0.9935897435897436, 'R': 0.775, 'F1': 0.8707865168539327, 'accuracy': 0.977}, 'sevenhandwriting': {'P': 0.7496598639455783, 'R': 0.9183333333333333, 'F1': 0.8254681647940076, 'accuracy': 0.9611666666666666}, 'eighthandwriting': {'P': 0.8448275862068966, 'R': 0.5716666666666667, 'F1': 0.6819085487077535, 'accuracy': 0.9466666666666667}, 'ninehandwriting': {'P': 0.9037900874635568, 'R': 0.5166666666666667, 'F1': 0.6574761399787912, 'accuracy': 0.9461666666666667}}
constraint accuracy:  77.75
Log file for dataNode is in: /home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/examples/MNIST_binary/logs/datanode.log
/opt/conda/lib/python3.7/site-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ../torch/csrc/utils/tensor_numpy.cpp:178.)
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
device is :  cuda:1
POI
ILP
SAM
SAM  and ILP
INFO:regr.program.program:Testing:
Testing:   0%|                                           | 0/20 [00:00<?, ?it/s]Log file for ilpOntSolver is in: /home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/examples/MNIST_binary/logs/ilpOntSolver.log
Log file for ilpOntSolverTime is in: /home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/examples/MNIST_binary/logs/ilpOntSolver.log
Testing: 100%|██████████████████████████████████| 20/20 [01:18<00:00,  3.91s/it]
INFO:regr.program.program: - loss:
INFO:regr.program.program:{'zerohandwriting': 0.046693649142980576, 'onehandwriting': 0.046517156064510345, 'twohandwriting': 0.12063489109277725, 'threehandwriting': 0.14082112908363342, 'fourhandwriting': 0.08375473320484161, 'fivehandwriting': 0.13074332475662231, 'sixhandwriting': 0.07740305364131927, 'sevenhandwriting': 0.13834071159362793, 'eighthandwriting': 0.1575535088777542, 'ninehandwriting': 0.1650066077709198}
INFO:regr.program.program: - Constraint loss:
INFO:regr.program.program:None
INFO:regr.program.program: - metric:
INFO:regr.program.program: - - ILP
INFO:regr.program.program:{'zerohandwriting': {'P': 0.9516666666666667, 'R': 0.9516666666666667, 'F1': 0.9516666666666667, 'accuracy': 0.9903333333333333}, 'onehandwriting': {'P': 0.8822652757078987, 'R': 0.9866666666666667, 'F1': 0.9315499606608969, 'accuracy': 0.9855}, 'twohandwriting': {'P': 0.9194139194139194, 'R': 0.8366666666666667, 'F1': 0.8760907504363001, 'accuracy': 0.9763333333333334}, 'threehandwriting': {'P': 0.9098196392785571, 'R': 0.7566666666666667, 'F1': 0.8262056414922658, 'accuracy': 0.9681666666666666}, 'fourhandwriting': {'P': 0.8458015267175573, 'R': 0.9233333333333333, 'F1': 0.8828685258964143, 'accuracy': 0.9755}, 'fivehandwriting': {'P': 0.7679083094555874, 'R': 0.8933333333333333, 'F1': 0.8258859784283513, 'accuracy': 0.9623333333333334}, 'sixhandwriting': {'P': 0.9072847682119205, 'R': 0.9133333333333333, 'F1': 0.9102990033222591, 'accuracy': 0.982}, 'sevenhandwriting': {'P': 0.9208103130755064, 'R': 0.8333333333333334, 'F1': 0.8748906386701663, 'accuracy': 0.9761666666666666}, 'eighthandwriting': {'P': 0.8022690437601296, 'R': 0.825, 'F1': 0.8134757600657353, 'accuracy': 0.9621666666666666}, 'ninehandwriting': {'P': 0.8518518518518519, 'R': 0.805, 'F1': 0.8277634961439589, 'accuracy': 0.9665}}
INFO:regr.program.program: - - argmax
INFO:regr.program.program:{'zerohandwriting': {'P': 0.9489795918367347, 'R': 0.93, 'F1': 0.9393939393939393, 'accuracy': 0.988}, 'onehandwriting': {'P': 0.9078341013824884, 'R': 0.985, 'F1': 0.9448441247002398, 'accuracy': 0.9885}, 'twohandwriting': {'P': 0.9479166666666666, 'R': 0.7583333333333333, 'F1': 0.8425925925925927, 'accuracy': 0.9716666666666667}, 'threehandwriting': {'P': 0.9290617848970252, 'R': 0.6766666666666666, 'F1': 0.7830279652844744, 'accuracy': 0.9625}, 'fourhandwriting': {'P': 0.864297253634895, 'R': 0.8916666666666667, 'F1': 0.8777686628383922, 'accuracy': 0.9751666666666666}, 'fivehandwriting': {'P': 0.7331460674157303, 'R': 0.87, 'F1': 0.7957317073170732, 'accuracy': 0.9553333333333334}, 'sixhandwriting': {'P': 0.9090909090909091, 'R': 0.8833333333333333, 'F1': 0.8960270498732037, 'accuracy': 0.9795}, 'sevenhandwriting': {'P': 0.9385245901639344, 'R': 0.7633333333333333, 'F1': 0.8419117647058824, 'accuracy': 0.9713333333333334}, 'eighthandwriting': {'P': 0.7517605633802817, 'R': 0.7116666666666667, 'F1': 0.7311643835616438, 'accuracy': 0.9476666666666667}, 'ninehandwriting': {'P': 0.812133072407045, 'R': 0.6916666666666667, 'F1': 0.747074707470747, 'accuracy': 0.9531666666666667}}
constraint accuracy:  80.63333333333334
Log file for dataNode is in: /home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/examples/MNIST_binary/logs/datanode.log
/opt/conda/lib/python3.7/site-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ../torch/csrc/utils/tensor_numpy.cpp:178.)
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
device is :  cuda:1
POI
ILP
SAM
SAM  and ILP
INFO:regr.program.program:Testing:
Testing:   0%|                                           | 0/20 [00:00<?, ?it/s]Log file for ilpOntSolver is in: /home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/examples/MNIST_binary/logs/ilpOntSolver.log
Log file for ilpOntSolverTime is in: /home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/examples/MNIST_binary/logs/ilpOntSolver.log
Testing: 100%|██████████████████████████████████| 20/20 [01:16<00:00,  3.82s/it]
INFO:regr.program.program: - loss:
INFO:regr.program.program:{'zerohandwriting': 0.04669393599033356, 'onehandwriting': 0.032742939889431, 'twohandwriting': 0.09276562929153442, 'threehandwriting': 0.19596122205257416, 'fourhandwriting': 0.08013009279966354, 'fivehandwriting': 0.12057779729366302, 'sixhandwriting': 0.09483986347913742, 'sevenhandwriting': 0.1241411417722702, 'eighthandwriting': 0.16472308337688446, 'ninehandwriting': 0.16554391384124756}
INFO:regr.program.program: - Constraint loss:
INFO:regr.program.program:None
INFO:regr.program.program: - metric:
INFO:regr.program.program: - - ILP
INFO:regr.program.program:{'zerohandwriting': {'P': 0.9630872483221476, 'R': 0.9566666666666667, 'F1': 0.959866220735786, 'accuracy': 0.992}, 'onehandwriting': {'P': 0.9276729559748428, 'R': 0.9833333333333333, 'F1': 0.9546925566343042, 'accuracy': 0.9906666666666667}, 'twohandwriting': {'P': 0.9375, 'R': 0.8, 'F1': 0.8633093525179856, 'accuracy': 0.9746666666666667}, 'threehandwriting': {'P': 0.6614645858343338, 'R': 0.9183333333333333, 'F1': 0.7690160502442429, 'accuracy': 0.9448333333333333}, 'fourhandwriting': {'P': 0.834575260804769, 'R': 0.9333333333333333, 'F1': 0.8811959087332809, 'accuracy': 0.9748333333333333}, 'fivehandwriting': {'P': 0.8538324420677362, 'R': 0.7983333333333333, 'F1': 0.8251507321274762, 'accuracy': 0.9661666666666666}, 'sixhandwriting': {'P': 0.9777777777777777, 'R': 0.88, 'F1': 0.9263157894736842, 'accuracy': 0.986}, 'sevenhandwriting': {'P': 0.7658402203856749, 'R': 0.9266666666666666, 'F1': 0.8386123680241326, 'accuracy': 0.9643333333333334}, 'eighthandwriting': {'P': 0.8888888888888888, 'R': 0.7066666666666667, 'F1': 0.7873723305478181, 'accuracy': 0.9618333333333333}, 'ninehandwriting': {'P': 0.9352678571428571, 'R': 0.6983333333333334, 'F1': 0.7996183206106869, 'accuracy': 0.965}}
INFO:regr.program.program: - - argmax
INFO:regr.program.program:{'zerohandwriting': {'P': 0.9575221238938053, 'R': 0.9016666666666666, 'F1': 0.9287553648068669, 'accuracy': 0.9861666666666666}, 'onehandwriting': {'P': 0.9395866454689984, 'R': 0.985, 'F1': 0.9617575264442637, 'accuracy': 0.9921666666666666}, 'twohandwriting': {'P': 0.9450980392156862, 'R': 0.8033333333333333, 'F1': 0.8684684684684684, 'accuracy': 0.9756666666666667}, 'threehandwriting': {'P': 0.6092206366630076, 'R': 0.925, 'F1': 0.7346128391793515, 'accuracy': 0.9331666666666667}, 'fourhandwriting': {'P': 0.8610223642172524, 'R': 0.8983333333333333, 'F1': 0.8792822185970637, 'accuracy': 0.9753333333333334}, 'fivehandwriting': {'P': 0.8006535947712419, 'R': 0.8166666666666667, 'F1': 0.8085808580858085, 'accuracy': 0.9613333333333334}, 'sixhandwriting': {'P': 0.9894291754756871, 'R': 0.78, 'F1': 0.8723205964585277, 'accuracy': 0.9771666666666666}, 'sevenhandwriting': {'P': 0.745945945945946, 'R': 0.92, 'F1': 0.8238805970149254, 'accuracy': 0.9606666666666667}, 'eighthandwriting': {'P': 0.8527918781725888, 'R': 0.56, 'F1': 0.6760563380281691, 'accuracy': 0.9463333333333334}, 'ninehandwriting': {'P': 0.9005681818181818, 'R': 0.5283333333333333, 'F1': 0.6659663865546219, 'accuracy': 0.947}}
constraint accuracy:  78.03333333333333
Log file for dataNode is in: /home/hlr/storage/egr/research-hlr/nafarali/new_meta/DomiKnowS/examples/MNIST_binary/logs/datanode.log
/opt/conda/lib/python3.7/site-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ../torch/csrc/utils/tensor_numpy.cpp:178.)
  return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
device is :  cuda:1
POI"""
p = '[\d]+[.\d]+|[\d]*[.][\d]+|[\d]+'
import re
for i in input.split("\n"):
    node=list(re.findall(p, i))
    if len(node)==1:
        print("constraint:",node[0])
    if len(node)>10:
        print("f1",sum([float(node[j+2]) for j in range(0,len(node)-2,4)])/len(node)*4)