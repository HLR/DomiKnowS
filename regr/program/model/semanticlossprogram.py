program = SampleLossProgram(
        graph, SolverModel,
        poi=(sudoku, empty_entry, same_row, same_col, same_table),
        inferTypes=['local/argmax'],
        # inferTypes=['ILP', 'local/argmax'],
#         metric={
#             'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},

        #metric={ 'softmax' : ValueTracker(prediction_softmax),
        #       'ILP': PRF1Tracker(DatanodeCMMetric()),
        #        'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))
        #       },
#         loss=MacroAverageTracker(NBCrossEntropyLoss()),
        
        sample = True,
        sampleSize=300, 
        sampleGlobalLoss = True
        )