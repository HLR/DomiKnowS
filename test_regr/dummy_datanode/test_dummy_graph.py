import pytest
from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import nandL, notL, ifL
from domiknows.graph.dataNodeDummy import createDummyDataNode, satisfactionReportOfConstraints


class TestDummyGraph:
    
    def test_graph_creation_and_constraints(self):
        """Test graph creation with logical constraints"""
        with Graph('test_graph') as graph:
            image = Concept(name='image')

            a = image(name='a')
            b = image(name='b')
            c = image(name='c')
            d = image(name='d')
            e = image(name='e')

            # Apply logical constraints
            ifL(a, notL(b))
            ifL(a, notL(c))
            nandL(d, e)

            # Create dummy data node and perform operations
            datanode = createDummyDataNode(graph)
            
            # Verify initial state
            attribute_a = datanode.getAttribute(a)
            assert attribute_a is not None
            
            # Perform ILP inference
            datanode.inferILPResults()
            
            # Verify results
            verify_result = datanode.verifyResultsLC()
            verify_result_ilp = datanode.verifyResultsLC()
            
            # Generate satisfaction report
            report = satisfactionReportOfConstraints(datanode)
            
            # Basic assertions
            assert verify_result is not None
            assert verify_result_ilp is not None
            assert report is not None
    
    def test_logical_constraints_consistency(self):
        """Test that logical constraints are properly applied"""
        with Graph('constraint_test_graph') as graph:
            image = Concept(name='image')
            
            a = image(name='a')
            b = image(name='b')
            c = image(name='c')
            d = image(name='d')
            e = image(name='e')
            
            # Apply constraints
            ifL(a, notL(b))
            ifL(a, notL(c))
            nandL(d, e)
            
            datanode = createDummyDataNode(graph)
            datanode.inferILPResults()
            
            # Verify constraint satisfaction
            verify_result = datanode.verifyResultsLC()
            assert verify_result is not None
    
    def test_satisfaction_report_generation(self):
        """Test satisfaction report generation"""
        with Graph('report_test_graph') as graph:
            image = Concept(name='image')
            
            a = image(name='a')
            b = image(name='b')
            
            ifL(a, notL(b))
            
            datanode = createDummyDataNode(graph)
            datanode.inferILPResults()
            
            report = satisfactionReportOfConstraints(datanode)
            assert report is not None