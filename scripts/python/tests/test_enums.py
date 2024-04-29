import unittest

from PyAutoscoper.connect import CostFunction, OptimizationInitializationHeuristic, OptimizationMethod


class TestEnums(unittest.TestCase):
    def test_cost_function(self) -> None:
        assert CostFunction.NORMALIZED_CROSS_CORRELATION.value == 0
        assert CostFunction.SUM_OF_ABSOLUTE_DIFFERENCES.value == 1
        assert CostFunction(0).name == "NORMALIZED_CROSS_CORRELATION"
        assert CostFunction(1).name == "SUM_OF_ABSOLUTE_DIFFERENCES"
        self.assertRaises(ValueError, CostFunction, 2)
        self.assertRaises(ValueError, CostFunction, "0")

    def test_optimization_initialization_heuristic(self) -> None:
        assert OptimizationInitializationHeuristic.CURRENT_FRAME.value == 0
        assert OptimizationInitializationHeuristic.PREVIOUS_FRAME.value == 1
        assert OptimizationInitializationHeuristic.LINEAR_EXTRAPOLATION.value == 2
        assert OptimizationInitializationHeuristic.SPLINE_INTERPOLATION.value == 3
        assert OptimizationInitializationHeuristic(0).name == "CURRENT_FRAME"
        assert OptimizationInitializationHeuristic(1).name == "PREVIOUS_FRAME"
        assert OptimizationInitializationHeuristic(2).name == "LINEAR_EXTRAPOLATION"
        assert OptimizationInitializationHeuristic(3).name == "SPLINE_INTERPOLATION"
        self.assertRaises(ValueError, OptimizationInitializationHeuristic, 4)
        self.assertRaises(ValueError, OptimizationInitializationHeuristic, "0")

    def test_optimization_method(self) -> None:
        assert OptimizationMethod.PARTICLE_SWARM_OPTIMIZATION.value == 0
        assert OptimizationMethod.DOWNHILL_SIMPLEX.value == 1
        assert OptimizationMethod(0).name == "PARTICLE_SWARM_OPTIMIZATION"
        assert OptimizationMethod(1).name == "DOWNHILL_SIMPLEX"
        self.assertRaises(ValueError, OptimizationMethod, 2)
        self.assertRaises(ValueError, OptimizationMethod, "0")
