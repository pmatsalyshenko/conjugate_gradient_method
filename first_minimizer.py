from typing import Callable, List, Dict

import numpy
import numpy as np
import scipy.optimize as optimize
from sympy import symbols, diff, sympify, solve
from scipy.optimize import rosen

from stopping_executor import StoppingExecutor
from utils import get_gradients_of_function, substitute_values_in_variables


class FirstMinimizer:
    #gradient - scipy.optimize.approx_fprime
    def __init__(self, x_0: List, function: str, error_signs: int):
        self.x_0 = x_0
        self.dimension = len(x_0)
        self.function = sympify(function)
        self.free_symbols = sorted(self.function.free_symbols, key=lambda sym: sym.name)
        self.gradients = get_gradients_of_function(self.function, self.free_symbols)
        self.stopping_executor = StoppingExecutor(self.function, error_signs)

    def _form_function_for_minimize(self):
        pass

    def _make_value_mapping(self, value):
        return {self.free_symbols[i]: value[i] for i in range(self.dimension)}

    def _minimize_scalar_function(self, function):
        new_der = diff(function)
        new_gamma_i = float(solve(new_der)[0])

        return new_gamma_i

    def _minimize_function_of_two_variables(self, function_for_minimizing, x_current):
        x_current = np.asarray([value for value in x_current.values()])

        def function_np(params):
            beta_float, gamma_float = params
            gamma, beta = sorted(function_for_minimizing.free_symbols, key=lambda x: x.name)
            return function_for_minimizing.subs([(beta, beta_float), (gamma, gamma_float)])

        return optimize.minimize(function_np, x_current)

    def _minimize_by_steepest_descent(self, x_current: List):
        # should_stop = False
        # while not should_stop:
        x_current = self._make_value_mapping(x_current)
        gamma_i = symbols('gamma')

        functions = [x_current[x_i_sym] - gamma_i*self.gradients[index] for index, x_i_sym in enumerate(x_current.keys())]

        functions_without_variables = substitute_values_in_variables(functions, x_current)

        scalar_function = self.function.subs([(symbol, functions_without_variables[index])
                                              for index, symbol in enumerate(self.free_symbols)])

        new_gamma_i = self._minimize_scalar_function(scalar_function)
        x_next = [function.subs(gamma_i, new_gamma_i) for function in functions_without_variables]

        should_stop = self.stopping_executor._first_stopping_criterion(x_current, x_next)
        x_current = x_next

        return x_next

    def _minimize_by_conjugate_gradient_method(self):
        is_should_stop = False
        is_first_iteration = True

        x_current = self.x_0
        x_previous = None

        while not is_should_stop:
            if is_first_iteration:
                x_next = self._minimize_by_steepest_descent(x_current)
                is_first_iteration = False

                x_previous, x_current = x_current, x_next
            else:
                gamma = symbols('gamma')
                beta = symbols('beta')

                x_current = self._make_value_mapping(x_current)
                x_previous = self._make_value_mapping(x_previous)
                print(x_current, x_previous)

                functions = [x_current[symbol] - gamma * self.gradients[index] + beta * (x_current[symbol] - x_previous[symbol])
                             for index, symbol in enumerate(x_current.keys())]

                functions_without_variables = substitute_values_in_variables(functions, x_current)
                function_for_minimizing = self.function.subs([(symbol, functions_without_variables[index])
                                                              for index, symbol in enumerate(self.free_symbols)])
                print(function_for_minimizing)
                new_params = self._minimize_function_of_two_variables(function_for_minimizing, x_current)




