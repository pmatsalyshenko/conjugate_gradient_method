from typing import List

import sympy
from numpy.lib.type_check import imag
from scipy.optimize import fsolve
from sympy import sympify, solve, diff, symbols, solveset, real_roots

from stopping_executor import StoppingExecutor
from utils import get_gradients_of_function, substitute_values_in_variables, make_value_mapping


class BaseMinimizer:
    def __init__(self, x_0: List, function: str, error_signs: int):
        self.x_0 = x_0
        self.dimension = len(x_0)
        self.function = sympify(function)
        self.free_symbols = sorted(self.function.free_symbols, key=lambda sym: sym.name)
        self.gradients = get_gradients_of_function(self.function, self.free_symbols)
        self.stopping_executor = StoppingExecutor(self.function, error_signs)

    def _minimize_scalar_function(self, function):
        new_der = diff(function)
        second_der = diff(new_der)
        try:
            results = iter(real_roots(new_der, next(iter(function.free_symbols))))
        except StopIteration:
            raise ValueError('Haven`t found a real roots')

        results = [result.evalf() for result in results]
        second_ders = [second_der.subs([(symbol, result) for index, symbol in enumerate(second_der.free_symbols)]) for result in results]
        results = [results[index] for index, der in enumerate(second_ders) if der > 0]

        function_values = [function.subs([(symbol, result) for index, symbol in enumerate(function.free_symbols)]) for result in results]
        index_of_min = function_values.index(min(function_values))

        return results[index_of_min]

    def get_first_element(self, x_current: List):
        x_current = make_value_mapping(x_current, self.free_symbols)
        gamma_i = symbols('gamma')

        functions = [x_current[x_i_sym] - gamma_i*self.gradients[index] for index, x_i_sym in enumerate(x_current.keys())]

        functions_without_variables = substitute_values_in_variables(functions, x_current)
        scalar_function = self.function.subs([(symbol, functions_without_variables[index])
                                              for index, symbol in enumerate(self.free_symbols)])

        scalar_function = (1-2*gamma_i)**2+25*(1-50*gamma_i)**2
        new_gamma_i = self._minimize_scalar_function(scalar_function)
        x_next = [function.subs(gamma_i, new_gamma_i) for function in functions_without_variables]

        return x_next