from typing import List

from sympy import sympify, solve, diff, symbols

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
        new_gamma_i = float(solve(new_der)[0])

        return new_gamma_i

    def get_first_element(self, x_current: List):
        is_should_stop = False

        while not is_should_stop:
            x_current = make_value_mapping(x_current, self.free_symbols)
            gamma_i = symbols('gamma')

            functions = [x_current[x_i_sym] - gamma_i*self.gradients[index] for index, x_i_sym in enumerate(x_current.keys())]

            functions_without_variables = substitute_values_in_variables(functions, x_current)
            scalar_function = self.function.subs([(symbol, functions_without_variables[index])
                                                  for index, symbol in enumerate(self.free_symbols)])

            new_gamma_i = self._minimize_scalar_function(scalar_function)
            x_next = [function.subs(gamma_i, new_gamma_i) for function in functions_without_variables]

            is_should_stop = self.stopping_executor._first_stopping_criterion(x_current, x_next)
            x_current = x_next

        return x_next