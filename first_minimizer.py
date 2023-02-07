import time

import numpy as np
import scipy.optimize as optimize
from sympy import symbols

from base_minimizer import BaseMinimizer
from utils import substitute_values_in_variables, make_value_mapping


class FirstMinimizer(BaseMinimizer):
    def _minimize_function_of_two_variables(self, function_for_minimizing, x_current):
        x_current = np.array([value for value in x_current.values()], dtype=np.float)

        def function_np(params):
            beta_float, gamma_float = params
            beta, gamma = sorted(function_for_minimizing.free_symbols, key=lambda x: x.name)
            return function_for_minimizing.subs([(beta, beta_float), (gamma, gamma_float)])

        return optimize.minimize(function_np, np.zeros(2), method='Nelder-Mead')

    def minimize(self):
        start_time = time.time()
        iterations_results = []
        i = 0
        is_should_stop = False
        is_first_iteration = True

        x_current = self.x_0
        x_previous = None

        while not is_should_stop:
            if is_first_iteration:
                x_next = self.get_first_element(x_current)
                is_first_iteration = False

                iterations_results.append(x_next)
                x_previous, x_current = x_current, x_next
                x_previous = make_value_mapping(x_previous, self.free_symbols)
            else:
                gamma = symbols('gamma')
                beta = symbols('beta')

                x_current = make_value_mapping(x_current, self.free_symbols)

                functions = [x_current[symbol] - gamma * self.gradients[index] + beta * (x_current[symbol] - x_previous[symbol])
                             for index, symbol in enumerate(x_current.keys())]

                functions_without_variables = substitute_values_in_variables(functions, x_current)
                function_for_minimizing = self.function.subs([(symbol, functions_without_variables[index])
                                                              for index, symbol in enumerate(self.free_symbols)])

                optimized_result = self._minimize_function_of_two_variables(function_for_minimizing, x_current)

                params_values = optimized_result.x
                beta, gamma = sorted(function_for_minimizing.free_symbols, key=lambda sym: sym.name)
                new_params = make_value_mapping(params_values, [beta, gamma])
                x_next = [function.subs([(symbol, new_params[symbol]) for symbol in [beta, gamma]]) for function in functions_without_variables]

                is_should_stop = self.stopping_executor.stopping_criterion(x_current, x_next)

                x_previous = x_current
                x_current = x_next
                i += 1

                iterations_results.append(x_next)

        print("Time of execution of first method %s" % (time.time() - start_time))
        return x_next, iterations_results





