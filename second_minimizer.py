import time
from copy import deepcopy

from sympy import symbols

from base_minimizer import BaseMinimizer
from utils import substitute_values_in_variables, calculate_norm, make_value_mapping


class SecondMinimizer(BaseMinimizer):
    def minimize(self):
        start_time = time.time()
        iterations_results = []
        i = 0

        is_should_stop = False
        is_first_iteration = True

        x_current = self.x_0
        x_previous = None

        x_current_mapping = make_value_mapping(x_current, self.free_symbols)
        p_previous = [-value for value in
                      substitute_values_in_variables(self.gradients, x_current_mapping)
                      ]
        r_previous = substitute_values_in_variables(self.gradients, x_current_mapping)

        while not is_should_stop:
            if is_first_iteration:
                x_next = self.get_first_element(x_current)
                is_first_iteration = False

                iterations_results.append(x_next)
                i += 1
                x_previous, x_current = x_current, x_next
                x_previous = make_value_mapping(x_previous, self.free_symbols)
            else:
                x_current_values = deepcopy(x_current)
                x_current = make_value_mapping(x_current, self.free_symbols)

                r_current = substitute_values_in_variables(self.gradients, x_current)

                if i // self.dimension == 0:
                    beta_current = 0
                else:
                    r_current_norm = calculate_norm(r_current)
                    r_previous_norm = calculate_norm(r_previous)
                    beta_current = r_current_norm / r_previous_norm

                p_current = [-r_current[i] + beta_current * p_previous[i] for i in range(self.dimension)]

                gamma = symbols('gamma')

                argument = [x_current_values[i] + gamma * p_current[i] for i in range(self.dimension)]

                function_for_minimizing = self.function.subs([
                    (self.free_symbols[i], argument[i])
                    for i in range(self.dimension)
                ])

                gamma_current = self._minimize_scalar_function(function_for_minimizing)

                x_next = [x_current_values[i] + gamma_current * p_current[i] for i in range(self.dimension)]

                is_should_stop = self.stopping_executor.stopping_criterion(x_current, x_next)

                x_previous, x_current = x_current, x_next
                p_previous = p_current
                r_previous = r_current
                i += 1

                iterations_results.append(x_next)
        print("Time of execution of second method %s" % (time.time() - start_time))
        return x_next, iterations_results