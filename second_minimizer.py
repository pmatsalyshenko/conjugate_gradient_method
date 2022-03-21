from copy import deepcopy

from sympy import symbols

from base_minimizer import BaseMinimizer
from utils import substitute_values_in_variables, calculate_norm, make_value_mapping


class SecondMinimizer(BaseMinimizer):
    def minimize(self):
        is_should_stop = False
        is_first_iteration = True

        x_current = self.x_0
        x_previous = None

        p_previous = [-value for value in
                      substitute_values_in_variables(self.gradients,
                                                     make_value_mapping(x_current, self.free_symbols))
                      ]

        while not is_should_stop:
            if is_first_iteration:
                x_next = self.get_first_element(x_current)
                is_first_iteration = False

                x_previous, x_current = x_current, x_next
                x_previous = make_value_mapping(x_previous, self.free_symbols)
            else:
                x_current_values = deepcopy(x_current)
                x_current = make_value_mapping(x_current, self.free_symbols)

                r_current = substitute_values_in_variables(self.gradients, x_current)

                r_current_norm = calculate_norm(x_current.values())
                r_previous_norm = calculate_norm(x_previous.values())
                beta_current = r_current_norm ** 2 / r_previous_norm ** 2

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

        return x_next