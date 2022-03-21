from utils import get_gradients_of_function, calculate_norm, substitute_values_in_variables, make_value_mapping


class StoppingExecutor:
    def __init__(self, function, error_signs):
        self.function = function
        self.error = 10**(-error_signs)
        self.free_symbols = sorted(self.function.free_symbols, key=lambda sym: sym.name)
        self.gradients = get_gradients_of_function(self.function, self.free_symbols)

    def _first_stopping_criterion(self, x_current, x_next):
        current_value = self.function.subs([(symbol, x_current[symbol]) for symbol in self.free_symbols])
        next_value = self.function.subs([(symbol, x_next[index]) for index, symbol in enumerate(self.free_symbols)])

        return current_value - next_value < self.error * (1 + abs(next_value))

    def _second_stopping_criterion(self, x_current, x_next):
        difference = [current_value - next_value for current_value, next_value in zip(x_current.values(), x_next)]
        norm_of_difference = calculate_norm(difference)
        norm_of_next_value = calculate_norm(x_next)

        return norm_of_difference < self.error ** (1/2) * (1 + norm_of_next_value)

    def _third_stopping_criterion(self, x_next):
        x_next_mapping = make_value_mapping(x_next, self.free_symbols)
        values_of_gradients = substitute_values_in_variables(self.gradients, x_next_mapping)
        norm_of_values_of_gradients = calculate_norm(values_of_gradients)
        values_of_function = self.function.subs([(symbol, x_next_mapping[symbol]) for symbol in self.free_symbols])

        return norm_of_values_of_gradients <= self.error ** (1/3) * (1 + abs(values_of_function))

    def stopping_criterion(self, x_current, x_next):
        return all([
            self._first_stopping_criterion(x_current, x_next),
            self._second_stopping_criterion(x_current, x_next),
            self._third_stopping_criterion(x_next)
        ])