from utils import get_gradients_of_function


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