from sympy import diff


def get_gradients_of_function(function, free_symbols):
    gradients = []
    for symbol in free_symbols:
        gradients.append(diff(function, symbol))

    return gradients


def calculate_norm(values):
    return sum([value ** 2 for value in values]) ** 0.5


def make_value_mapping(values, symbols):
    return {symbols[index]: element for index, element in enumerate(values)}


def substitute_values_in_variables(functions, x_i):
    result = []
    for func in functions:
        for symbol in func.free_symbols:
            try:
                result.append(func.subs(symbol, x_i[symbol]))
            except KeyError:
                continue

    return result