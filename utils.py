from sympy import diff


def get_gradients_of_function(function, free_symbols):
    gradients = []
    for symbol in free_symbols:
        gradients.append(diff(function, symbol))

    return gradients


def substitute_values_in_variables(functions, x_i):
    result = []
    for func in functions:
        for symbol in func.free_symbols:
            try:
                result.append(func.subs(symbol, x_i[symbol]))
            except KeyError:
                continue

    return result