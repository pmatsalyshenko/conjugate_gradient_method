import math

import matplotlib.pyplot as plt
from sympy import sympify

from first_minimizer import FirstMinimizer
from second_minimizer import SecondMinimizer
from utils import make_value_mapping, substitute_values_in_variables


class MainExecutor:
    def __init__(self, input_data):
        self._form_arguments(input_data)

    def _form_arguments(self, input_data):
        expected_values = []
        arguments_for_minimizers = []
        for i in input_data:
            expected_values.append(i.pop())
            arguments_for_minimizers.append(i)

        self._expected_values = expected_values
        self._arguments_for_minimizers = arguments_for_minimizers

    def _prepare_data_to_plot(self, results, minimizer, ev):
        values_mappings = [
            make_value_mapping(result, minimizer.free_symbols)
            for result
            in results
        ]

        values = [
            minimizer.function.subs(
                [(symbol, mapping[symbol]) for symbol in minimizer.free_symbols]
            ) for mapping
            in values_mappings
        ]

        returned_values = []
        for value in values:
            try:
                returned_values.append(math.log(abs(value - ev), 10))
            except ValueError:
                break

        return returned_values

    def _build_plot(self, first_iterations_results, second_iterations_results):
        x1 = list(iter(range(len(first_iterations_results))))

        y1 = first_iterations_results
        # plotting the line 1 points
        plt.plot(x1, y1, label="First method")

        # line 2 points
        x2 = list(iter(range(len(second_iterations_results))))
        y2 = second_iterations_results
        # plotting the line 2 points
        plt.plot(x2, y2, label="Second method", linestyle='--')

        # naming the x axis
        plt.xlabel('Amount of iterations')
        # naming the y axis
        plt.ylabel('log |f(x) - f(x*)|')
        # giving a title to my graph
        plt.title('Comparing two methods')

        # show a legend on the plot
        plt.legend()

        # function to show the plot
        plt.show()

    def execute(self):
        for args, ev in zip(self._arguments_for_minimizers, self._expected_values):
            first_minimizer = FirstMinimizer(args[0], args[1], args[2])
            second_minimizer = SecondMinimizer(args[0], args[1], args[2])

            first_result, first_iterations_results = first_minimizer.minimize()
            second_result, second_iterations_results = second_minimizer.minimize()
            print(f'x0:{args[2]}')
            first_zero_res_sub = sympify(args[1]).subs(
                [(symbol, args[2]) for i, symbol in enumerate(sympify(args[1]).free_symbols)])
            print(f'f(x0):{first_zero_res_sub}')

            first_res_sub = sympify(args[1]).subs([(symbol, first_result[i]) for i, symbol in enumerate(sympify(args[1]).free_symbols)])
            print([(symbol, first_result[i]) for i, symbol in enumerate(sympify(args[1]).free_symbols)])
            second_res_sub = sympify(args[1]).subs([(symbol, second_result[i]) for i, symbol in enumerate(sympify(args[1]).free_symbols)])
            print(f'Value of first: {first_res_sub}')
            print(f'Value of second: {second_res_sub}')
            print(f'First result: {first_result}')
            print(f'Second result: {second_result}')
            print(f"First alg iterations:{len(first_iterations_results)}")
            print(f"Second alg iterations:{len(second_iterations_results)}")


            ev = first_minimizer.function.subs(
                [(symbol, ev[i]) for i, symbol in enumerate(first_minimizer.free_symbols)]
            )

            first_function_values = self._prepare_data_to_plot(first_iterations_results, first_minimizer, ev)
            second_function_values = self._prepare_data_to_plot(second_iterations_results, second_minimizer, ev)
            self._build_plot(first_function_values, second_function_values)





