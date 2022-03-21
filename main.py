from first_minimizer import FirstMinimizer
from second_minimizer import SecondMinimizer

first_executor = FirstMinimizer([1,1], 'x_1**2 + 25*x_2**2', 6)
second_executor = SecondMinimizer([1,1], 'x_1**2 + 25*x_2**2', 6)

first_result = first_executor.minimize()
second_result = second_executor.minimize()

print([f'{value:.6f}' for value in first_result])
print([f'{value:.6f}' for value in second_result])