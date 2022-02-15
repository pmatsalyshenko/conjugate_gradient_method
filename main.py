from first_minimizer import FirstMinimizer

executor = FirstMinimizer([1,1], 'x_1**2 + 25*x_2**2', 6)

executor._minimize_by_conjugate_gradient_method()