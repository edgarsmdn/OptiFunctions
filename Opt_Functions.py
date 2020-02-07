import numpy as np

'''
-------------------------------------------------------------------------------
---------------- TEST FUNCTIONS FOR OPTIMIZATION PROBLEMS  --------------------
----------------------- Coded by Edgar Sanchez -------------------------------
-------------------------------------------------------------------------------
'''

###############################################################################
# --------------------------- PLANE-SHAPED ------------------------------------
###############################################################################

def s_sum(variables):
    '''
    Simple sum function
    Minimum at (dimension * (left limit)) at x = [left limits]
    Usually domain of evaluation is [0, 100]
    Source: Edgar Sanchez
    Retrieved: 18/06/2018
    '''
    return np.sum(variables)    

###############################################################################
# --------------------------- BOWL-SHAPED -------------------------------------
###############################################################################

def bohachevsky(variables):
    '''
    Bohachevsky function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-15, 15]
    Source: http://infinity77.net/global_optimization/test_functions_nd_B.html
    Retrieved: 18/06/2018
    '''
    dimension = len(variables)
    a = np.power(variables[:dimension-1], 2)
    b = 2 * np.power(variables[1:dimension], 2)
    c = -0.3 * np.cos(np.multiply(3*np.pi, variables[:dimension-1]))
    d = -0.4 * np.cos(np.multiply(4*np.pi, variables[1:dimension]))
    f_sum = a + b + c + d + 0.7
    return np.sum(f_sum)

def brown(variables):
    '''
    Brown function
    Minimum at 0 at x = [1, ..., n]
    Usually domain of evaluation is [-1, 1]
    Source: http://infinity77.net/global_optimization/test_functions_nd_B.html#n-d-test-functions-b
    Retrieved: 19/06/2018
    
    Notes:
        1. Evaluation domain in website is incorrect. The correct is [-1, 1]
    '''
    dimension = len(variables)
    a = np.power(variables[:dimension-1], 2)
    b = np.power(variables[1:dimension], 2) + 1
    c = np.power(variables[1:dimension], 2)
    d = np.power(variables[:dimension-1], 2) + 1
    return np.sum(np.power(a,b) + np.power(c,d))

def difpow_sum(variables):
    '''
    Sum of different squares function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-1,1]
    Source: http://www.sfu.ca/~ssurjano/sumpow.html
    Retrieved: 18/06/2018
    '''
    dimension = len(variables)
    return np.sum(np.power(np.abs(variables),range(2,dimension+2)))

def exponential(variables):
    '''
    Exponential function
    Minimum at -1 at x = [zeros]
    Usually domain of evaluation is [-1, 1]
    Source: http://infinity77.net/global_optimization/test_functions_nd_E.html#n-d-test-functions-e
    Retrieved: 20/06/2018
    '''
    return -np.exp(-0.5 * np.sum(np.power(variables, 2)))

def hypEll(variables):
    '''
    Rotated hyper-ellipsoid function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-65.536, 65.536]
    Source: http://www.sfu.ca/~ssurjano/rothyp.html
    Retrieved: 18/06/2018
    '''
    dimension = len(variables)
    return np.sum([sphere_sum(variables[0:i]) for i in range(1, dimension+1)])

def perm(variables, beta=10):
    '''
    Perm function
    Minimum at 0 at x = [1, 1/2, ..., 1/dim]
    Usually domain of evaluation is [-dim, dim]
    Source: http://www.sfu.ca/~ssurjano/perm0db.html
    Retrieved: 18/06/2018
    
    Notes:
        1. beta: Optional constant. If not given, default value is 10
    '''
    dimension = len(variables)
    s_sec = np.zeros(dimension) # To store results
    a = np.array(range(1,dimension + 1)) + beta
    for i in range(1, dimension + 1):
        b = np.power(variables, i)
        c = np.power(1/np.array(range(1, dimension +1)), i)
        f_sum = np.sum(np.multiply(a, (b-c)))
        s_sec[i-1] = np.power(f_sum, 2)
    return np.sum(s_sec)

def sargan(variables):
    '''
    Sargan function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-100, 100]
    Source: http://infinity77.net/global_optimization/test_functions_nd_S.html#n-d-test-functions-s
    Retrieved: 20/06/2018
    '''
    dimension = len(variables)
    result = np.zeros(dimension) # To store results
    for i in range(1, dimension + 1):
        a = dimension * (np.power(variables[i-1], 2) + 0.4*np.sum(np.delete(variables, i-1)))
        result[i-1] = a
    return np.sum(result)

def schwefel_1(variables, alpha=np.sqrt(np.pi)):
    '''
    Schwefel 1 function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-100, 100]
    Source: http://infinity77.net/global_optimization/test_functions_nd_S.html#n-d-test-functions-s
    Retrieved: 21/06/2018
    
    Notes:
        1. Set to np.sqrt(np.pi) if not given
    '''
    return np.power(np.sum(np.power(variables, 2)), alpha)

def sphere_sum(variables):
    '''
    Sphere function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-5.12,5.12]
    Source: http://www.sfu.ca/~ssurjano/spheref.html
    Retrieved: 18/06/2018
    '''
    return np.sum(np.power(variables,2))

def trid(variables):
    '''
    Trid function
    Minimum at (-dim*(dim+4)*(dim-1)/6) at x = [i*(dim+1-i)] for i = 1,2,...,dim
    Usually domain of evaluation is [-dim**2,dim**2]
    Source: http://www.sfu.ca/~ssurjano/trid.html
    Retrieved: 18/06/2018
    '''
    dimension = len(variables)
    a=np.sum(np.power(variables-1,2))
    b=np.sum(np.multiply(variables[1:],variables[:dimension-1]))
    return a - b

###############################################################################
# ------------------------- MANY LOCAL MINIMA ---------------------------------
###############################################################################

def ackley(variables, a=20, b=0.2, c=2*np.pi):
    '''
    Ackley function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-32.768, 32.768]
    Source: http://www.sfu.ca/~ssurjano/ackley.html
    Retrieved: 18/06/2018
    
    Notes:
        1. a, b and c are constants set as recommended in the website
    '''
    dimension = len(variables)
    r = -a*np.exp(-b*np.sqrt((1/dimension)*np.sum(np.power(variables,2))))
    s = -np.exp((1/dimension)*np.sum(np.cos(c*variables)))
    return r + s + a + np.exp(1.0)

def alpine1(variables):
    '''
    Alpine 1 function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-10, 10]
    Source: http://infinity77.net/global_optimization/test_functions_nd_A.html#n-d-test-functions-a
    Retrieved: 19/06/2018
    '''
    return np.sum(np.abs(np.multiply(variables, np.sin(variables)) + 0.1 * variables))

def alpine2(variables):
    '''
    Alpine 2 function
    Minimum at -6.1295 at x = [7.917, ..., 7.917]
    Usually domain of evaluation is [0, 10]
    Source: http://infinity77.net/global_optimization/test_functions_nd_A.html#n-d-test-functions-a
    Retrieved: 19/06/2018
    '''
    return np.prod(np.multiply(np.sqrt(variables), np.sin(variables)))

def deb_1(variables):
    '''
    Deb 1 function
    Minimum at -1, it has 5**dimension global minima
    Usually domain of evaluation is [-1, 1]
    Source: http://infinity77.net/global_optimization/test_functions_nd_D.html#n-d-test-functions-d
    Retrieved: 20/06/2018
    
    Notes:
        1. Minimum of the website is wrong. Correct is -1
    '''
    dimension = len(variables)
    return - np.sum(np.power(np.sin(5*np.pi*variables), 6)) / dimension

def griewank(variables):
    '''
    Griewank function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-600, 600]
    Source: http://www.sfu.ca/~ssurjano/griewank.html
    Retrieved: 18/06/2018
    '''
    dimension = len(variables)
    a = np.sum(np.power(variables,2))/4000.0
    b = np.prod(np.cos(np.divide(variables, np.sqrt(range(1, dimension + 1)))))
    return  a - b + 1

def levy(variables):
    '''
    Levy function
    Minimum at 0 at x = [ones]
    Usually domain of evaluation is [-10, 10]
    Source: http://www.sfu.ca/~ssurjano/levy.html
    Retrieved: 18/06/2018
    '''
    dimension = len(variables)
    varW = 1 + (variables-1)/4.0
    a = np.power(np.sin(np.pi * varW[0]),2)
    b1 = np.power(varW[:dimension-1] - 1,2)
    b2 = 1 + 10 * np.power(np.sin(np.pi * varW[:dimension-1] + 1),2)
    b = np.sum(np.multiply(b1,b2))
    c1 = np.power(varW[dimension-1] - 1,2)
    c2 = 1 + np.power(np.sin(2 * np.pi * varW[dimension-1]),2)
    c = np.multiply(c1,c2)
    return a + b + c

def mishra_11(variables):
    '''
    Mishra 11 function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-10, 10]
    Source: http://infinity77.net/global_optimization/test_functions_nd_M.html#n-d-test-functions-m
    Retrieved: 20/06/2018
    '''
    dimension = len(variables)
    a = np.sum(np.abs(variables)) / dimension
    b = np.prod(np.abs(variables))
    return np.power((a - np.power(b, 1/dimension)), 2)

def multimodal(variables):
    '''
    Multimodal function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-10, 10]
    Source: http://infinity77.net/global_optimization/test_functions_nd_M.html#n-d-test-functions-m
    Retrieved: 20/06/2018
    '''
    return np.multiply(np.sum(np.abs(variables)), np.prod(np.abs(variables)))

def penalty_1(variables, a=10, k=100, m=4):
    '''
    Penalty 1 function
    Minimum at 0 at x = [- ones]
    Usually domain of evaluation is [-50, 50]
    Source: http://infinity77.net/global_optimization/test_functions_nd_P.html#n-d-test-functions-p
    Retrieved: 20/06/2018
    
    Notes:
        1. a set to 10 if not given
        2. k set to 100 if not given
        m. set to 4 if not given
    '''
    dimension = len(variables)
    yi = 1 + 1/4*(variables + 1)
    b = 10 * np.power(np.sin(np.pi * yi[0]), 2)
    c1 = np.power((yi[:dimension-1] - 1), 2)
    c2 = 1 + 10*np.power(np.sin(np.pi* yi[1:]), 2)
    c = np.sum(np.multiply(c1, c2))
    d = np.power(yi[dimension-1] - 1, 2)
    e1 = np.zeros(dimension) # To store the u values
    i = 0
    for x in variables:
        if x > a:
            u = k*(x - a)**m
        if x >= -a and x <= a:
            u = 0
        if x < -a:
            u = k*(-x - a)**m
        e1 [i] = u
        i += 1
    e = np.sum(e1)
    return np.pi/30 * (b + c + d) + e

def penalty_2(variables, a=5, k=100, m=4):
    '''
    Penalty 2 function
    Minimum at 0 at x = [ones]
    Usually domain of evaluation is [-50, 50]
    Source: http://infinity77.net/global_optimization/test_functions_nd_P.html#n-d-test-functions-p
    Retrieved: 20/06/2018
    
    Notes:
        1. a set to 5 if not given
        2. k set to 100 if not given
        3. m set to 4 if not given
    '''
    dimension = len(variables)
    b = np.power(np.sin(3*np.pi*variables[0]), 2)
    c1 = np.power(variables[:dimension-1] - 1, 2)
    c2 = 1 + np.power(np.sin(3*np.pi * variables[1:]), 2)
    c = np.sum(np.multiply(c1, c2))
    d = np.power(variables[dimension-1] - 1, 2) * (1 + np.power(np.sin(2*np.pi * variables[dimension-1]), 2))
    e1 = np.zeros(dimension) # To store the u values
    i = 0
    for x in variables:
        if x > a:
            u = k*(x - a)**m
        if x >= -a and x <= a:
            u = 0
        if x < -a:
            u = k*(-x - a)**m
        e1 [i] = u
        i += 1
    e = np.sum(e1)
    return 0.1 * (b + c + d) + e

def pinter(variables):
    '''
    Pinter function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-10, 10]
    Source: http://infinity77.net/global_optimization/test_functions_nd_P.html#n-d-test-functions-p
    Retrieved: 20/06/2018
    '''
    dimension = len(variables)
    a = np.sum(np.multiply(range(1, dimension + 1), np.power(variables, 2)))
    x_1 = np.concatenate((variables[dimension-1:], variables[:dimension-1]))
    x1 = np.concatenate((variables[1:], variables[:1]))
    A = np.multiply(x_1, np.sin(variables)) + np.sin(x1)
    B = np.power(x_1, 2) - 2*variables + 3*x1 - np.cos(variables) + 1
    b = 20*np.array(range(1, dimension + 1))*np.power(np.sin(A), 2)
    c = np.multiply(range(1, dimension + 1), np.log10(1 + np.multiply(range(1,dimension + 1), np.power(B,2))))
    return a + np.sum(b) + np.sum(c)

def rana(variables):
    '''
    Rana function
    Minimum at -928.5478 at x = [-500, ..., -500]
    Usually domain of evaluation is [-500.000001, 500.000001]
    Source: http://infinity77.net/global_optimization/test_functions_nd_R.html#n-d-test-functions-r
    Retrieved: 20/06/2018
    '''
    a = np.multiply(np.multiply(variables, np.sin(np.sqrt(np.abs(variables[0] - variables + 1)))), np.cos(np.sqrt(np.abs(variables[0] + variables + 1))))
    b = np.multiply(np.multiply(variables + 1, np.sin(np.sqrt(np.abs(variables[0] + variables + 1)))), np.cos(np.sqrt(np.abs(variables[0] - variables + 1))))
    return np.sum(a + b)

def rastrigin(variables):
    '''
    Rastrigin function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-5.12, 5.12]
    Source: http://www.sfu.ca/~ssurjano/rastr.html
    Retrieved: 18/06/2018
    '''
    dimension = len(variables)
    return 10 * dimension + np.sum(np.power(variables,2) - 10 * np.cos(2*np.pi*variables))

def salomon(variables):
    '''
    Salomon function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-100, 100]
    Source: http://infinity77.net/global_optimization/test_functions_nd_S.html#n-d-test-functions-s
    Retrieved: 20/06/2018
    '''
    a = np.cos(2*np.pi* np.sqrt(np.sum(np.power(variables, 2))))
    b = 0.1 * np.sqrt(np.sum(np.power(variables, 2)))
    return 1 - a + b

def schwefel_26(variables):
    '''
    Schwefel 26 function
    Minimum at 0 at x = [420.9687,..., 420.9687]
    Usually domain of evaluation is [-500, 500]
    Source: http://www.sfu.ca/~ssurjano/schwef.html
    Retrieved: 18/06/2018
    '''
    dimension = len(variables)
    a = np.sum(np.multiply(variables,np.sin(np.sqrt(np.abs(variables)))))
    return 418.9829 * dimension - a

def sine_envelope(variables):
    '''
    Sine envelope function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-100, 100]
    Source: http://infinity77.net/global_optimization/test_functions_nd_S.html#n-d-test-functions-s
    Retrieved: 21/06/2018
    
    Notes:
        1. The website has two errors: 
            1) The "0.5" in the numerator must be outside the sine
            2) It must not have the "minus sign" at the beginning of the function
    '''
    dimension = len(variables)
    a = np.power(np.sin(np.sqrt(np.power(variables[1:], 2) + np.power(variables[:dimension-1], 2))), 2) - 0.5
    b = np.power(0.001* (np.power(variables[1:] , 2) + np.power(variables[:dimension-1], 2)) + 1, 2)
    return np.sum(a/b + 0.5)

def trigonometric_2(variables):
    '''
    Trigonometric 2 function
    Minimum at 1 at x = [0.9, ..., 0.9]
    Usually domain of evaluation is [-500, 500]
    Source: http://infinity77.net/global_optimization/test_functions_nd_T.html#n-d-test-functions-t
    Retrieved: 21/06/2018
    '''
    a = 8 * np.power(np.sin(7*np.power(variables - 0.9, 2)), 2)
    b = 6 * np.power(np.sin(14*np.power(variables - 0.9, 2)), 2)
    c = np.power(variables - 0.9, 2)
    return 1 + np.sum(a + b + c)

def vincent(variables):
    '''
    Vincent function
    Minimum at (-dimension) at x = [7.70628098, ..., 7.70628098]
    Usually domain of evaluation is [0.25, 10]
    Source: http://infinity77.net/global_optimization/test_functions_nd_V.html#n-d-test-functions-v
    Retrieved: 21/06/2018
    '''
    return - np.sum(np.sin(10.0 * np.log(variables)))

def weierstrass(variables, a=0.5, b=3, kmax=20):
    '''
    Weierstrass function
    Minimum at 4 (it seems to change if you change the value of "a") at x = [zeros]
    Usually domain of evaluation is [-0.5, 0.5]
    Source: http://infinity77.net/global_optimization/test_functions_nd_W.html#n-d-test-functions-w
    Retrieved: 21/06/2018
    
    Notes:
        1. a: equal to 0.5 if not given
        2. b: equal to 3 if not given
        3. kmax: equal to 20 if not given
    '''
    dimension = len(variables)
    results = np.zeros(dimension) # To store results
    for i in range(1, dimension + 1):
        first = np.sum(np.power(a, range(kmax+1)) * np.cos(2*np.pi*np.power(b, range(kmax+1)) * (variables[i-1] + 0.5)))
        second = dimension * np.sum(np.power(a, range(kmax+1)) * np.cos(np.pi * np.power(b, range(kmax+1))))
        results[i-1] = first - second
    return np.sum(results)

def whitley(variables):
    '''
    Whitley function
    Minimum at 0 at x = [ones]
    Usually domain of evaluation is [-10.24, 10.24]
    Source: http://infinity77.net/global_optimization/test_functions_nd_W.html#n-d-test-functions-w
    Retrieved: 21/06/2018
    
    Notes:
        1. The function seems to have a lower minimum than 0
    '''
    dimension = len(variables)
    results = np.zeros(dimension) # To store results
    for i in range(1, dimension + 1):
        a = np.power((100 * np.power(np.power(variables[i-1], 2) - variables, 2) + np.power(1 - variables, 2)), 2) / 4000
        b = np.cos(100*np.power(np.power(variables[i-1], 2) - variables, 2) + np.power(1 - variables, 2))
        results[i-1] = np.sum(a - b + 1)
    return np.sum(results)

def xinsheyang_2(variables):
    '''
    Xin She Yang 2 function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-2*np.pi, 2*np.pi]
    Source: http://infinity77.net/global_optimization/test_functions_nd_X.html#n-d-test-functions-x
    Retrieved: 21/06/2018
    '''
    a = np.sum(np.abs(variables))
    b = np.exp(np.sum(np.sin(np.power(variables, 2))))
    return a/b

def xinsheyang_4(variables):
    '''
    Xin She Yang 4 function
    Minimum at -1 at x = [zeros]
    Usually domain of evaluation is [-10, 10]
    Source: http://infinity77.net/global_optimization/test_functions_nd_X.html#n-d-test-functions-x
    Retrieved: 21/06/2018
    '''
    a = np.sum(np.power(np.sin(variables), 2))
    b = np.exp(-np.sum(np.power(variables, 2)))
    c = np.exp(-np.sum(np.power(np.sin(np.sqrt(np.abs(variables))), 2)))
    return (a - b)*c

###############################################################################
# ---------------------------- PLATE-SHAPED -----------------------------------
###############################################################################

def amgm(variables):
    '''
    Arithmetic Mean - Geometric Mean Equality function
    Minimum at 0 at x1 = x2 = ... = xn
    Usually domain of evaluation is [0, 10]
    Source: http://infinity77.net/global_optimization/test_functions_nd_A.html#n-d-test-functions-a
    Retrieved: 19/06/2018
    '''
    dimension = len(variables)
    am = np.sum(variables)/dimension
    gm = np.power(np.prod(variables), 1/dimension)
    return (am - gm)**2

def csendes(variables):
    '''
    Csendes function (also called Infinity function)
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-1, 1]
    Source: http://infinity77.net/global_optimization/test_functions_nd_C.html#n-d-test-functions-c
    Retrieved: 19/06/2018
    '''
    a = np.power(variables, 6)
    b = 2 + np.sin(1/variables)
    return np.sum(np.multiply(a, b))

def katsuura(variables, d=32):
    '''
    Katsuura function
    Minimum at 1 at x = [zeros]
    Usually domain of evaluation is [0, 100]
    Source: http://infinity77.net/global_optimization/test_functions_nd_K.html#n-d-test-functions-k
    Retrieved: 21/06/2018
    
    Notes:
        1. d: set to 32 if not given
    '''
    dimension = len(variables)
    results = np.zeros(dimension) # To store results
    for i in range(dimension):
        a = 1 + (i + 1)* np.sum(np.multiply(np.floor(np.multiply(np.power(2, range(1, d + 1)), variables[i])), np.power(2, -np.arange(1,d+1, dtype=float))))
        results[i] = a
    return np.prod(results)

def mishra_1(variables):
    '''
    Mishra 1 function
    Minimum at 2 at x = [ones]
    Usually domain of evaluation is [0, 1]
    Source: http://infinity77.net/global_optimization/test_functions_nd_M.html#n-d-test-functions-m
    Retrieved: 20/06/2018
    '''
    dimension = len(variables)
    xn = dimension - np.sum(variables[:dimension-1])
    return (1 + xn)**xn

def mishra_2(variables):
    '''
    Mishra 2 function
    Minimum at 2 at x = [ones]
    Usually domain of evaluation is [0, 1]
    Source: http://infinity77.net/global_optimization/test_functions_nd_M.html#n-d-test-functions-m
    Retrieved: 20/06/2018
    '''
    dimension = len(variables)
    xn = dimension - np.sum(variables[:dimension-1] + variables[1:])/2
    return (1 + xn)**xn

def mishra_7(variables):
    '''
    Mishra 7 function
    Minimum at 0 at x = [np.sqrt(dimension), ..., np.sqrt(dimension)]
    Usually domain of evaluation is [-10, 10]
    Source: http://infinity77.net/global_optimization/test_functions_nd_M.html#n-d-test-functions-m
    Retrieved: 20/06/2018
    '''
    dimension = len(variables)
    return np.power((np.prod(variables) - np.math.factorial(dimension)), 2)

def quintic(variables):
    '''
    Quintic function
    Minimum at 0 at x = [- ones]
    Usually domain of evaluation is [-10, 10]
    Source: http://infinity77.net/global_optimization/test_functions_nd_Q.html#n-d-test-functions-q
    Retrieved: 20/06/2018
    '''
    return np.sum(np.abs(np.power(variables, 5) - 3*np.power(variables, 4) + 4*np.power(variables, 3) + 2*np.power(variables, 2) - 10*variables - 4))

def schwefel_4(variables):
    '''
    Schewefel 4 function
    Minimum at 0 at x = [ones]
    Usually domain of evaluation is [0, 10]
    Source: http://infinity77.net/global_optimization/test_functions_nd_S.html#n-d-test-functions-s
    Retrieved: 21/06/2018
    '''
    a = np.power(variables - 1, 2)
    b = np.power(variables[0] - np.power(variables, 2), 2)
    return np.sum(a + b)

def zakharov(variables):
    '''
    Zakharov function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-5, 10]
    Source: https://www.sfu.ca/~ssurjano/zakharov.html
    Retrieved: 19/06/2018
    '''
    dimension = len(variables)
    a = sphere_sum(variables)
    b = np.power(np.sum(0.5 * np.multiply(range(1,dimension+1), variables)),2)
    c = b**2
    return a + b + c

###############################################################################
# ---------------------------- VALLEY-SHAPED ----------------------------------
###############################################################################

def dixon_price(variables):
    '''
    Dixon-price function
    Minimum at 0 at x = [2**(-(2**i - 2)/(2**i))] for i = 1, ..., dimension
    Usually domain of evaluation is [-10, 10]
    Source: https://www.sfu.ca/~ssurjano/dixonpr.html
    Retrieved: 19/06/2018
    '''
    dimension = len(variables)
    a = np.power((variables[0]-1), 2)
    b = np.sum(np.multiply(range(2,dimension+1), np.power(2 * np.power(variables[1:],2) - variables[:dimension-1], 2)))
    return a + b

def rosenbrock(variables):
    '''
    Rosenbrock function
    Minimum at 0 at x = [ones]
    Usually domain of evaluation is [-5, 10]
    Source: https://www.sfu.ca/~ssurjano/rosen.html
    Retrieved: 19/06/2018
    '''
    dimension = len(variables)
    return np.sum(100 * np.power(variables[1:] - np.power(variables[:dimension-1], 2), 2) + np.power(variables[:dimension-1] - 1, 2))

def schwefel_2(variables):
    '''
    Schwefel 2 function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-100, 100]
    Source: http://infinity77.net/global_optimization/test_functions_nd_S.html#n-d-test-functions-s
    Retrieved: 21/06/2018
    '''
    dimension = len(variables)
    result = np.zeros(dimension) #To store results
    for i in range(1, dimension + 1):
        a = np.power(np.sum(variables[:i]), 2)
        result[i-1] = a
    return np.sum(result)

###############################################################################
# ------------------------------- OTHERS --------------------------------------
###############################################################################

def deceptive(variables, beta=2):
    '''
    Deceptive function (in its general form)
    Minimum at -1 at x = alpha
    Usually domain of evaluation is [0, 1]
    Source: http://infinity77.net/global_optimization/test_functions_nd_D.html#n-d-test-functions-d
            Marcin, M., Csezlaw, S., Test functions for optimization needs, 2005
    Retrieved: 20/06/2018
    
    Notes:
        1. alpha: an array of numbers between 0 and 1 (not inclusive) of same dimension as the problem
        2. beta: a fixed non-linearity factor. It's set to 2 if not specified
    '''
    dimension = len(variables)
    g = np.zeros(dimension) # To store results
    #!!! You should modify alpha here
    alpha=np.array([0.5 for i in range(len(variables))])
    for i in range(dimension):
        if variables[i] == alpha[i]:
            g_i = 5.0*variables[i]/alpha[i] - 4.0
        elif variables[i] >= 0 and variables[i] <= 4/5 * alpha[i]:
            g_i = -variables[i]/alpha[i] + 4/5
        elif variables[i] > 4/5 * alpha[i] and variables[i] <= alpha[i]:
            g_i = 5.0*variables[i]/alpha[i] - 4.0
        elif variables[i] > alpha[i] and variables[i] <= (1 + 4*alpha[i])/5:
            g_i = 5.0*(variables[i] - alpha[i])/(alpha[i] - 1.0)
        elif variables[i] > (1 + 4*alpha[i])/5 and variables[i] <= 1:
            g_i = (variables[i] - 1.0)/(1.0 - alpha[i])
        g[i] = g_i
    return -(np.sum(g)/dimension)**beta

def deflected_corrugated_spring(variables, alpha=5, K=5):
    '''
    Deflected corrugated spring function
    Minimum at 0 at x = [alpha, ..., alpha]
    Usually domain of evaluation is [0, 2*alpha]
    Source: http://infinity77.net/global_optimization/test_functions_nd_D.html#n-d-test-functions-d
    Retrieved: 20/06/2018
    
    Notes:
        1. alpha: set to 5 if not given
        2. K: set to 5 if not given
    '''
    a = np.power(variables - alpha, 2)
    b = np.cos(K*np.sqrt(np.sum(a)))
    return 0.1 * np.sum(a-b)

def perm_d(variables, beta=10):
    '''
    Perm d function
    Minimum at 0 at x = [1, 2, ..., dim]
    Usually domain of evaluation is [-dim, dim]
    Source: https://www.sfu.ca/~ssurjano/permdb.html
    Retrieved: 19/06/2018
    
    Notes:
        1. beta: Optional constant. If not given, default value is 10
    '''
    dimension = len(variables)
    s_sec = np.zeros(dimension) # To store results
    for i in range(1, dimension + 1):
        a = np.power(range(1,dimension + 1), i) + beta
        b = np.power(variables / range(1, dimension + 1), i) - 1
        f_sum = np.sum(np.multiply(a, b))
        s_sec[i-1] = np.power(f_sum, 2)
    return np.sum(s_sec)

def plateau(variables):
    '''
    Plateau function
    Minimum at 30 at x = [zeros]
    Usually domain of evaluation is [-5.12, 5.12]
    Source: http://infinity77.net/global_optimization/test_functions_nd_P.html#n-d-test-functions-p
    Retrieved: 20/06/2018
    
    Notes:
        1. It is necessary to specify the absolute value to obtain the proper behaiviour
    '''
    return 30 + np.sum(np.floor(np.abs(variables)))

def powell(variables):
    '''
    Powell function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-4, 5]
    Source: https://www.sfu.ca/~ssurjano/powell.html
    Retrieved: 19/06/2018
    
    Notes:
        1. This function only works for dimension >= 4
    '''
    dimension = len(variables)
    upper_limit = dimension // 4
    result = np.zeros(upper_limit) # To store results
    for index in range(1, upper_limit + 1):
        # Defines the variables for each index in the sum
        p_4i_3 = variables[4*index - 3-1]
        p_4i_2 = variables[4*index - 2-1]
        p_4i_1 = variables[4*index - 1-1]
        p_4i = variables[4*index - 1]
        # Makes the operations
        a = (p_4i_3 + 10*p_4i_2)**2
        b = 5*(p_4i_1 - p_4i)**2
        c = (p_4i_2 - 2*p_4i_1)**4
        d = 10*(p_4i_3 - p_4i)**4
        result[index - 1] = a + b + c + d
    return np.sum(result)

def qing(variables):
    '''
    Qing function
    Minimum at 0 at x = [+-np.sqrt(1), ..., +-np.sqrt(dimension)]
    Usually domain of evaluation is [-500, 500]
    Source: http://infinity77.net/global_optimization/test_functions_nd_Q.html#n-d-test-functions-q
    Retrieved: 20/06/2018
    '''
    dimension = len(variables)
    return np.sum(np.power((np.power(variables, 2) - range(1, dimension + 1)), 2))

def quartic(variables):
    '''
    Quartic function
    Minimum at 0.5 at x = [zeros]
    Usually domain of evaluation is [-1.28, 1.28]
    Source: Momin J., Xin-She Y., A literature survey of benchmark functions for global optimization problems
    Retrieved: 20/06/2018
    '''
    dimension = len(variables)
    a = np.sum(np.multiply(range(1,dimension+1),np.power(variables,4)))
    b = 0.5
    return a + b

def schwefel_20(variables):
    '''
    Schwefel 20 function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-100, 100]
    Source: http://infinity77.net/global_optimization/test_functions_nd_S.html#n-d-test-functions-s
    Retrieved: 21/06/2018
    '''
    return np.sum(np.abs(variables))

def schwefel_21(variables):
    '''
    Schwefel 21 function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-100, 100]
    Source: http://infinity77.net/global_optimization/test_functions_nd_S.html#n-d-test-functions-s
    Retrieved: 21/06/2018
    '''
    return max(np.abs(variables))

def schwefel_22(variables):
    '''
    Schwefel 22 function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-100, 100]
    Source: http://infinity77.net/global_optimization/test_functions_nd_S.html#n-d-test-functions-s
    Retrieved: 21/06/2018
    '''
    return np.sum(np.abs(variables)) + np.prod(np.abs(variables))

def step2(variables):
    '''
    Step 2 function
    Minimum at 0 at x = [0.5, ..., 0.5]
    Usually domain of evaluation is [-100, 100]
    Source: Momin J., Xin-She Y., A literature survey of benchmark functions for global optimization problems
    Retrieved: 20/06/2018
    '''
    return np.sum(np.power(np.floor(variables + 0.5),2))

def stepint(variables):
    '''
    Stepint function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-5.12, 5.12]
    Source: Momin J., Xin-She Y., A literature survey of benchmark functions for global optimization problems
    Retrieved: 21/06/2018
    
    Notes:
        1. It is necessary to specify the absolute value in the function to produce expected behaviour
        2. The minimum of the paper is wrong. Correct minimum is 25
    '''
    return 25.0 + np.sum(np.floor(np.abs(variables)))

def styblinski_tang(variables):
    '''
    Styblinski-Tang function
    Minimum at -78.332 at x = [-2.903534, ..., -2.903534]
    Usually domain of evaluation is [-5, 5]
    Source: Momin J., Xin-She Y., A literature survey of benchmark functions for global optimization problems
    Retrieved: 19/06/2018
    '''
    return 0.5 * np.sum(np.power(variables, 4) - 16 * np.power(variables, 2) + 5 * variables)

def trigonometric_1(variables):
    '''
    Trigonometric 1 function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [0, np.pi]
    Source: http://infinity77.net/global_optimization/test_functions_nd_T.html#n-d-test-functions-t
    Retrieved: 21/06/2018
    '''
    dimension = len(variables)
    results = np.zeros(dimension) # To store results
    for i in range(1, dimension + 1):
        a = dimension - np.sum(np.cos(variables)) + i*(1 - np.cos(variables[i-1]) - np.sin(variables[i-1]))
        results[i-1] = np.power(a, 2)
    return np.sum(results)

def xinsheyang_3(variables, beta=15, m=3):
    '''
    Xin She Yang 3 function
    Minimum at -1 at x = [zeros]
    Usually domain of evaluation is [-20, 20]
    Source: http://infinity77.net/global_optimization/test_functions_nd_X.html#n-d-test-functions-x
    Retrieved: 21/06/2018
    
    Notes:
        1. beta: set as 15 if not given
        2. m: set as 3 if not given
    '''
    a = np.exp(-np.sum(np.power(variables / beta, 2*m)))
    b = 2*np.exp(-np.sum(np.power(variables, 2))) * np.prod(np.power(np.cos(variables), 2))
    return a - b

def yaoliu_4(variables):
    '''
    Yao Liu 4 function
    Minimum at 0 at x = [zeros]
    Usually domain of evaluation is [-10, 10]
    Source: http://infinity77.net/global_optimization/test_functions_nd_Y.html#n-d-test-functions-y
    Retrieved: 21/06/2018
    '''
    return np.abs(max(variables))

def zerosum(variables):
    '''
    Zero sum function
    Minimum at 0 where np.sum(variables) = 0
    Usually domain of evaluation is [-10, 10]
    Source: http://infinity77.net/global_optimization/test_functions_nd_Z.html#n-d-test-functions-z
    Retrieved: 21/06/2018
    '''
    if np.sum(variables) == 0:
        return 0
    else:
        return 1 + np.power((10000 * np.abs(np.sum(variables))), 0.5)
