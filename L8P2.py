import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

#Primeiramente, vamos definir os behaviours envolvidos

#Caixa PR
def p_PR(a,b,x,y):
    if  x*y == ((a+b)%2):
        return 1/2
    else:
        return 0

#Local
def p_L(a,b,x,y):
    if (a==0) and (b==0):
        return 1
    else:
        return 0

#Isotrópico
def p_I(a,b,x,y):
    return 1/4

#Combinação convexa do enunciado 
def p(a,b,x,y,alpha,beta):
    return alpha*p_PR(a,b,x,y) + (1-alpha)*(beta*p_L(a,b,x,y) + (1-beta)*p_I(a,b,x,y))

#As distribuições marginais são:
def p_A(a,x,alpha,beta):
    marginal = 0
    for b in [0, 1]:
        marginal += p(a,b,x,0,alpha,beta)
    return marginal

def p_B(b,y,alpha,beta):
    marginal = 0
    for a in [0, 1]:
        marginal += p(a,b,0,y,alpha,beta)
    return marginal

#Array contendo 50 valores de beta no intervalo [0,1]
valores_beta = np.linspace(0, 1, 50)

#Montando os elementos da matriz gamma:
def matrix_element(matrix, i, j):

    vector_1 = np.eye(1, 5, i)[0]
    vector_2 = np.eye(1, 5, j).T

    return (vector_1@(matrix @ vector_2))[0]

#funçao que maximiza alpha para cada beta
def alpha_maximo(beta):
    alpha = cp.Variable(nonneg=True)
    gamma = cp.Variable(shape=(5, 5), hermitian=True)
    pA_00,pA_01 = p_A(0,0,alpha,beta),p_A(0,1,alpha,beta)
    pB_00,pB_01 = p_B(0,0,alpha,beta),p_B(0,1,alpha,beta)

    constraints = [alpha <= 1, gamma >> 0, matrix_element(gamma, 0, 0) == 1, matrix_element(gamma, 0, 1) == pA_00,
        matrix_element(gamma, 0, 2) == pA_01, matrix_element(gamma, 0, 3) == pB_00, matrix_element(gamma, 0, 4) == pB_01,
        matrix_element(gamma, 1, 1) == pA_00, matrix_element(gamma, 1, 3) == p(0,0,0,0,alpha,beta),
        matrix_element(gamma, 1, 4) == p(0,0,0,1,alpha,beta), matrix_element(gamma, 2, 2) == pA_01,
        matrix_element(gamma, 2, 3) == p(0,0,1,0,alpha,beta), matrix_element(gamma, 2, 4) == p(0,0,1,1,alpha,beta),
        matrix_element(gamma, 3, 3) == pB_00, matrix_element(gamma, 4, 4) == pB_01]  
    problem = cp.Problem(cp.Maximize(alpha), constraints)
    return problem.solve() 

#Lista com o valor maximo de alpha obito para cada beta
valores_alpha = []
for beta in valores_beta:
    valores_alpha.append(alpha_maximo(beta))

plt.plot(valores_beta, valores_alpha, label = "Q1")
plt.xlabel("Beta")
plt.ylabel("Alpha máximo")
plt.legend()
plt.grid()
plt.show()