
# Funciones para el curso de Ecuaciones Diferenciales Ordinarias

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def f(x):
    return x**2
def df(x):
    return 2*x

def g(x,y):
    return (3/2)*(y**2-7*y+10)

#a,b: extremos del intervalo
#tol es la tolerancia aceptada para acercarse a la solución
def biseccion(xa,xb,tol):
    while (np.abs(xa-xb)>=tol):
        xm = (xa+xb)/2.0
        caso = f(xa)*f(xm)
        #caso 1: resultado menor a 0, acotamos el intervalo tal que [xa,xm]
        if caso<0:
            xb = xm
        #caso 2: resultado mayor a 0, acotamos el intervalo tal que [xm,xb]
        elif caso>0:
            xa = xm
        #caso 3: se ha encontrado la raiz, tal que xa y xb son iguales (caso==0 o dentro de la tolerancia)
        else:
            xa = xm
            xb = xm
    return xm

respuesta_1 = biseccion(1, 2, 0.01)
print("El método de la bisección entrega una raíz en x=" + str(respuesta_1))

respuesta_2 = biseccion(1, 2, 1e-10)
print("El método de la bisección entrega una raíz en x=" + str(respuesta_2))

respuesta_3 = biseccion(1, 2, 1e-20)
print("El método de la bisección entrega una raíz en x=" + str(respuesta_3))


def secante(f, x0, x1, tol):   
    while (np.abs(x0-x1)>=tol):
        x2 = x1 - f(x1) * (x1 - x0)/(f(x1) - f(x0))
        x0 = x1
        x1 = x2
    return x2

respuesta_4 = secante(f, 1, 2, 0.01)

print("El método de la secante entrega una raíz en x=" + str(respuesta_4))


# FUNCIÓN DE METODO RK22
def RK22(g,x0,xn,y0,n):
    X = np.linspace(x0, xn, n+1) # VECTOR X, PTOS POR DONDE VOY PASANDO EN EL EJE X
    Y = np.linspace(x0, xn, n+1) # VECTOR Y, SOLO LO QUIEPO PARA QUE TENGA EL MISMO TAMAÑO QUE X
    Y[0] = y0
    h = (xn - x0)/n    
    for i in range(n):
        # METODO DE LA FORMULA 𝑦𝑖+1=𝑦𝑖+ℎ2[𝑓(𝑥𝑖,𝑦𝑖)+𝑓(𝑥𝑖+1,𝑦𝑖+ℎ𝑓(𝑥𝑖,𝑦𝑖))]
        Y[i+1] = Y[i] + (h/2)*(g(X[i],Y[i]) + g(X[i+1], Y[i]+ h* g(X[i],Y[i]))) 
    # CREO EL DICCIONARIO/una vez definido, no puedo modificarlos, de forma independiente
    ret = dict()  
    ret["x"] = X
    ret["y"] = Y
    
    return ret

respuesta_5 = RK22(g, 1, 2, 0.01, 10)

print("El método de RK22 entrega una raíz en x=" + str(respuesta_5))

# FUNCIÓN DE METODO RK44
def RK44(g,x0,xn,y0,n):
    X = np.linspace(x0, xn, n+1) # VECTOR X, PTOS POR DONDE VOY PASANDO EN EL EJE X
    Y = np.linspace(x0, xn, n+1) # VECTOR Y, SOLO LO QUIEPO PARA QUE TENGA EL MISMO TAMAÑO QUE X
    Y[0] = y0
    h = (xn - x0)/n    
    for i in range(n):
        # METODO DE LA FORMULA 𝑦𝑖+1=𝑦𝑖+ℎ6[𝑘1+2𝑘2+2𝑘3+𝑘4]
        k1 = g(X[i],Y[i])
        k2 = g(X[i] + h/2, Y[i] + h/2 * k1)
        k3 = g(X[i] + h/2, Y[i] + h/2 * k2)
        k4 = g(X[i] + h, Y[i] + h * k3)
        Y[i+1] = Y[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    # CREO EL DICCIONARIO/una vez definido, no puedo modificarlos, de forma independiente
    ret = dict()  
    ret["x"] = X
    ret["y"] = Y
    
    return ret

respuesta_6 = RK44(g, 1, 2, 0.01, 10)
print("El método de RK44 entrega una raíz en x=" + str(respuesta_6))

def metodo_Newton_Rhapson(x0,f,df,t): 
    while True:
        x1 = x0 - (f(x0)/df(x0))
        if np.abs(x0-x1)<t:
            break
        x0 = x1
    return x1

respuesta_7 = metodo_Newton_Rhapson(1,f,df,0.01)
print("El método de Newton-Rhapson entrega una raíz en x=" + str(respuesta_7))

def steffensen(f, x0, tol):   
    while True:
        x1 = x0 - (f(x0)**2/(f(x0+f(x0))-f(x0)))
        if np.abs(x0-x1)<tol:
            break
        x0 = x1
    return x1

respuesta_8 = steffensen(f, 1, 0.01)
print("El método de Steffensen entrega una raíz en x=" + str(respuesta_8))


def metodo_euler(g, x0, xn, y0, n): 
    
    X = np.linspace(x0,xn,n+1)     
    Y = np.linspace(x0,xn,n+1)                 
    Y[0] = y0      
    h = (xn-x0)/n  
    for i in range(n):
        Y[i+1] = Y[i] + g(X[i],Y[i])*h
        print(X[i+1],Y[i+1]) 
            
    ret = dict()
    ret['x'] = X
    ret['y'] = Y

    return ret

respuesta_11 = metodo_euler(g, 1, 2, 0.01, 10)
print("El método de Euler entrega una raíz en x=" + str(respuesta_11))


def metodo_euler_mejorado(g, x0, xn, y0, n): 
    
    X = np.linspace(x0,xn,n+1)         
    Y = np.linspace(x0,xn,n+1)                 
    Y[0] = y0      
    h = (xn-x0)/n      
    
    for i in range(n):
        Y[i+1] = Y[i] + (h/2)*(g(X[i],Y[i])+g(X[i+1],Y[i]+h*g(X[i],Y[i])))   
        print(X[i+1], Y[i+1])
    
    #para graficar
    ret = dict()
    ret['x'] = X
    ret['y'] = Y
        
    return ret

respuesta_8 = metodo_euler_mejorado(g, 1, 2, 0.01, 10)
print("El método de Euler mejorado entrega una raíz en x=" + str(respuesta_8))


def Error_euler(g, y, x0, xn, y0, n): 
    
    X = np.linspace(x0,xn,n+1)         
    Y = np.linspace(x0,xn,n+1)
    E = np.linspace(x0,xn,n+1)
    Y[0] = y0      
    h = (xn-x0)/n      
    
    for i in range(n):
        Y[i+1] = Y[i] + (h/2)*(g(X[i],Y[i])+g(X[i+1],Y[i]+h*g(X[i],Y[i])))   
        E[i+1] = np.abs(y(X[i+1])-Y[i+1])
    F=E[1::]
    return max(F)

        
def y(x): return x*np.log(x)+2*x
error_2=Error_euler(g, y , 1, 10, 2, 45)   
print("El error del método de Euler mejorado es=" + str(error_2))


def metodo_RK4(g, x0, xn, y0, n):
    
    X = np.linspace(x0,xn,n+1)
    Y = np.linspace(x0,xn,n+1)                 
    Y[0] = y0        
    h = (xn-x0)/n     
    
    for i in range(n):
        K1=g(X[i],Y[i]) 
        K2=g(X[i]+h/2,Y[i]+(h/2)*K1) 
        K3=g(X[i]+h/2,Y[i]+(h/2)*K2) 
        K4=g(X[i]+h,Y[i]+h*K3)
        Y[i+1] = Y[i] +(h/6)*(K1+2*K2+2*K3+K4)
        
        print(X[i+1],Y[i+1])
        
    coord = dict()
    coord['x'] = X
    coord['y'] = Y
        
    return coord

respuesta_9 = metodo_RK4(g, 1, 2, 0.01, 10)
print("El método de RK4 entrega una raíz en x=" + str(respuesta_9))


def metodoBolzano(f, a, b, tol):
    while True:
        x = (a+b)/2
        if np.abs(a-x)<tol:
            break
        if f(a)*f(x)<0:
            b = x
        else:
            a = x
    return x

respuesta_10 = metodoBolzano(f, 1, 2, 0.01)
print("El método de Bolzano entrega una raíz en x=" + str(respuesta_10))


def heaviside(x):
    if x<0:
        return 0
    elif x>=0:
        return 1
    

#graficar todo
plt.figure(figsize=(10,10))
plt.plot(respuesta_5["x"],respuesta_5["y"],label="RK22")
plt.plot(respuesta_6["x"],respuesta_6["y"],label="RK44")
plt.plot(respuesta_9["x"],respuesta_9["y"],label="RK4")
plt.legend()
plt.show()


#funcion donde le doy un array de n funciones para sacar el heaviside
def heaviside_array(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        y[i] = heaviside(x[i])
    return y



t,s = sp.symbols("t,s")
h = t**2
sp.laplace_transform(h,t,s)
sp.laplace_transform(h,t,s, noconds=True)

F= 2/s**3
sp.inverse_laplace_transform(F,s,t, noconds = True)

print("La transformada de Laplace de t^2 es: " + str(sp.laplace_transform(h,t,s)))
print("La transformada inversa de Laplace de 2/s^3 es: " + str(sp.inverse_laplace_transform(F,s,t)))


A= sp.Symbol("A", real=True, positive = True)
B,C =sp.symbols("B,C", real = True)
j=sp.exp(A*t)*(sp.cos(B*t) + sp.sin(C*t))
print("La transformada de Laplace de j es: " + str(sp.laplace_transform(j,t,s)))


H = (s*(s+1))/(s+4)**2
sp.inverse_laplace_transform(H,s,t,noconds = True)
print("La transformada inversa de Laplace de H es: " + str(sp.inverse_laplace_transform(H,s,t)))

H.apart(s) #obtener fracciones parciales, util para comparar si el resultado es correcto aunque otra vez es muy teorico
# en pocas palabras tu confia en el pc.


i=3-sp.Heaviside(t-1)*3+sp.Heaviside(t-1)*(t-1)-sp.Heaviside(t-2)*(t-1)+sp.Heaviside(t-2)*t
sp.plot(i, xlim=(0, 10), ylim=(0, 10))

I=sp.laplace_transform(i,t,s,noconds=True)

#le añado la igualdad a la funcion de laplace
J=i/(s**3+2*s**2)+1/(s*(s**3+2*s**2))+1/(s**3+2*s**2)
#obtengo la inversa
j=sp.inverse_laplace_transform(J,s,t,noconds=True)

sp.plot(j, xlim=(-10, 10), ylim=(0, 10))


from sympy import symbols, laplace_transform, Function

def laplace_transform_function(expr):

    t, s = symbols('t s')
    F = Function('F')(t)
    return laplace_transform(expr, t, s, noconds=True)

# Example usage
t = symbols('t')
expr = t**2  # Example expression, t^2
laplace_transform_function(expr)


def inverse_laplace_transform_function(expr):
    s, t = symbols('s t')
    return inverse_laplace_transform(expr, s, t)

# Example usage
s = symbols('s')
expr = 2/s**3  # Example expression, 2/s^3
inverse_laplace_transform_function(expr)



#function to give two laplace functions and calculate its sum and thein its inverse
def laplace_sum(f1,f2):
    t, s = symbols('t s')
    F = F1+F2
    f = inverse_laplace_transform(F, s, t)
    return f



#funcion para graficar
def graficar(x,y):
    plt.figure(figsize=(10,10))
    plt.plot(x,y)
    plt.show()



# Newton-Raphson Method
def newton_raphson(x0, func, tol):
    # Convert func to a lambda function for numerical evaluation
    func_num = sp.lambdify(x, func, 'numpy')
    derivative_func = sp.lambdify(x, sp.diff(func, x), 'numpy')
    for _ in range(100):
        x1 = x0 - func_num(x0) / derivative_func(x0)
        if abs(x0 - x1) < tol:
            return x1
        x0 = x1
    return x1

# Bisection Method
def bisection(a, b, tol, func):
    while np.abs(a - b) >= tol:
        mid = (a + b) / 2.0
        if func(a) * func(mid) < 0:
            b = mid
        elif func(mid) * func(b) < 0:
            a = mid
        else:
            return mid
    return mid

# Testing the methods
theta = 1
print("La función q(theta) es: ", q_numeric(theta))

root_newton = newton_raphson(1, q_symbolic(x), 0.001)
print("El método de Newton-Raphson entrega una raíz en x =", root_newton)

root_bisection = bisection(1, 2, 0.01, q_numeric)
print("El método de Bisección entrega una raíz en x =", root_bisection)

