#Importar Qiskit
from qiskit import *
from qiskit import IBMQ, Aer
from qiskit import QuantumCircuit, transpile
from qiskit.tools.monitor import job_monitor
#Importar herramientas para graficar
from qiskit.visualization import plot_histogram
#Importar otras bibliotecas
import numpy as np
import sys
import matplotlib.pyplot as plt
from sympy import Matrix, pprint, MatrixSymbol, expand, mod_inverse

#Cadena de bits oculta
b = '101'
n = len(b)

#Construcción del algoritmo de Simon
simon_circuit = QuantumCircuit(2*n,n)
#Aplicar compuertas de Hadamard antes de consultar la función oráculo
simon_circuit.h(range(n))
#Aplicar una barrera de separación visual
simon_circuit.barrier()
#Construcción de la función oráculo (Simon Oracle)
simon_circuit.cx(range(n), range(n, 2*n))

k=0
for i in range(n-1,-1,-1):
    if b[i]=='1':
        m=n
        for j in range(n-1,-1,-1):
            if b[j]=='1':
                simon_circuit.cx(k,m)
        m+=1
    break
k+=1

#Aplicar una barrera de separación visual
simon_circuit.barrier()

#Aplicar compuertas de Hadamard al registro de entrada
simon_circuit.h(range(n))

#Medir qubits
simon_circuit.measure(range(n),range(n))

#Dibujar el circuito del algoritmo
simon_circuit.draw(output='mpl')

#Simular el circuito usando el simulador Qiskit Aer
aer_sim = Aer.get_backend('aer_simulator')
results = aer_sim.run(simon_circuit).result()
counts = results.get_counts()
plot_histogram(counts)

#Calcular el prodcuto escalar de los resultados para obtener un sistema de ecuaciones
def bdotz(b, z):
    accum = 0
    for i in range(len(b)):
        accum += int(b[i]) * int(z[i])
    return (accum % 2)

for z in counts:
    print( '{}.{} = {} (mod 2)'.format(b, z, bdotz(b,z)))

#Ejecutar el circuito en la computadora cuántica de IBM
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
device = provider.get_backend('ibm_nairobi')

job = execute(simon_circuit, backend = device, shots = 1024)
print(job.job_id())
job_monitor(job)

#Obtener y graficar los resultados del dispositivo real
device_result = job.result().get_counts()
plot_histogram(device_result)

#Calcular el prodcuto escalar de los resultados para obtener un sistema de ecuaciones
def bdotz(b, z):
    accum = 0
    for i in range(len(b)):
        accum += int(b[i]) * int(z[i])
    return (accum % 2)

for z in device_result:
    print( '{}.{} = {} (mod 2) ({:.1f}%)'.format(b, z, bdotz(b,z), device_result[z]*100/1024))

#Proceso clásico para obtener la cadena de bits oculta empleando eliminación
#gaussiana. Para los resultados obtenidos usando el simulador de Qiskit.
base = [ (k[::-1],v) for k,v in counts.items() if k != "0"*n ]
#Ordenar base por probabilidades
base.sort(key = lambda x: x[1], reverse=True)
A = []
for k, v in base:
    A.append([ int(c) for c in k ])
A = Matrix(A)

A_transformed = A.rref(iszerofunc=lambda x: x % 2==0)

#Convertir racionales y negativos en forma escalonada de fila reducida (rref)
    def mod(x,modulus):
    numer,denom = x.as_numer_denom()
    return numer*mod_inverse(denom,modulus) % modulus

#Tratar con valores negativos y fraccionarios
A_new = A_transformed[0].applyfunc(lambda x: mod(x,2))

print("La cadena de bits oculta b%d,...,b1,b0 solo satisface las ecuaciones: " %(n-1))
rows,cols = A_new.shape
for r in range(rows):
    Ar = [ "b"+str(i)+"" for i,v in enumerate(list(A_new[r,:])) if v==1]
    if len(Ar) > 0:
        tStr = " + ".join(Ar)
        print(tStr, "= 0 (mod 2)")

#Proceso clásico para obtener la cadena de bits oculta empleando fuerza bruta
def find_b(results):
    #Calcular n a partir de los resultados proporcionados
    n_ = len(next(iter(results)))
    #Una lista para almacenar los posibles valores que podrían funcionar
    possible_bs = []
    for i in range(1,2**n):
        it_works = True
        for key in results:
            if bitwise_dot(i, int(key,2)) == 1:
                it_works = False
        if it_works:
            possible_bs.append("{0:0{digits}b}".format(i, digits=n_))
    return possible_bs

#bitwise_dot realiza el producto escalar (mod 2) para usar en la búsqueda de una función
def bitwise_dot(a, b, strings=False):
    if strings:
        a_str = a
        b_str = b
    else:
        if a < b:
            return bitwise_dot(b, a)
        a_str = "{0:0b}".format(a)
        b_str = "{0:0{digits}b}".format(b, digits=len(a_str))
    result = 0
    for i in range(len(a_str)):
        if a_str[i] == '1' and b_str[i] == '1':
            result += 1
    return result % 2

#Para los resultados obtenidos usando el simulador de Qiskit.
print("La cadena de bits oculta b es",find_b(counts))

#Para obtener la cadena de bits oculta b de los resultados obtenidos de la computadora de IBM
#Creamos una variable con los resultados que tienen los porcentajes más altos y su producto
#escalar es cero
device_counts={'000':115,'010':107,'101':126,'111':135}
device_counts

#Proceso clásico para obtener la cadena de bits oculta empleando eliminación
#gaussiana. (Resultados de la computadora de IBM)

base = [ (k[::-1],v) for k,v in device_counts.items() if k != "0"*n ]

#Ordenar base por probabilidades
base.sort(key = lambda x: x[1], reverse=True)

A = []
for k, v in base:
    A.append([ int(c) for c in k ])

A = Matrix(A)

A_transformed = A.rref(iszerofunc=lambda x: x % 2==0)

#Convertir racionales y negativos en forma escalonada de fila reducida (rref)
def mod(x,modulus):
    numer,denom = x.as_numer_denom()
    return numer*mod_inverse(denom,modulus) % modulus

#Tratar con valores negativos y fraccionarios
A_new = A_transformed[0].applyfunc(lambda x: mod(x,2))

print("La cadena de bits oculta b%d,...,b1,b0 solo satisface las ecuaciones: " %(n-1))
rows,cols = A_new.shape
for r in range(rows):
    Ar = [ "b"+str(i)+"" for i,v in enumerate(list(A_new[r,:])) if v==1]
    if len(Ar) > 0:
        tStr = " + ".join(Ar)
        print(tStr, "= 0 (mod 2)")

#Proceso clásico para obtener la cadena de bits oculta empleando fuerza bruta
#(Resultados de la computadora de IBM)
print("La cadena de bits oculta b es",find_b(device_counts))