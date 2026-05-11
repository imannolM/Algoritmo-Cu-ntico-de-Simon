# Algoritmo Cuántico de Simon

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Qiskit](https://img.shields.io/badge/Qiskit-6929C4?style=for-the-badge&logo=ibm&logoColor=white)
![IBM Quantum](https://img.shields.io/badge/IBM%20Quantum-052FAD?style=for-the-badge&logo=ibm&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

**Proyecto Terminal de Licenciatura — Ingeniería Física, UAM Azcapotzalco (2022)**  
Asesor: Dr. Alejandro Kunold Bello

> Implementación completa del Algoritmo Cuántico de Simon en Python/Qiskit, con simulación en Qiskit Aer y ejecución en hardware cuántico real de IBM (ibm_nairobi). El algoritmo de Simon es el primero en demostrar una ventaja exponencial sobre su contraparte clásica y es la base teórica del Algoritmo de Shor.

---

## 📋 Descripción del proyecto

El **Problema de Simon** consiste en determinar si una función oráculo $f: \{0,1\}^n \to \{0,1\}^n$ es 1:1 o 2:1, y en el caso 2:1, encontrar la cadena oculta $b$ tal que $f(x) = f(x \oplus b)$ para todo $x$.

- **Solución clásica:** requiere $O(2^{n/2})$ consultas al oráculo (complejidad exponencial).
- **Solución cuántica (Simon):** requiere solo $O(n)$ consultas — **aceleración exponencial**.

Este proyecto analiza el algoritmo desde sus fundamentos matemáticos (álgebra lineal, mecánica cuántica), implementa el circuito cuántico en Qiskit, y valida los resultados tanto en simulador como en la computadora cuántica real de IBM.

---

## ⚙️ Estructura del circuito

El circuito cuántico del algoritmo opera sobre dos registros de $n$ qubits:

```
|0⟩⊗ⁿ ──H──────────────────── Qf ──H──── Medición
|0⟩⊗ⁿ ──────────────────────────────────
```

**Pasos del algoritmo:**
1. Inicializar dos registros de $n$-qubits en $|0\rangle^{\otimes n}$
2. Aplicar la transformación de Walsh-Hadamard $W_n$ al primer registro
3. Consultar el oráculo de Simon $Q_f$
4. Medir el segundo registro (colapso a $|f(x_0)\rangle$)
5. Aplicar nuevamente $W_n$ al primer registro
6. Medir el primer registro → obtener vectores $z$ con $b \cdot z = 0 \pmod{2}$
7. **Post-proceso clásico:** resolver el sistema lineal con eliminación gaussiana mod 2

---

## 🧪 Experimentos realizados

| Ejemplo | Cadena oculta $b$ | Bits ($n$) | Hardware | Resultado |
|---------|-------------------|------------|----------|-----------|
| 1 | `10` | 2 | Manual (analítico) | ✅ `b = 10` |
| 2 | `101` | 3 | Manual (analítico) | ✅ `b = 101` |
| 3 | `1010` | 4 | Qiskit Aer (simulador) | ✅ `b = 1010` |
| 4 | `001101` | 6 | Qiskit Aer (simulador) | ✅ `b = 001101` |
| 5 | `01` | 2 | **IBM Quantum (ibm_nairobi)** | ✅ `b = 01` |
| 6 | `011` | 3 | **IBM Quantum (ibm_nairobi)** | ✅ `b = 011` |

Los resultados en hardware real incluyen análisis del **ruido cuántico** y filtrado de mediciones por producto escalar $b \cdot z = 0 \pmod 2$.

---

## 📊 Resultados destacados

**Simulador Qiskit Aer — cadena `b = 101` (3 bits):**

Las 4 cadenas medidas satisfacen $b \cdot z = 0 \pmod{2}$:
```
101.000 = 0 (mod 2)
101.010 = 0 (mod 2)
101.101 = 0 (mod 2)
101.111 = 0 (mod 2)
```
Post-proceso con eliminación gaussiana → **`b = ['101']`** ✅

**Hardware real IBM (ibm_nairobi) — cadena `b = 011` (3 bits):**

El ruido cuántico genera resultados erróneos con probabilidades bajas (~8-10%). Filtrando por producto escalar y aplicando eliminación gaussiana se recupera correctamente **`b = ['011']`** ✅

---

## 🔬 Métodos clásicos de post-proceso implementados

### 1. Eliminación Gaussiana (mod 2)
Resuelve el sistema $\{z_i : b \cdot z_i = 0\}$ para obtener $b$ de forma eficiente en $O(n^2)$.

### 2. Fuerza Bruta
Verifica todos los posibles valores de $b \in \{0,1\}^n$ — útil para validación en $n$ pequeños, pero ineficiente para $n$ grandes.

---

## 🛠️ Tecnologías y dependencias

```python
qiskit          # Framework de computación cuántica de IBM
qiskit-aer      # Simulador cuántico de alto rendimiento
qiskit-ibmq-provider  # Acceso a hardware cuántico real de IBM
numpy           # Álgebra lineal y operaciones numéricas
sympy           # Álgebra simbólica (eliminación gaussiana mod 2)
matplotlib      # Visualización de histogramas de mediciones
```

### Instalación

```bash
git clone https://github.com/imannolM/simon-algorithm-quantum.git
cd simon-algorithm-quantum
pip install qiskit qiskit-aer qiskit-ibmq-provider numpy sympy matplotlib
```

---

## 🚀 Uso

### Ejecución en simulador (sin cuenta IBM)

```python
from qiskit import Aer
# Definir cadena oculta
b = '101'
n = len(b)

# Construir y simular el circuito
aer_sim = Aer.get_backend('aer_simulator')
results = aer_sim.run(simon_circuit).result()
counts = results.get_counts()
```

### Ejecución en hardware real IBM Quantum

```python
from qiskit import IBMQ
IBMQ.load_account()  # Requiere token IBM Quantum
provider = IBMQ.get_provider(hub='ibm-q')
device = provider.get_backend('ibm_nairobi')
job = execute(simon_circuit, backend=device, shots=1024)
```

> Para acceder a hardware real, se requiere una cuenta gratuita en [IBM Quantum](https://quantum-computing.ibm.com/).

---

## 📁 Estructura del repositorio

```
simon-algorithm-quantum/
│
├── algoritmo-qiskit.py      # Implementación principal en Python
├── numpy.ipynb              # Notebook Jupyter con análisis y visualizaciones
│
├── paper/
│   └── El_Algoritmo_Cuantico_de_Simon.pdf  # Reporte académico completo (UAM-A, 2022)
│
└── README.md
```

---

## 📐 Fundamentos teóricos cubiertos

El reporte académico incluye una derivación completa de:

- **Álgebra lineal cuántica:** notación de Dirac, espacios vectoriales duales, producto tensorial, descomposición espectral
- **Postulados de la Mecánica Cuántica:** superposición, colapso de función de onda, evolución temporal unitaria
- **Compuertas cuánticas:** Hadamard, CNOT, X, Y, Z, compuertas universales
- **Paralelismo cuántico y entrelazamiento**
- **Transformación de Walsh-Hadamard** y su representación $H^{\otimes n}[i,j] = \frac{1}{\sqrt{2^n}}(-1)^{i \cdot j}$
- **Construcción completa del oráculo de Simon**
- **Análisis de complejidad:** $O(n)$ vs $O(2^{n/2})$

---

## 🏆 Contexto académico

Este proyecto fue desarrollado como **Proyecto Terminal de Licenciatura** en Ingeniería Física en la Universidad Autónoma Metropolitana, Unidad Azcapotzalco (Trimestre 22-P), bajo la asesoría del Dr. Alejandro Kunold Bello.

El trabajo obtuvo el reconocimiento **ANFEI 2023** al mejor egresado nacional de Licenciatura en Ingeniería Física.

---

## 🔗 Referencias

- Nielsen & Chuang — *Quantum Computation and Quantum Information*, Cambridge University Press
- Nakahara & Ohmi — *Quantum Computing: From Linear Algebra to Physical Realizations*, CRC Press (2008)
- Simon, D.R. — *"On the Power of Quantum Computation"*, IEEE 1994
- Shor, P.W. — *"Algorithms for Quantum Computation: Discrete Logarithms and Factoring"*, IEEE 1994
- [IBM Quantum](https://quantum-computing.ibm.com/) | [Qiskit Documentation](https://qiskit.org/)

---

## 👤 Autor

**Osiris Imannol De Jesús Monroy**  
Ingeniero Físico — UAM Azcapotzalco  
Próximamente: Maestría en IA y Ciencia de Datos — IPN

[![LinkedIn](https://img.shields.io/badge/LinkedIn-imannol--monroy-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/imannol-monroy)
[![GitHub](https://img.shields.io/badge/GitHub-imannolM-181717?style=flat&logo=github)](https://github.com/imannolM)
