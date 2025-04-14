from sympy import symbols
from sympy.matrices import Matrix
c, d, e = symbols("c1/(d1+e1), c2/(d2+e1), c1/(d4+e5)")
A = Matrix([[c,d], [1, -e]])
b = Matrix([2, 0])
print("A:", A.solve(b))