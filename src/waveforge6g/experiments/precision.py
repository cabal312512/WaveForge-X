"""Tiny evaluator-only Decimal reference for independent arithmetic checks."""

from decimal import Decimal, localcontext

import numpy as np


def decimal_reference(matrix, y, noise, digits=80):
    """Solve the exact represented binary inputs at high decimal precision.

    Real embedding avoids a third-party multiprecision dependency. This is only
    for N<=16 adversarial checks, never a deployed receive path or fallback.
    """
    n = len(y)
    if n > 16:
        raise ValueError("tiny evaluator only")
    real = np.block([[matrix.real, -matrix.imag], [matrix.imag, matrix.real]])
    with localcontext() as context:
        context.prec = digits
        c = [[Decimal.from_float(float(v)) for v in row] for row in real]
        values = [Decimal.from_float(float(v)) for v in np.r_[y.real, y.imag]]
        variance = Decimal.from_float(float(noise))
        size = 2*n
        a = [[sum(c[k][i]*c[k][j] for k in range(size))+(variance if i == j else 0)
              for j in range(size)] for i in range(size)]
        b = [sum(c[k][i]*values[k] for k in range(size)) for i in range(size)]
        for j in range(size):
            pivot = max(range(j, size), key=lambda i: abs(a[i][j]))
            a[j], a[pivot] = a[pivot], a[j]
            b[j], b[pivot] = b[pivot], b[j]
            for i in range(j+1, size):
                multiplier = a[i][j]/a[j][j]
                for k in range(j+1, size):
                    a[i][k] -= multiplier*a[j][k]
                b[i] -= multiplier*b[j]
        result = [Decimal(0)]*size
        for j in range(size-1, -1, -1):
            result[j] = (b[j]-sum(a[j][k]*result[k] for k in range(j+1, size)))/a[j][j]
    return np.array([float(result[i])+1j*float(result[n+i]) for i in range(n)])
