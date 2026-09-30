"""Two parity blocks give F(w)=P3(w)/Q2(w), with singular cases separate.

For a positive 2x2 unnormalized block B, its SLD contribution is
  [2 tr(B'^2) - tr(B')^2 + det(B)'^2/det(B)] / tr(B).
Here B=[[a(1-w), c sqrt(w(1-w))], [c* sqrt(w(1-w)), b w]].
Each numerator is quadratic and each trace linear. Summing the two parity
blocks gives a cubic over a quadratic, generically a quartic stationarity
equation. Primes are temperature-parameter derivatives with w held fixed.
"""
import numpy as np
from numpy.polynomial import Polynomial as Poly

PARITY_BASIS = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                         [0, 1, 0, -1], [1, 0, -1, 0]])/np.sqrt(2)


def rational_coefficients(k, dk):
    if max(np.linalg.norm(m-m[::-1, ::-1]) for m in (k, dk)) > 1e-12:
        raise ValueError('Parity invariance is required')
    kp, dp = [PARITY_BASIS.T @ m @ PARITY_BASIS for m in (k, dk)]
    numerators, denominators = [], []
    for indices in ([0, 1], [2, 3]):
        block, deriv = [m[np.ix_(indices, indices)]/2 for m in (kp, dp)]
        a, b, c = block[0, 0].real, block[1, 1].real, block[0, 1]
        da, db, dc = deriv[0, 0].real, deriv[1, 1].real, deriv[0, 1]
        det = a*b-abs(c)**2
        if det <= 1e-14 or min(a, b) <= 1e-14:
            raise ValueError('Singular block: use the spectral QFI including its support cutoff')
        ddet = da*b+a*db-2*np.real(c.conjugate()*dc)
        num = Poly([da, -da-db])**2 + Poly([0, 1, -1])*(4*abs(dc)**2+ddet**2/det)
        numerators.append(num)
        denominators.append(Poly([a, b-a]))
    p = numerators[0]*denominators[1]+numerators[1]*denominators[0]
    q = denominators[0]*denominators[1]
    return p, q


def stationary_candidates(k, dk):
    """Interior real roots plus endpoints; assess endpoints with spectral QFI."""
    p, q = rational_coefficients(k, dk)
    equation = p.deriv()*q-p*q.deriv()
    roots = equation.roots()
    return np.array([0., 1.] + [float(r.real) for r in roots
                    if abs(r.imag) < 1e-8 and 0 < r.real < 1])
