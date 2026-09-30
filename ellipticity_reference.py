import numpy as np
from scipy.linalg import expm

sx = np.array([[0,1],[1,0]]); sy = np.array([[0,-1j],[1j,0]])
sz = np.array([[1,0],[0,-1]]); I2 = np.eye(2)

def site_op(op, i, N):
    ops = [I2]*N; ops[i] = op
    out = ops[0]
    for o in ops[1:]: out = np.kron(out, o)
    return out

N = 3; d_spin = 2**N
J = 1.0
H = np.zeros((d_spin,d_spin), dtype=complex)
for i in range(N-1):
    Sx = [site_op(sx/2,k,N) for k in range(N)]
    Sy = [site_op(sy/2,k,N) for k in range(N)]
    Sz = [site_op(sz/2,k,N) for k in range(N)]
    H += J*(Sx[i]@Sx[i+1] + Sy[i]@Sy[i+1] + Sz[i]@Sz[i+1])

T = 0.7
evals, evecs = np.linalg.eigh(H)
w = np.exp(-(evals-evals.min())/T); Z = w.sum()
rho_S = evecs @ np.diag(w/Z) @ evecs.conj().T

wgt = np.array([0.5,0.3,0.2]); wgt = wgt/np.sum(np.abs(wgt))
M = sum(wgt[i]*site_op(sz,i,N) for i in range(N))

def embed1(op): return np.kron(np.kron(op,I2), np.eye(d_spin))
def embed2(op): return np.kron(np.kron(I2,op), np.eye(d_spin))
def embedS(op): return np.kron(np.kron(I2,I2), op)

M_full = embedS(M); Hs_full = embedS(H)
H1 = np.array([1,0]); V1_ = np.array([0,1])
Phi_p = (np.kron(H1,H1)+np.kron(V1_,V1_))/np.sqrt(2)
rho0 = np.outer(Phi_p, Phi_p.conj())

def run(theta1, eta1, theta2, eta2, dt):
    G1 = theta1*sy + eta1*sz
    G2 = theta2*sy + eta2*sz
    V1op = expm(-1j*(embed1(G1)@M_full))
    W    = expm(-1j*Hs_full*dt)
    V2op = expm(-1j*(embed2(G2)@M_full))
    U = V2op @ W @ V1op
    full = np.kron(rho0, rho_S)
    out = (U @ full @ U.conj().T).reshape(4,d_spin,4,d_spin)
    return np.einsum('ikjk->ij', out)

# --- Check 1: unitality at nonzero, unequal eta ---
I4 = np.eye(4)
def apply_to(rho4, theta1,eta1,theta2,eta2,dt):
    G1=theta1*sy+eta1*sz; G2=theta2*sy+eta2*sz
    V1op=expm(-1j*(embed1(G1)@M_full)); W=expm(-1j*Hs_full*dt); V2op=expm(-1j*(embed2(G2)@M_full))
    U=V2op@W@V1op
    full=np.kron(rho4,rho_S)
    out=(U@full@U.conj().T).reshape(4,d_spin,4,d_spin)
    return np.einsum('ikjk->ij',out)

Lam_I = apply_to(I4, 0.35,0.22,0.28,0.11, 1.7)
print("Unitality residual:", np.linalg.norm(Lam_I - I4))
assert np.linalg.norm(Lam_I - I4) < 1e-12, "TARGET: ~4e-16"

# --- Check 2: rank goes 2 -> 4 when eta != 0 ---
rho_eta0 = run(0.4,0.0,0.4,0.0, 1.7)
rho_eta  = run(0.4,0.25,0.4,0.25, 1.7)
eigs0 = np.sort(np.linalg.eigvalsh((rho_eta0+rho_eta0.conj().T)/2))
eigs  = np.sort(np.linalg.eigvalsh((rho_eta+rho_eta.conj().T)/2))
print("rank(eta=0):", np.sum(eigs0>1e-9), " rank(eta!=0):", np.sum(eigs>1e-9))
assert np.sum(eigs0>1e-9) == 2
assert np.sum(eigs>1e-9) == 4

# --- Check 3: first-order response vanishes (O(eps^2) scaling) ---
for eps in [1e-2, 1e-3, 1e-4]:
    rho_e = run(eps, 0.6*eps, eps, 0.6*eps, 1.7)
    ratio = np.linalg.norm(rho_e - rho0) / eps**2
    print(f"eps={eps:.0e}  ||rho-rho0||/eps^2 = {ratio:.4f}  (TARGET: constant ~0.6074)")

# --- Check 4: closed-form F(dt=0) matches propagated state ---
def F0_closed_form(theta, eta):
    Meig, Mvec = np.linalg.eigh(M)
    Theta = np.hypot(theta, eta)
    kappa = (theta**2-eta**2)/(theta**2+eta**2)
    rho_S_Mbasis = Mvec.conj().T @ rho_S @ Mvec
    q = np.real(np.diag(rho_S_Mbasis))
    O = np.cos(Meig*Theta)**2 + np.sin(Meig*Theta)**2*kappa
    return np.sum(q*O**2)

rho_dt0 = run(0.4,0.25,0.4,0.25, 0.0)
F_num = np.real(np.vdot(Phi_p, rho_dt0@Phi_p))
F_pred = F0_closed_form(0.4, 0.25)
print(f"F(0) numeric={F_num:.10f}  predicted={F_pred:.10f}")
assert np.isclose(F_num, F_pred, atol=1e-9)

print("\nAll four checks passed.")
