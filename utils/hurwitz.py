#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batched Hurwitz (complex Givens) parameterization for SU(n) in NumPy (complex).

Encode:
  U_B [B, n, n] (complex128) -> theta_B [B, k], phi_B [B, k], lam_B [B, n]
  where k = n(n-1)/2 and sum(lam_B[b]) ≡ 0 (mod 2π) for every batch element.

Decode:
  (theta_B [B, k], phi_B [B, k], lam_head_B [B, n-1]) -> U_B [B, n, n]

Also included:
  su_pack_features/su_unpack_features for [B, n^2-1] vectors up to global phase.
"""

import math
import numpy as np

TWO_PI = 2.0 * math.pi
PI_OVER_2 = 0.5 * math.pi

# -------------------------- utils --------------------------

def _wrap_to_2pi(x):
    y = np.mod(x, TWO_PI)
    y = y + TWO_PI * (y < 0.0)
    return y

def _project_to_su_batch(U_B):
    """Remove global phase per batch: U <- U * exp(-i * arg(det(U))/n)."""
    B, n, _ = U_B.shape
    det = np.linalg.det(U_B)
    phi = np.angle(det) / n                  # [B]
    return U_B * np.exp(-1j * phi)[:, None, None]

def _left_apply_block_rows_inplace(A, i, j, c, s, eiph):
    """
    A <- G @ A on rows (i, j) for all batches.
    G = [[ c,        e^{iφ}s],
         [-e^{-iφ}s,  c     ]]
    c,s,eiph are shape [B].
    """
    Ai = A[:, i, :].copy()
    Aj = A[:, j, :].copy()
    A[:, i, :] = c[:, None] * Ai + (eiph * s)[:, None] * Aj
    A[:, j, :] = (-np.conj(eiph) * s)[:, None] * Ai + c[:, None] * Aj

def _left_apply_block_rows_inplace_H(A, i, j, c, s, eiph):
    """
    A <- G^H @ A on rows (i, j) for all batches.
    G^H = [[ c,       -e^{iφ}s],
           [ e^{-iφ}s,  c     ]]
    """
    Ai = A[:, i, :].copy()
    Aj = A[:, j, :].copy()
    A[:, i, :] = c[:, None] * Ai + (-eiph * s)[:, None] * Aj
    A[:, j, :] = (np.conj(eiph) * s)[:, None] * Ai + c[:, None] * Aj

# -------------------------- ENCODE --------------------------

def su_encode_batched_np(U_B, eps=1e-30, tiny=1e-300):
    """
    Encode a batch of unitaries (complex128) into Hurwitz angles for SU(n).
    Inputs:
      U_B: [B, n, n] complex128 (if 2D, it’s treated as B=1)
    Returns:
      theta_B: [B, k], phi_B: [B, k], lam_B: [B, n] with sum(lam_B[b]) ≡ 0 (mod 2π)
    """
    U_B = np.asarray(U_B)
    if U_B.ndim == 2:
        U_B = U_B[None, ...]
    assert U_B.ndim == 3 and U_B.shape[1] == U_B.shape[2]
    B, n, _ = U_B.shape
    k = n * (n - 1) // 2

    # Project to SU(n) once (numerical stability); we still enforce sum λ = 0 at the end.
    A = _project_to_su_batch(U_B).astype(np.complex128, copy=True)

    theta_B = np.zeros((B, k), dtype=np.float64)
    phi_B   = np.zeros((B, k), dtype=np.float64)

    idx = 0
    for i in range(0, n - 1):           # column
        for j in range(n - 1, i, -1):   # rows below diag (bottom-up)
            a = A[:, i, i]              # [B]
            b = A[:, j, i]              # [B]
            aa = np.abs(a)**2
            bb = np.abs(b)**2
            r  = np.sqrt(aa + bb + eps)

            theta = np.zeros(B, dtype=np.float64)
            phi   = np.zeros(B, dtype=np.float64)

            m_b_big   = bb >= tiny
            m_a_small = aa < tiny
            m_general = m_b_big & (~m_a_small)
            m_a_zero  = m_b_big & m_a_small

            # General case
            if np.any(m_general):
                ag = a[m_general]; bg = b[m_general]; rg = r[m_general]
                phi[m_general]   = _wrap_to_2pi(np.angle(ag) - np.angle(bg))
                c = np.clip(np.abs(ag) / rg, 0.0, 1.0)
                s = np.clip(np.abs(bg) / rg, 0.0, 1.0)
                theta[m_general] = np.arctan2(s, c)

            # a≈0, b≠0  -> θ = π/2, φ = -arg(b)
            if np.any(m_a_zero):
                bz = b[m_a_zero]
                theta[m_a_zero] = PI_OVER_2
                phi[m_a_zero]   = _wrap_to_2pi(-np.angle(bz))

            # b≈0 -> θ=0 (already), φ arbitrary (0)
            theta_B[:, idx] = theta
            phi_B[:, idx]   = phi

            c = np.cos(theta)
            s = np.sin(theta)
            eiph = np.exp(1j * phi)
            _left_apply_block_rows_inplace(A, i, j, c, s, eiph)
            idx += 1

    # Extract diag phases and enforce SU(n) by adjusting only the last entry
    d = np.diagonal(A, axis1=1, axis2=2)          # [B, n]
    d = d / np.clip(np.abs(d), tiny, None)
    lam = _wrap_to_2pi(np.angle(d))               # [B, n]
    lam_head = lam[:, :n-1].copy()
    lam_last = _wrap_to_2pi(-np.sum(lam_head, axis=1, keepdims=True))
    lam_full = np.concatenate([lam_head, lam_last], axis=1)
    return theta_B, phi_B, lam_full

# -------------------------- DECODE --------------------------

def su_decode_batched_np(theta_B, phi_B, lam_head_B, n):
    """
    Decode a batch to SU(n) (complex128).
    Inputs:
      theta_B: [B, k], phi_B: [B, k], lam_head_B: [B, n-1]
    Returns:
      U_B: [B, n, n] complex128, det ≈ 1, round-trip ~1e-15 in float64.
    """
    theta_B = np.asarray(theta_B, dtype=np.float64)
    phi_B   = np.asarray(phi_B,   dtype=np.float64)
    lam_head_B = np.asarray(lam_head_B, dtype=np.float64)
    B, k = theta_B.shape
    assert phi_B.shape == (B, k)
    assert lam_head_B.shape == (B, n - 1)

    # Build full λ with sum ≡ 0
    lam_last = _wrap_to_2pi(-np.sum(lam_head_B, axis=1, keepdims=True))
    lam = np.concatenate([lam_head_B, lam_last], axis=1)     # [B, n]

    # Start from D = diag(e^{i λ})
    U = np.zeros((B, n, n), dtype=np.complex128)
    idxs = np.arange(n)
    U[:, idxs, idxs] = np.exp(1j * lam)

    # Replay G^H in reverse order
    angles_idx = k - 1
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            th = theta_B[:, angles_idx]
            ph = phi_B[:,  angles_idx]
            angles_idx -= 1
            c = np.cos(th); s = np.sin(th); eiph = np.exp(1j * ph)
            _left_apply_block_rows_inplace_H(U, i, j, c, s, eiph)

    return U

# -------------------------- pack/unpack to n^2-1 features --------------------------

def su_pack_features(theta_B, phi_B, lam_B):
    """
    (θ, φ, λ) -> [B, n^2-1] by concatenating θ, φ, and the first n-1 phases.
    """
    B, k = theta_B.shape
    n = lam_B.shape[1]
    feats = np.concatenate([theta_B, phi_B, lam_B[:, :n-1]], axis=1)
    assert feats.shape[1] == n*n - 1
    return feats

def su_unpack_features(feats, n):
    """
    [B, n^2-1] -> (theta_B, phi_B, lam_head_B).
    """
    B, D = feats.shape
    k = n*(n-1)//2
    assert D == 2*k + (n-1)
    theta = feats[:, :k]
    phi   = feats[:, k:2*k]
    lamh  = feats[:, 2*k:2*k + (n-1)]
    return theta, phi, lamh

def su_encode_to_features_np(U_B):
    th, ph, lam = su_encode_batched_np(U_B)
    return su_pack_features(th, ph, lam)

def su_decode_from_features_np(feats, n):
    th, ph, lamh = su_unpack_features(feats, n)
    return su_decode_batched_np(th, ph, lamh, n)

# -------------------------- quick test --------------------------

def _random_unitary_batched_np(B, n, seed=0):
    """
    Generate a random U(n) batch on CPU using NumPy complex, shape [B, n, n].
    """
    rng = np.random.default_rng(seed)

    # Complex Gaussian, batched QR (NumPy supports stacked QR)
    Z = rng.standard_normal((B, n, n)) + 1j * rng.standard_normal((B, n, n))
    Q, R = np.linalg.qr(Z)  # [B,n,n]

    # Fix column phases so Q is unitary in a canonical way:
    diag = np.diagonal(R, axis1=1, axis2=2)                 # [B,n]
    phase = diag / np.clip(np.abs(diag), 1e-30, None)        # [B,n]
    Dphase = np.zeros((B, n, n), dtype=np.complex128)
    idx = np.arange(n)
    Dphase[:, idx, idx] = np.conj(phase)                     # diag(conj(phase))
    Q = Q @ Dphase                                           # [B,n,n]

    # Add random diagonal phase to make a generic U(n)
    lam0 = TWO_PI * rng.random((B, n))
    D = np.zeros((B, n, n), dtype=np.complex128)
    D[:, idx, idx] = np.exp(1j * lam0)
    return Q @ D

def _rel_err(U, V):
    return float(np.linalg.norm(U - V) / np.linalg.norm(V))

if __name__ == "__main__":
    B, n = 8, 4
    U = _random_unitary_batched_np(B, n, seed=123)
    U_su = _project_to_su_batch(U)

    theta, phi, lam = su_encode_batched_np(U)
    U_rec = su_decode_batched_np(theta, phi, lam[:, :n-1], n)

    rel = _rel_err(U_rec, U_su)
    dets = np.linalg.det(U_rec)
    print(f"[B={B}, n={n}] rel. error: {rel:.3e}")
    print(f"[B={B}, n={n}] |det| mean/min/max: {np.mean(np.abs(dets)):.6f} / "
          f"{np.min(np.abs(dets)):.6f} / {np.max(np.abs(dets)):.6f}")

    feats = su_encode_to_features_np(U)            # [B, n^2-1]
    U_rt  = su_decode_from_features_np(feats, n)
    print(f"[features] rel. error: {_rel_err(U_rt, U_su):.3e}, feats.shape={feats.shape}")

