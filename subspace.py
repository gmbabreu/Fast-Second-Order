"""
subspace.py
===========
Subspace Gauss-Newton optimizer experiments on MNIST.

  Method 0 — Pure Subspace GN:        Δθ = P Δu
  Method 1 — Hybrid:                  Δθ = P Δu + AdamW((I − PP^T)g)
  Method 2 — Gradient-Augmented GN:   P' = QR([G, g]),  Δθ = P' Δu'
  Baseline — AdamW (standard)

P ∈ R^{M×k}: random Gaussian columns (optionally orthonormalised via QR).

For Method 2, the subspace is span({k random Gaussians} ∪ {g}), so the GN
solve is guaranteed to be at least as good as the hybrid method, since
span(P') ⊇ span(P) and g lies in span(P') too.

H_red = JP^T JP / B  built via forward-mode JVP — no M-dimensional HVPs.

Install:
    pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    pip install flax optax matplotlib
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random, jvp
import flax.linen as nn
import optax
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from typing import Sequence
import time
import os
import gzip
import struct
import urllib.request

print("JAX devices:", jax.devices())


# ═══════════════════════════════════════════════════════════════
# 1. Dataset — MNIST
# ═══════════════════════════════════════════════════════════════

def _download_if_needed(url: str, dest_path: str):
    if os.path.exists(dest_path):
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {os.path.basename(dest_path)}...")
    urllib.request.urlretrieve(url, dest_path)


def _read_idx_images_gz(path: str):
    with gzip.open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return jnp.array(data.reshape(n, rows * cols), dtype=jnp.uint8)


def _read_idx_labels_gz(path: str):
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number in {path}: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return jnp.array(data.reshape(n,), dtype=jnp.uint8)


def load_mnist(cache_dir: str = './data', train_limit: int = 60000,
               val_limit: int = 10000):
    base = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images':  't10k-images-idx3-ubyte.gz',
        'test_labels':  't10k-labels-idx1-ubyte.gz',
    }
    for fname in files.values():
        _download_if_needed(base + fname, os.path.join(cache_dir, fname))

    X_train = _read_idx_images_gz(os.path.join(cache_dir, files['train_images']))
    Y_train = _read_idx_labels_gz(os.path.join(cache_dir, files['train_labels']))
    X_test  = _read_idx_images_gz(os.path.join(cache_dir, files['test_images']))
    Y_test  = _read_idx_labels_gz(os.path.join(cache_dir, files['test_labels']))

    return (
        X_train[:train_limit], Y_train[:train_limit],
        X_test[:val_limit],    Y_test[:val_limit],
    )


# ═══════════════════════════════════════════════════════════════
# 2. MLP (Flax)
# ═══════════════════════════════════════════════════════════════

class MLP(nn.Module):
    hidden: Sequence[int]
    d_out: int

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.tanh(x)
        return nn.Dense(self.d_out)(x)


# ═══════════════════════════════════════════════════════════════
# 3. Loss and metric
# ═══════════════════════════════════════════════════════════════

def cross_entropy_loss(params, model, X, Y):
    logits = model.apply(params, X)
    labels = Y.astype(jnp.int32)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))


def classification_accuracy(params, model, X, Y):
    logits = model.apply(params, X)
    preds  = jnp.argmax(logits, axis=-1)
    labels = Y.astype(jnp.int32)
    return jnp.mean(preds == labels)


# ═══════════════════════════════════════════════════════════════
# 4. Flat ↔ pytree helpers
# ═══════════════════════════════════════════════════════════════

def params_to_vec(params) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(params)
    return jnp.concatenate([l.ravel() for l in leaves])


def vec_to_params(vec, params_ref):
    leaves, treedef = jax.tree_util.tree_flatten(params_ref)
    new_leaves, idx = [], 0
    for leaf in leaves:
        n = leaf.size
        new_leaves.append(vec[idx: idx + n].reshape(leaf.shape))
        idx += n
    return treedef.unflatten(new_leaves)


# ═══════════════════════════════════════════════════════════════
# 5. JP matrix via forward-mode JVP
#
#  JP = JVP(f, θ, P)  ∈ R^{B*D_out × k}
#  Built by vmapping a single-column JVP over the k columns of P.
#  Cost: k × (one forward pass).  No M-dimensional intermediates.
# ═══════════════════════════════════════════════════════════════

def jp_matrix(params, model, X, P):
    def fwd(p):
        return model.apply(p, X)

    def jvp_col(v_vec):
        v_tree = vec_to_params(v_vec, params)
        _, out = jvp(fwd, (params,), (v_tree,))
        return out                                    # (B, D_out)

    JP = vmap(jvp_col, in_axes=1, out_axes=-1)(P)    # (B, D_out, k)
    B  = X.shape[0]
    return JP.reshape(B * JP.shape[1], P.shape[1])   # (B*D_out, k)


def jvp_single(params, model, X, v_vec):
    """Compute J @ v for a single vector v ∈ R^M. Returns shape (B*D_out,)."""
    v_tree = vec_to_params(v_vec, params)
    _, out = jvp(lambda p: model.apply(p, X), (params,), (v_tree,))
    return out.reshape(-1)


# ═══════════════════════════════════════════════════════════════
# 6. Subspace constructors
# ═══════════════════════════════════════════════════════════════

def random_subspace(key, M: int, k: int, use_qr: bool = True) -> jnp.ndarray:
    """
    Returns P ∈ R^{M×k} with orthonormal columns (use_qr=True, default)
    or column-normalised vectors (use_qr=False).
    """
    A = random.normal(key, (M, k))
    if use_qr:
        P, _ = jnp.linalg.qr(A)   # true orthonormal basis
        return P
    return A / jnp.linalg.norm(A, axis=0, keepdims=True)



# ═══════════════════════════════════════════════════════════════
# 7. Residual + loss (one forward pass, no extra cost)
# ═══════════════════════════════════════════════════════════════

def _residual_and_loss(params, model, X, Y):
    pred   = model.apply(params, X)                        # (B, D_out)
    probs  = jax.nn.softmax(pred, axis=-1)
    onehot = jax.nn.one_hot(Y.astype(jnp.int32), pred.shape[-1])
    r      = (probs - onehot).reshape(-1)                  # (B*D_out,)
    loss   = -jnp.mean(jnp.sum(onehot * jax.nn.log_softmax(pred, axis=-1),
                                axis=-1))
    return loss, r


# ═══════════════════════════════════════════════════════════════
# 8. GN subspace update helpers
# ═══════════════════════════════════════════════════════════════

def _gn_solve(JP, r, B, k, damping):
    """Solve the reduced GN system: (JP^T JP / B + λI) Δu = -JP^T r / B."""
    H_red = JP.T @ JP 
    g_red = JP.T @ r 
    return jnp.linalg.solve(H_red, -g_red)


def gn_subspace_update_m0(params, model, loss_fn, X, Y, P, damping):
    """Pure subspace GN — no VJP needed."""
    k       = P.shape[1]
    B       = X.shape[0]
    JP      = jp_matrix(params, model, X, P)
    loss, r = _residual_and_loss(params, model, X, Y)
    delta_u     = _gn_solve(JP, r, B, k, damping)
    delta_theta = P @ delta_u
    ls_err      = jnp.sum((JP @ delta_u + r) ** 2)
    return delta_theta, loss, ls_err


def _loss_grad_residual(params, model, loss_fn, X, Y):
    """
    Single combined pass: loss, grad_tree, g_vec, residual.

    jax.value_and_grad over _residual_and_loss gives us loss + gradient in
    one forward+backward.  The residual r comes out as the primal return
    value (has_aux=True treats it as auxiliary output).  This avoids the
    double-forward that would occur if grad and _residual_and_loss were
    called separately.
    """
    (loss, r), g_tree = jax.value_and_grad(
        lambda p: _residual_and_loss(p, model, X, Y), has_aux=True)(params)
    g = params_to_vec(g_tree)
    return loss, g_tree, g, r


def gn_subspace_update_m1(params, model, loss_fn, X, Y, P, damping):
    """Hybrid GN + AdamW complement. Returns JP and r for LS error computation."""
    k                    = P.shape[1]
    B                    = X.shape[0]
    loss, g_tree, g, r   = _loss_grad_residual(params, model, loss_fn, X, Y)
    JP                   = jp_matrix(params, model, X, P)
    delta_u              = _gn_solve(JP, r, B, k, damping)
    delta_theta          = P @ delta_u
    return delta_theta, g, g_tree, loss, JP, r

  


# ═══════════════════════════════════════════════════════════════
# 9. Line search helpers
#
#  Backtracking grid search over α ∈ {2^{-i/2} : i=0,...,n_steps-1}.
#  For each candidate α we evaluate the *true* loss at
#      θ_candidate = θ_flat + α * direction
#  and pick the α that gives the lowest loss.
#
#  This is kept outside JIT deliberately: each α evaluation is a
#  separate JAX call, and we need Python-level branching to pick the
#  best one.  The per-evaluation cost is one forward pass.
# ═══════════════════════════════════════════════════════════════

# Default candidate step sizes: 2^{0}, 2^{-1/2}, ..., 2^{-4/2}
# (matches the paper's grid for full GN; extend ls_n_steps for layerwise)
_DEFAULT_LS_ALPHAS = tuple(2.0 ** (-i / 2) for i in range(5))


@partial(jit, static_argnums=(1, 2))
def _eval_loss(params, model, loss_fn, X, Y):
    """Single forward pass returning scalar loss. Used by line search."""
    return loss_fn(params, model, X, Y)


def line_search(params_flat, direction, params_ref, model, loss_fn, X, Y,
                alphas=_DEFAULT_LS_ALPHAS):
    """
    Pick α* = argmin_{α ∈ alphas} L(θ + α * direction).

    Parameters
    ----------
    params_flat : 1-D jnp.ndarray  — current parameters as flat vector
    direction   : 1-D jnp.ndarray  — proposed update direction (already
                  incorporates any lr scaling from the GN solve)
    params_ref  : pytree            — used only for shape reference
    alphas      : tuple of floats   — candidate step sizes in decreasing order

    Returns
    -------
    best_alpha  : float
    best_loss   : float
    """
    best_alpha = alphas[0]
    best_loss  = float('inf')

    for alpha in alphas:
        candidate_flat   = params_flat + alpha * direction
        candidate_params = vec_to_params(candidate_flat, params_ref)
        loss_val         = float(_eval_loss(candidate_params, model, loss_fn, X, Y))
        if loss_val < best_loss:
            best_loss  = loss_val
            best_alpha = alpha

    return best_alpha, best_loss


# ═══════════════════════════════════════════════════════════════
# 10. Jitted step functions
# ═══════════════════════════════════════════════════════════════

@partial(jit, static_argnums=(1, 2))
def _direction_pure_subspace(params, model, loss_fn, X, Y, P, damping, lr):
    """Returns (lr * delta_theta, loss, ls_err) — does NOT apply the update."""
    delta_theta, loss, ls_err = gn_subspace_update_m0(
        params, model, loss_fn, X, Y, P, damping)
    return lr * delta_theta, loss, ls_err


def step_pure_subspace(params, model, loss_fn, X, Y, P, damping, lr,
                       use_line_search=True, ls_alphas=_DEFAULT_LS_ALPHAS):
    direction, loss, ls_err = _direction_pure_subspace(
        params, model, loss_fn, X, Y, P, damping, lr)
    flat = params_to_vec(params)
    if use_line_search:
        alpha, loss = line_search(flat, direction, params, model, loss_fn, X, Y,
                                  alphas=ls_alphas)
    else:
        alpha = 1.0
    new_params = vec_to_params(flat + alpha * direction, params)
    return new_params, loss, ls_err


@partial(jit, static_argnums=(2, 3, 4))
def _direction_hybrid_complement(params, opt_state_comp, opt_comp,
                                 model, loss_fn, X, Y, P, damping, lr):
    """
    Returns (lr * gn_direction, adam_complement_vec, opt_state_new, loss).
    The GN direction and complement Adam update are kept separate so that
    line search can scale only the GN component.
    """
    delta_theta, g, g_tree, loss, JP, r = gn_subspace_update_m1(
        params, model, loss_fn, X, Y, P, damping)

    # Run Adam on the full gradient first, then project into the complement.
    updates, opt_state_comp_new = opt_comp.update(g_tree, opt_state_comp, params)
    adam_update_vec   = params_to_vec(updates)
    adam_update_vec   = adam_update_vec - P @ (P.T @ adam_update_vec)

    # LS error of the total update direction (GN + adam complement).
    # J * delta_theta = JP @ delta_u  (already computed).
    # J * adam_vec requires one extra JVP.
    delta_u   = jnp.linalg.solve(JP.T @ JP, -(JP.T @ r))   # reuse — cheap
    J_gn      = JP @ delta_u
    J_adam    = jvp_single(params, model, X, adam_update_vec)
    ls_err    = jnp.sum((J_gn + J_adam + r) ** 2)

    return lr * delta_theta, adam_update_vec, opt_state_comp_new, loss, ls_err


def step_hybrid_complement(params, opt_state_comp, opt_comp,
                           model, loss_fn, X, Y, P, damping, lr,
                           use_line_search=True, ls_alphas=_DEFAULT_LS_ALPHAS):
    gn_dir, adam_vec, opt_state_comp_new, loss, ls_err = _direction_hybrid_complement(
        params, opt_state_comp, opt_comp, model, loss_fn, X, Y, P, damping, lr)
    flat = params_to_vec(params)
    if use_line_search:
        # Line-search scales only the GN subspace direction; complement is fixed.
        alpha, loss = line_search(flat + adam_vec, gn_dir, params,
                                  model, loss_fn, X, Y, alphas=ls_alphas)
    else:
        alpha = 1.0
    new_params = vec_to_params(flat + adam_vec + alpha * gn_dir, params)
    return new_params, opt_state_comp_new, loss, ls_err


@partial(jit, static_argnums=(2, 3, 4, 9))
def _direction_grad_augmented(params, opt_state_comp, opt_comp, model, loss_fn,
                               X, Y, P_rand, damping, use_qr, lr):
    """Returns (lr * delta_theta, opt_state_comp_new, loss)."""
    loss, g_tree, g, r = _loss_grad_residual(params, model, loss_fn, X, Y)

    # Pass g through Adam to get a preconditioned direction, use that as the
    # augmented column instead of the raw gradient.
    updates, opt_state_comp_new = opt_comp.update(g_tree, opt_state_comp, params)
    g_adam = params_to_vec(updates)

    G_aug = jnp.concatenate([P_rand, g_adam[:, None]], axis=1)   # (M, k+1)
    if use_qr:
        P_prime, _ = jnp.linalg.qr(G_aug)
    else:
        P_prime = G_aug / jnp.linalg.norm(G_aug, axis=0, keepdims=True)

    k_prime     = P_prime.shape[1]
    B           = X.shape[0]
    JP          = jp_matrix(params, model, X, P_prime)

    delta_u     = _gn_solve(JP, r, B, k_prime, damping)
    delta_theta = P_prime @ delta_u
    ls_err      = jnp.sum((JP @ delta_u + r) ** 2)

    return lr * delta_theta, opt_state_comp_new, loss, ls_err


def step_grad_augmented(params, opt_state_comp, opt_comp,
                        model, loss_fn, X, Y, P_rand, damping, use_qr, lr,
                        use_line_search=True, ls_alphas=_DEFAULT_LS_ALPHAS):
    """
    Gradient-augmented subspace GN step.
    P_rand holds the random columns (refreshed on schedule);
    the gradient is passed through Adam and the result used as the augmented column.
    """
    gn_dir, opt_state_comp_new, loss, ls_err = _direction_grad_augmented(
        params, opt_state_comp, opt_comp, model, loss_fn, X, Y,
        P_rand, damping, use_qr, lr)
    flat = params_to_vec(params)
    if use_line_search:
        alpha, loss = line_search(flat, gn_dir, params,
                                  model, loss_fn, X, Y, alphas=ls_alphas)
    else:
        alpha = 1.0
    new_params = vec_to_params(flat + alpha * gn_dir, params)
    return new_params, opt_state_comp_new, loss, ls_err


@partial(jit, static_argnums=(2, 3, 4))
def step_adamw(params, opt_state, opt, model, loss_fn, X, Y):
    loss, grads = jax.value_and_grad(loss_fn)(params, model, X, Y)
    updates, opt_state_new = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state_new, loss


# ═══════════════════════════════════════════════════════════════
# 11. Iteration-based training loop
# ═══════════════════════════════════════════════════════════════

WARMUP_STEPS = 10


def train_iters(method: str, key, X_train, Y_train, X_val, Y_val, model,
                loss_fn,
                n_iters:    int   = 2000,
                batch_size: int   = 256,
                val_every:  int   = 50,
                metric_fn=None,
                metric_name='metric',
                preprocess_batch_fn=None,
                k=20, damping=1e-4, lr=1.0, refresh_every=50,
                use_qr: bool = True,
                adam_lr=1e-3, weight_decay=1e-2,
                use_line_search: bool = True,
                ls_alphas=_DEFAULT_LS_ALPHAS,
                ):
    """
    Train for `n_iters` recorded steps (plus WARMUP_STEPS discarded steps).

    Parameters
    ----------
    method        : 'pure_subspace' | 'hybrid_complement' |
                    'grad_augmented' | 'adamw'
    k             : Subspace dimension (random columns).
                    For grad_augmented the effective dim is k+1.
    damping       : Tikhonov damping added to H_red diagonal.
                    Default 1e-4 prevents singular solves.
    refresh_every : How often to resample the random columns of P.
                    The gradient column in grad_augmented is always fresh.
    use_qr        : Whether to orthonormalise via QR (recommended).
    use_line_search : If True, do a grid line search over ls_alphas after
                    computing the GN direction at each step.  Not applied
                    to the AdamW baseline.
    ls_alphas     : Tuple of candidate step-size multipliers evaluated by
                    the line search (default: 2^{0}, 2^{-1/2}, ..., 2^{-2}).
    """
    key, init_key, p_key = random.split(key, 3)
    params = model.init(init_key, X_train[:1])
    M = params_to_vec(params).shape[0]
    k_str = f'{k}+1(g)' if method == 'grad_augmented' else (
            k if method != 'adamw' else '—')
    print(f"  [{method}]  M={M},  k={k_str},  use_qr={use_qr}")

    # Initialise random subspace (methods that need it).
    # For grad_augmented we store raw Gaussian columns — QR is deferred to
    # inside the step function where g is appended, so a single factorisation
    # covers all k+1 columns at once.  Other methods QR here if use_qr=True.
    if method == 'adamw':
        P_rand = None
    elif method == 'grad_augmented':
        P_rand = random.normal(p_key, (M, k))   # raw, no QR yet
    else:
        P_rand = random_subspace(p_key, M, k, use_qr=use_qr)

    opt            = optax.adamw(adam_lr, weight_decay=weight_decay)
    opt_state      = opt.init(params)
    opt_comp       = optax.adamw(adam_lr, weight_decay=weight_decay)
    opt_state_comp = opt_comp.init(params)

    N = X_train.shape[0]

    iter_list, train_list, step_time_list         = [], [], []
    val_iter_list, val_loss_list, val_metric_list = [], [], []
    ls_err_list                                   = []

    # step  : total steps taken (including warmup)
    # rec_step : recorded steps (post-warmup)
    step     = 0
    rec_step = 0
    t_start  = None

    print(f"  Warming up ({WARMUP_STEPS} steps)...", end=" ", flush=True)

    def batch_iter():
        """Infinite iterator over shuffled mini-batches."""
        epoch = 0
        while True:
            perm = random.permutation(random.PRNGKey(epoch), N)
            for start in range(0, N, batch_size):
                yield (X_train[perm[start:start+batch_size]],
                       Y_train[perm[start:start+batch_size]])
            epoch += 1

    for X_b, Y_b in batch_iter():
        if preprocess_batch_fn is not None:
            X_b, Y_b = preprocess_batch_fn(X_b, Y_b)

        t0 = time.perf_counter()

        if method == 'pure_subspace':
            params, batch_loss, batch_ls_err = step_pure_subspace(
                params, model, loss_fn, X_b, Y_b, P_rand, damping, lr,
                use_line_search=use_line_search, ls_alphas=ls_alphas)
            jax.effects_barrier()
            batch_loss   = float(batch_loss)
            batch_ls_err = float(batch_ls_err)

        elif method == 'hybrid_complement':
            params, opt_state_comp, batch_loss, batch_ls_err = step_hybrid_complement(
                params, opt_state_comp, opt_comp,
                model, loss_fn, X_b, Y_b, P_rand, damping, lr,
                use_line_search=use_line_search, ls_alphas=ls_alphas)
            jax.effects_barrier()
            batch_loss   = float(batch_loss)
            batch_ls_err = float(batch_ls_err)

        elif method == 'grad_augmented':
            params, opt_state_comp, batch_loss, batch_ls_err = step_grad_augmented(
                params, opt_state_comp, opt_comp,
                model, loss_fn, X_b, Y_b, P_rand, damping, use_qr, lr,
                use_line_search=use_line_search, ls_alphas=ls_alphas)
            jax.effects_barrier()
            batch_loss   = float(batch_loss)
            batch_ls_err = float(batch_ls_err)

        elif method == 'adamw':
            params, opt_state, batch_loss = step_adamw(
                params, opt_state, opt, model, loss_fn, X_b, Y_b)
            jax.effects_barrier()
            batch_loss   = float(batch_loss)
            batch_ls_err = float('nan')

        step_dur = time.perf_counter() - t0
        step += 1

        # Refresh random columns on schedule.
        # grad_augmented: store raw Gaussians; QR happens inside the step
        # function once g is appended.  Other GN methods: QR here.
        if method != 'adamw' and step % refresh_every == 0:
            key, p_key = random.split(key)
            if method == 'grad_augmented':
                P_rand = random.normal(p_key, (M, k))
            else:
                P_rand = random_subspace(p_key, M, k, use_qr=use_qr)

        # Warmup gate: discard timing until compilation is done.
        if step <= WARMUP_STEPS:
            if step == WARMUP_STEPS:
                jax.effects_barrier()
                t_start = time.perf_counter()
                print("done.")
            continue

        iter_list.append(rec_step)
        train_list.append(batch_loss)
        ls_err_list.append(batch_ls_err)
        step_time_list.append(step_dur)
        rec_step += 1

        if rec_step % val_every == 0:
            X_ve, Y_ve = X_val, Y_val
            if preprocess_batch_fn is not None:
                X_ve, Y_ve = preprocess_batch_fn(X_val, Y_val)
            val_loss = float(loss_fn(params, model, X_ve, Y_ve))
            val_iter_list.append(rec_step)
            val_loss_list.append(val_loss)
            metric_msg = ""
            if metric_fn is not None:
                vm = float(metric_fn(params, model, X_ve, Y_ve))
                val_metric_list.append(vm)
                metric_msg = f"  {metric_name}={vm:.4f}"
            if rec_step % 1000 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"    iter={rec_step:5d}  train={batch_loss:.5f}"
                      f"  val={val_loss:.5f}{metric_msg}  ({elapsed:.0f}s)")

        if rec_step >= n_iters:
            break

    total_time = time.perf_counter() - t_start
    avg_ms = 1e3 * float(np.mean(step_time_list))
    print(f"  → {rec_step} iters in {total_time:.1f}s  ({avg_ms:.1f} ms/step)")

    return {
        'iter':        np.array(iter_list),
        'train_loss':  np.array(train_list),
        'ls_err':      np.array(ls_err_list),
        'val_iter':    np.array(val_iter_list),
        'val_loss':    np.array(val_loss_list),
        'val_metric':  np.array(val_metric_list),
        'step_times':  np.array(step_time_list),
        'total_time':  total_time,
    }


# ═══════════════════════════════════════════════════════════════
# 12. Sweep helpers
# ═══════════════════════════════════════════════════════════════

def run_method_comparison(key, X_train, Y_train, X_val, Y_val, model, loss_fn,
                          experiments, n_iters, batch_size, val_every,
                          metric_fn=None, metric_name=None,
                          preprocess_batch_fn=None):
    """Run a list of (method, kwargs) experiments and return results dict."""
    results = {}
    for method, extra in experiments:
        print(f"\n{'─'*40}\n  {method}\n{'─'*40}")
        key, run_key = random.split(key)
        results[method] = train_iters(
            method, run_key,
            X_train, Y_train, X_val, Y_val, model, loss_fn,
            n_iters=n_iters, batch_size=batch_size, val_every=val_every,
            metric_fn=metric_fn, metric_name=metric_name,
            preprocess_batch_fn=preprocess_batch_fn,
            **extra,
        )
    return results, key


def run_k_sweep(key, X_train, Y_train, X_val, Y_val, model, loss_fn,
                method, k_values, n_iters, batch_size, val_every,
                adamw_result=None,
                metric_fn=None, metric_name=None,
                preprocess_batch_fn=None,
                **method_kwargs):
    """
    Run one method across multiple k values, plus an AdamW baseline.
    If adamw_result is provided (precomputed), it is reused.
    Returns (k_results, adamw_result).
    """
    if adamw_result is None:
        print(f"\n{'─'*40}\n  k-sweep baseline: AdamW\n{'─'*40}")
        key, run_key = random.split(key)
        adamw_result = train_iters(
            'adamw', run_key, X_train, Y_train, X_val, Y_val, model, loss_fn,
            n_iters=n_iters, batch_size=batch_size, val_every=val_every,
            metric_fn=metric_fn, metric_name=metric_name,
            preprocess_batch_fn=preprocess_batch_fn,
        )

    k_results = {}
    for k in k_values:
        label = f'{method}_k{k}'
        print(f"\n{'─'*40}\n  {method}  k={k}\n{'─'*40}")
        key, run_key = random.split(key)
        k_results[k] = train_iters(
            method, run_key,
            X_train, Y_train, X_val, Y_val, model, loss_fn,
            n_iters=n_iters, batch_size=batch_size, val_every=val_every,
            metric_fn=metric_fn, metric_name=metric_name,
            preprocess_batch_fn=preprocess_batch_fn,
            k=k, **method_kwargs,
        )

    return k_results, adamw_result, key


# ═══════════════════════════════════════════════════════════════
# 13. Plots
# ═══════════════════════════════════════════════════════════════

STYLES = {
    'pure_subspace':     dict(color='#e07b39', lw=2, ls='-',
                              label='M0: Pure '),
    'hybrid_complement': dict(color='#2ca02c', lw=2, ls='--',
                              label='M1: Hybrid'),
    'grad_augmented':    dict(color='#9467bd', lw=2, ls='-.',
                              label='M2: Augmented P'),
    'adamw':             dict(color='#3a7ebf', lw=2, ls=':',
                              label='Baseline: AdamW'),
}


def _label_with_time(method, res):
    s = dict(STYLES[method])
    t = res.get('total_time', 0.0)
    s['label'] = f"{s['label']}  ({t:.0f}s)"
    return s

def plot_method_comparison(results: dict, n_iters: int, k: int,
                           metric_name=None, subtitle="",
                           save_path="results.png"):
    """
    Experiment A — 3 Panels:
    (A) Smoothed Train Loss, (B) Val Loss, (C) Inner LS Error.
    """
    fig, (ax_train, ax_val, ax_ls) = plt.subplots(1, 3, figsize=(18, 5))

    def get_ema(data, beta=0.95):
        """Simple Exponential Moving Average for smoother plots."""
        data = np.array(data)
        if len(data) == 0: return data
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = beta * ema[i-1] + (1 - beta) * data[i]
        return ema

    for method, res in results.items():
        if len(res['iter']) > 0:
            style = _label_with_time(method, res)
            # Plot raw noise with low alpha
            ax_train.plot(res['iter'], res['train_loss'], color=style['color'], alpha=0.15)
            # Plot EMA trend line
            ax_train.plot(res['iter'], get_ema(res['train_loss']), **style)
            
    ax_train.set_xlabel('Iteration')
    ax_train.set_ylabel('Training loss')
    ax_train.set_title('(A) Train loss (Smoothed)')
    ax_train.set_yscale('log')
    ax_train.set_xlim(0, n_iters)
    ax_train.legend(fontsize=8)
    ax_train.grid(True, alpha=0.3)

    for method, res in results.items():
        if len(res['val_iter']) > 0:
            ax_val.plot(res['val_iter'], res['val_loss'],
                        **_label_with_time(method, res))
    ax_val.set_xlabel('Iteration')
    ax_val.set_ylabel('Validation loss')
    ax_val.set_title('(B) Val loss vs iteration')
    ax_val.set_yscale('log')
    ax_val.set_xlim(0, n_iters)
    ax_val.legend(fontsize=8)
    ax_val.grid(True, alpha=0.3)

    # --- New Panel: Inner LS Error ---
    for method, res in results.items():
        # AdamW produces NaNs for LS error; we only plot valid subspace data
        if len(res['iter']) > 0 and not np.all(np.isnan(res['ls_err'])):
            style = dict(STYLES[method])
            ax_ls.plot(res['iter'], res['ls_err'], color=style['color'], alpha=0.15)
            ax_ls.plot(res['iter'], get_ema(res['ls_err']), **style)

    ax_ls.set_xlabel('Iteration')
    ax_ls.set_ylabel('Inner LS Error')
    ax_ls.set_title('(C) Inner Optimization Error')
    ax_ls.set_yscale('log')
    ax_ls.set_xlim(0, n_iters)
    ax_ls.grid(True, alpha=0.3)

    plt.suptitle(
        f'Subspace Optimizers vs AdamW — {n_iters} Iterations\n{subtitle}',
        fontsize=12, y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {save_path}")


def plot_k_sweep(k_results: dict, adamw_result: dict, k_values,
                 n_iters: int, method: str, subtitle="",
                 save_path="k_sweep.png"):
    """
    k-sweep plot — two panels:
      (A) Val loss curves per k + AdamW baseline
      (B) Final val loss vs k
    """
    method_color = STYLES.get(method, {}).get('color', '#2ca02c')
    method_label = STYLES.get(method, {}).get('label', method)

    cmap   = plt.cm.Purples if method == 'grad_augmented' else plt.cm.Greens
    colors = [cmap(0.35 + 0.55 * i / max(len(k_values) - 1, 1))
              for i in range(len(k_values))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_curves, ax_final = axes

    for (k, res), col in zip(k_results.items(), colors):
        if len(res['val_iter']) > 0:
            t = res.get('total_time', 0.0)
            ax_curves.plot(res['val_iter'], res['val_loss'],
                           color=col, lw=2, label=f'k={k}  ({t:.0f}s)')

    if len(adamw_result['val_iter']) > 0:
        t_adam = adamw_result.get('total_time', 0.0)
        adamw_style = {**STYLES['adamw'], 'label': f"AdamW baseline  ({t_adam:.0f}s)"}
        ax_curves.plot(adamw_result['val_iter'], adamw_result['val_loss'],
                       **adamw_style)

    ax_curves.set_xlabel('Iteration')
    ax_curves.set_ylabel('Validation loss')
    ax_curves.set_title(f'(A) {method_label} val loss vs iteration — varying k')
    ax_curves.set_yscale('log')
    ax_curves.set_xlim(0, n_iters)
    ax_curves.legend(fontsize=8, ncol=2)
    ax_curves.grid(True, alpha=0.3)

    final_losses = [float(res['val_loss'][-1]) if len(res['val_loss']) > 0
                    else float('nan') for res in k_results.values()]
    ax_final.plot(k_values, final_losses, 'o-', color=method_color,
                  lw=2, ms=7, label=method_label)
    if len(adamw_result['val_loss']) > 0:
        ax_final.axhline(float(adamw_result['val_loss'][-1]),
                         **STYLES['adamw'])
    ax_final.set_xlabel('Subspace dimension k (random columns)')
    ax_final.set_ylabel('Final validation loss')
    ax_final.set_title('(B) Final val loss vs k')
    ax_final.set_yscale('log')
    ax_final.legend(fontsize=8)
    ax_final.grid(True, alpha=0.3)

    plt.suptitle(
        f'{method_label} vs AdamW — varying k\n{subtitle}',
        fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {save_path}")


def print_summary_table(results: dict, metric_name=None):
    has_metric = (metric_name is not None and
                  any(len(r['val_metric']) > 0 for r in results.values()))
    w = 96 if has_metric else 72
    print(f"\n{'─'*w}")
    hdr = f"{'Method':<38} {'Iters':>7} {'ms/step':>9} {'Final val loss':>16}"
    if has_metric:
        hdr += f" {'Final ' + metric_name:>16}"
    hdr += f" {'Time (s)':>10}"
    print(hdr)
    print('─'*w)
    for method, res in results.items():
        n  = int(res['iter'][-1]) + 1 if len(res['iter']) > 0 else 0
        ms = float(np.mean(res['step_times'])) * 1e3
        vl = float(res['val_loss'][-1]) if len(res['val_loss']) > 0 else float('nan')
        t  = float(res['total_time'])
        label = STYLES.get(method, {}).get('label', method)
        row = (f"  {label:<36} {n:>7d} {ms:>8.1f}ms "
               f"{vl:>16.5f}")
        if has_metric:
            vm = float(res['val_metric'][-1]) if len(res['val_metric']) > 0 else float('nan')
            row += f" {vm:>16.4f}"
        row += f" {t:>9.1f}s"
        print(row)
    print('─'*w)


# ═══════════════════════════════════════════════════════════════
# 14. Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    key = random.PRNGKey(0)

    # ── Which experiments to run ───────────────────────────────
    RUN_A = True    # method comparison at fixed k (all 4 methods)
    RUN_C = False    # hybrid k-sweep vs AdamW
    RUN_D = False    # grad_augmented k-sweep vs AdamW

    # ── Data ──────────────────────────────────────────────────
    X_train, Y_train, X_val, Y_val = load_mnist(
        cache_dir='./data', train_limit=60000, val_limit=10000)

    def preprocess_batch(X_b, Y_b):
        return X_b.astype(jnp.float32) / 255.0, Y_b.astype(jnp.int32)

    model       = MLP(hidden=(128, 64), d_out=10)
    loss_fn     = cross_entropy_loss
    metric_fn   = classification_accuracy
    metric_name = 'accuracy'
    batch_size  = 128
    subtitle    = 'MNIST Classification, MLP (128×64), batch=128'

    # ── Shared hyperparameters ─────────────────────────────────
    N_ITERS     = 6000
    VAL_EVERY   = 50
    K           = 100        # random subspace columns for Exp A
    REFRESH     = 200        # resample random cols every N steps
    USE_QR      = True      # orthonormalise subspace via QR

    # k-sweep settings
    K_VALUES    = [25, 50, 100, 200]
    N_ITERS_K   = 2000

    # Common GN hyperparameters
    GN_COMMON = dict(
        damping=0,       # regularisation for the reduced GN solve
        lr=1.0,
        refresh_every=REFRESH,
        use_qr=USE_QR,
        adam_lr=1e-3,
        weight_decay=1e-2,
    )

    # ══════════════════════════════════════════════════════════
    # Experiment A — method comparison at fixed k
    # Includes M2 (grad_augmented) alongside M0, M1, AdamW.
    # ══════════════════════════════════════════════════════════
    if RUN_A:
        EXPERIMENTS_A = [
            ('pure_subspace',
             dict(k=K, **GN_COMMON)),
            ('hybrid_complement',
             dict(k=K, **GN_COMMON)),
            ('grad_augmented',
             dict(k=K, **GN_COMMON)),
            ('adamw',
             dict(adam_lr=1e-3, weight_decay=1e-2)),
        ]

        print("\n" + "═"*55)
        print("  EXPERIMENT A — method comparison (all 4 methods)")
        print("═"*55)
        results_A, key = run_method_comparison(
            key, X_train, Y_train, X_val, Y_val, model, loss_fn,
            EXPERIMENTS_A, N_ITERS, batch_size, VAL_EVERY,
            metric_fn=metric_fn, metric_name=metric_name,
            preprocess_batch_fn=preprocess_batch,
        )

        print_summary_table(results_A, metric_name=metric_name)
        plot_method_comparison(
            results_A, n_iters=N_ITERS, k=K,
            metric_name=metric_name,
            subtitle=f'{subtitle},  k={K}',
            save_path='results_method_comparison.png')

    # ══════════════════════════════════════════════════════════
    # Experiment C — hybrid k-sweep vs AdamW
    # ══════════════════════════════════════════════════════════
    if RUN_C:
        print("\n" + "═"*55)
        print("  EXPERIMENT C — hybrid k-sweep vs AdamW")
        print("═"*55)
        k_results_hybrid, adamw_result_c, key = run_k_sweep(
            key, X_train, Y_train, X_val, Y_val, model, loss_fn,
            method='hybrid_complement',
            k_values=K_VALUES,
            n_iters=N_ITERS_K,
            batch_size=batch_size,
            val_every=VAL_EVERY,
            metric_fn=metric_fn, metric_name=metric_name,
            preprocess_batch_fn=preprocess_batch,
            **GN_COMMON,
        )
        plot_k_sweep(
            k_results_hybrid, adamw_result_c,
            k_values=K_VALUES,
            n_iters=N_ITERS_K,
            method='hybrid_complement',
            subtitle=subtitle,
            save_path='results_hybrid_k_sweep.png')

    # ══════════════════════════════════════════════════════════
    # Experiment D — grad_augmented k-sweep vs AdamW
    # Reuses the AdamW baseline from Exp C if available.
    # ══════════════════════════════════════════════════════════
    if RUN_D:
        print("\n" + "═"*55)
        print("  EXPERIMENT D — grad_augmented k-sweep vs AdamW")
        print("═"*55)
        # Reuse AdamW baseline from Exp C if already computed.
        _adamw_baseline = adamw_result_c if RUN_C else None
        k_results_ga, adamw_result_d, key = run_k_sweep(
            key, X_train, Y_train, X_val, Y_val, model, loss_fn,
            method='grad_augmented',
            k_values=K_VALUES,
            n_iters=N_ITERS_K,
            batch_size=batch_size,
            val_every=VAL_EVERY,
            adamw_result=_adamw_baseline,
            metric_fn=metric_fn, metric_name=metric_name,
            preprocess_batch_fn=preprocess_batch,
            **GN_COMMON,
        )
        plot_k_sweep(
            k_results_ga, adamw_result_d,
            k_values=K_VALUES,
            n_iters=N_ITERS_K,
            method='grad_augmented',
            subtitle=subtitle,
            save_path='results_grad_augmented_k_sweep.png')