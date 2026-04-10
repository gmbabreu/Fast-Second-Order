"""
full_gn_sanity.py
=================
Sanity-check for the Gauss-Newton machinery in subspace.py.

Problem
-------
  Quadratic regression:  y = X @ w_true + noise,  loss = (1/2B) ||f(θ) - y||²

  f is a linear model (no hidden layers), so J is constant and full GN
  converges to the exact least-squares solution in ONE step (damping=0).
  This makes it easy to verify correctness against a known ground truth.

Full Gauss-Newton
-----------------
  Subspace GN with P = I_M.  The only change vs subspace.py is passing
  P = jnp.eye(M) so that JP = J (the full Jacobian).  Every other code
  path — jp_matrix, _gn_solve, ls_err, results dict, plot — is identical.

Methods compared
----------------
  • Full GN            (P = I)
  • Subspace GN k=4    (random subspace, refreshed every step)
  • Subspace GN k=20
  • Subspace GN k=100
  • Hybrid complement  (GN in subspace + Adam in complement, from subspace.py m1)
  • Grad-augmented     (random P + Adam-preconditioned gradient column, from subspace.py m2)
  • AdamW baseline

What to expect
--------------
  • Full GN (λ=0)   : loss ≈ noise floor after step 1 (exact LS solution).
  • Subspace GN k=5 : slower; random 5-D subspace covers little curvature.
  • Subspace GN k=15: faster than k=5, still slower than full GN.
  • Hybrid           : GN handles the subspace, Adam cleans up the complement.
  • Grad-augmented   : augmenting with gradient direction accelerates subspace GN.
  • AdamW            : smooth but many steps to converge.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, random, jvp
import optax
import numpy as np
import matplotlib.pyplot as plt
import time

print("JAX devices:", jax.devices())


# ═══════════════════════════════════════════════════════════════
# 1.  Synthetic quadratic (linear) regression dataset
# ═══════════════════════════════════════════════════════════════

def make_regression(key, n_train=500, n_val=200, d=30, noise_std=0.1):
    """
    y = X @ w_true + b_true + ε
    X ~ N(0,1),  w_true ~ N(0,1),  ε ~ N(0, noise_std²)
    d kept small so full GN (P = I_{d+1}) is cheap.
    """
    k1, k2, k3, k4, k5, k6 = random.split(key, 6)
    w_true  = random.normal(k1, (d,))
    b_true  = random.normal(k2, ())
    X_train = random.normal(k3, (n_train, d))
    y_train = X_train @ w_true + b_true + noise_std * random.normal(k4, (n_train,))
    X_val   = random.normal(k5, (n_val, d))
    y_val   = X_val   @ w_true + b_true + noise_std * random.normal(k6, (n_val,))
    return X_train, y_train, X_val, y_val, w_true, b_true


# ═══════════════════════════════════════════════════════════════
# 2.  Linear model  (plain pytree — no Flax needed)
#     params = {'w': (d,), 'b': ()}
# ═══════════════════════════════════════════════════════════════

def init_params(key, d):
    k1, k2 = random.split(key)
    return {'w': 0.01 * random.normal(k1, (d,)), 'b': jnp.zeros(())}


def predict(params, X):
    """(B,)"""
    return X @ params['w'] + params['b']


def mse_loss(params, X, y):
    """(1/2B) ||f(θ) - y||²"""
    r = predict(params, X) - y
    return 0.5 * jnp.mean(r ** 2)


def residual_and_loss(params, X, y):
    """Returns (loss, r) where r = f(θ) - y ∈ R^B."""
    r    = predict(params, X) - y
    loss = 0.5 * jnp.mean(r ** 2)
    return loss, r


# ═══════════════════════════════════════════════════════════════
# 3.  Flat ↔ pytree helpers  (identical to subspace.py)
# ═══════════════════════════════════════════════════════════════

def params_to_vec(params):
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
# 4.  JP matrix via forward-mode JVP  (identical to subspace.py)
# ═══════════════════════════════════════════════════════════════

def jp_matrix(params, X, P):
    """
    JP = J P  ∈ R^{B × k}.
    Full GN: pass P = I_M  →  JP = J (full Jacobian).
    Cost: k forward passes via JVP.
    """
    def fwd(p):
        return predict(p, X)

    def jvp_col(v_vec):
        v_tree = vec_to_params(v_vec, params)
        _, out = jvp(fwd, (params,), (v_tree,))
        return out                              # (B,)

    return vmap(jvp_col, in_axes=1, out_axes=-1)(P)   # (B, k)


def jvp_single(params, X, v_vec):
    """J @ v  — one JVP column, v_vec is a flat parameter vector."""
    v_tree = vec_to_params(v_vec, params)
    _, out = jvp(lambda p: predict(p, X), (params,), (v_tree,))
    return out   # (B,)


# ═══════════════════════════════════════════════════════════════
# 5.  GN solve  (identical signature to subspace.py _gn_solve)
# ═══════════════════════════════════════════════════════════════

def _gn_solve(JP, r, damping):
    """
    Solve  (JP^T JP / B + λI) Δu = -JP^T r / B.
    Full GN: Δu = Δθ directly (P = I).
    """
    B = r.shape[0]
    H = JP.T @ JP / B
    g = JP.T @ r / B
    H = H + damping * jnp.eye(H.shape[0])
    return jnp.linalg.solve(H, -g)


_DEFAULT_LS_ALPHAS = tuple(2.0 ** (-i / 2) for i in range(10))


@jit
def _eval_loss(params, X, y):
    """Single forward pass returning scalar loss. Used by line search."""
    return mse_loss(params, X, y)


def line_search(params_flat, direction, params_ref, X, y,
                alphas=_DEFAULT_LS_ALPHAS):
    """Pick alpha that minimises true loss over candidate step sizes."""
    best_alpha = alphas[0]
    best_loss = float('inf')

    for alpha in alphas:
        candidate_flat = params_flat + alpha * direction
        candidate_params = vec_to_params(candidate_flat, params_ref)
        loss_val = float(_eval_loss(candidate_params, X, y))
        if loss_val < best_loss:
            best_loss = loss_val
            best_alpha = alpha
    return best_alpha, best_loss


# ═══════════════════════════════════════════════════════════════
# 6.  Step functions
# ═══════════════════════════════════════════════════════════════

def _gn_step(params, X, y, P, damping,
             use_line_search=True, ls_alphas=_DEFAULT_LS_ALPHAS):
    """
    Generic GN step for any P.
    Returns (new_params, train_loss, ls_err).
    ls_err = ||JP Δu + r||²  — inner linearised residual (same as subspace.py).
    """
    loss, r  = residual_and_loss(params, X, y)
    JP       = jp_matrix(params, X, P)          # (B, k)
    delta_u  = _gn_solve(JP, r, damping)        # (k,) or (M,) for full GN
    delta    = P @ delta_u                      # (M,)   always
    ls_err   = jnp.sum((JP @ delta_u + r) ** 2)
    flat = params_to_vec(params)
    if use_line_search:
        alpha, loss = line_search(flat, delta, params, X, y, alphas=ls_alphas)
    else:
        alpha = 1.0
    new_params = vec_to_params(flat + alpha * delta, params)
    return new_params, float(loss), float(ls_err)


def _hybrid_step(params, opt_state, opt, X, y, P, damping,
                 use_line_search=True, ls_alphas=_DEFAULT_LS_ALPHAS):
    """
    Hybrid complement step (mirrors subspace.py gn_subspace_update_m1 + step logic).

    1. Compute loss, gradient, and residual in one pass.
    2. Run Adam on the full gradient to get a preconditioned update.
    3. Project Adam update into the P-complement: adam_vec -= P (P^T adam_vec).
    4. GN solve in the P-subspace.
    5. Total update = complement_adam + gn_delta.
    ls_err accounts for both components.
    """
    # Combined loss + grad + residual (mirrors _loss_grad_residual in subspace.py)
    (loss, r), g_tree = jax.value_and_grad(
        lambda p: residual_and_loss(p, X, y), has_aux=True)(params)

    # Adam step on full gradient
    updates, opt_state_new = opt.update(g_tree, opt_state, params)
    adam_vec = params_to_vec(updates)                         # flat (M,)

    # Project Adam into complement of P  (zero if P = I)
    adam_complement = adam_vec - P @ (P.T @ adam_vec)

    # GN solve in subspace
    JP      = jp_matrix(params, X, P)                        # (B, k)
    delta_u = _gn_solve(JP, r, damping)
    gn_delta = P @ delta_u                                   # (M,)

    # LS error: linearised residual of the *combined* update
    J_gn   = JP @ delta_u                                    # (B,)
    J_adam = jvp_single(params, X, adam_complement)          # (B,)
    ls_err = float(jnp.sum((J_gn + J_adam + r) ** 2))

    flat = params_to_vec(params)
    if use_line_search:
        # Scale only GN direction; keep complement Adam update fixed.
        alpha, loss = line_search(flat + adam_complement, gn_delta,
                                  params, X, y, alphas=ls_alphas)
    else:
        alpha = 1.0
    new_params = vec_to_params(flat + adam_complement + alpha * gn_delta, params)
    return new_params, opt_state_new, float(loss), ls_err


def _grad_augmented_step(params, opt_state, opt, X, y, P_rand, damping,
                         use_qr=True,
                         use_line_search=True, ls_alphas=_DEFAULT_LS_ALPHAS):
    """
    Gradient-augmented subspace GN (mirrors subspace.py step_grad_augmented).

    1. Compute loss, gradient, and residual.
    2. Pass gradient through Adam → preconditioned direction g_adam.
    3. Augment: G = [P_rand | g_adam], then orthonormalise via QR → P'.
    4. GN solve in the augmented subspace P'.
    """
    (loss, r), g_tree = jax.value_and_grad(
        lambda p: residual_and_loss(p, X, y), has_aux=True)(params)

    updates, opt_state_new = opt.update(g_tree, opt_state, params)
    g_adam = params_to_vec(updates)[:, None]                 # (M, 1)

    G_aug = jnp.concatenate([P_rand, g_adam], axis=1)        # (M, k+1)
    if use_qr:
        P_prime, _ = jnp.linalg.qr(G_aug)
    else:
        P_prime = G_aug / jnp.linalg.norm(G_aug, axis=0, keepdims=True)

    JP      = jp_matrix(params, X, P_prime)
    delta_u = _gn_solve(JP, r, damping)
    delta   = P_prime @ delta_u
    ls_err  = float(jnp.sum((JP @ delta_u + r) ** 2))

    flat = params_to_vec(params)
    if use_line_search:
        alpha, loss = line_search(flat, delta, params, X, y, alphas=ls_alphas)
    else:
        alpha = 1.0
    new_params = vec_to_params(flat + alpha * delta, params)
    return new_params, opt_state_new, float(loss), ls_err


def _step_adamw_jit(params, opt_state, opt, X, y):
    loss, grads = jax.value_and_grad(mse_loss)(params, X, y)
    updates, new_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), new_state, loss


# ═══════════════════════════════════════════════════════════════
# 7.  Training loop — returns the same results dict as subspace.py
#     Keys: iter, train_loss, ls_err, val_iter, val_loss,
#           val_metric, step_times, total_time
# ═══════════════════════════════════════════════════════════════

def train(method, key,
          X_train, y_train, X_val, y_val, d,
          n_iters=300, val_every=10,
          k=10, damping=0.0, refresh_every=50,
          adam_lr=1e-2, weight_decay=1e-4,
          batch_size=None,
          use_line_search=True, ls_alphas=_DEFAULT_LS_ALPHAS):
    """
    method : 'full_gn' | 'subspace_gn' | 'hybrid' | 'grad_augmented' | 'adamw'
    batch_size : None → full dataset each step.
    """
    key, init_key, p_key = random.split(key, 3)
    params = init_params(init_key, d)
    M      = params_to_vec(params).shape[0]
    N      = X_train.shape[0]
    BS     = N if batch_size is None else batch_size

    # Build initial P
    if method == 'full_gn':
        P     = jnp.eye(M)
        k_eff = M
    elif method in ('subspace_gn', 'hybrid', 'grad_augmented'):
        A, _  = jnp.linalg.qr(random.normal(p_key, (M, k)))
        P     = A
        k_eff = k
    else:
        P     = None
        k_eff = None

    opt       = optax.adamw(adam_lr, weight_decay=weight_decay)
    opt_state = opt.init(params)

    iter_list, train_list, ls_list = [], [], []
    val_iter_list, val_loss_list   = [], []
    step_time_list                 = []

    step     = 0
    rec_step = 0
    t_start  = None

    def batch_iter():
        while True:
            perm = np.random.permutation(N)
            for start in range(0, N, BS):
                yield X_train[perm[start: start + BS]], y_train[perm[start: start + BS]]

    print(f"  [{method}]  M={M},  k={k_eff},  damping={damping}")
    jax.effects_barrier()
    t_start = time.perf_counter()

    for X_b, y_b in batch_iter():

        t0 = time.perf_counter()

        if method == 'full_gn':
            params, bloss, bls = _gn_step(params, X_b, y_b, P, damping)
            #params, opt_state, bloss, bls = _hybrid_step(params, opt_state, opt, X_b, y_b, P, damping,            use_line_search=use_line_search, ls_alphas=ls_alphas)
        elif method == 'subspace_gn':
            if step > 0 and step % refresh_every == 0:
                key, p_key = random.split(key)
                A, _ = jnp.linalg.qr(random.normal(p_key, (M, k)))
                P = A
            params, bloss, bls = _gn_step(
                params, X_b, y_b, P, damping,
                use_line_search=use_line_search, ls_alphas=ls_alphas)

        elif method == 'hybrid':
            if step > 0 and step % refresh_every == 0:
                key, p_key = random.split(key)
                A, _ = jnp.linalg.qr(random.normal(p_key, (M, k)))
                P = A
            params, opt_state, bloss, bls = _hybrid_step(
                params, opt_state, opt, X_b, y_b, P, damping,
                use_line_search=use_line_search, ls_alphas=ls_alphas)

        elif method == 'grad_augmented':
            # P_rand holds k-1 columns; Adam gradient fills the last slot
            P_rand = P[:, :-1] if P.shape[1] > 1 else P
            if step > 0 and step % refresh_every == 0:
                key, p_key = random.split(key)
                k_rand = max(1, k - 1)
                A, _ = jnp.linalg.qr(random.normal(p_key, (M, k_rand)))
                P_rand = A
                P = jnp.concatenate([P_rand, P_rand[:, :1]], axis=1)  # placeholder
            params, opt_state, bloss, bls = _grad_augmented_step(
                params, opt_state, opt, X_b, y_b, P_rand, damping,
                use_qr=True,
                use_line_search=use_line_search, ls_alphas=ls_alphas)

        else:  # adamw
            params, opt_state, bloss = _step_adamw_jit(
                params, opt_state, opt, X_b, y_b)
            jax.effects_barrier()
            bloss = float(bloss)
            bls   = float('nan')

        step_dur = time.perf_counter() - t0
        step += 1

        iter_list.append(rec_step)
        train_list.append(bloss)
        ls_list.append(bls)
        step_time_list.append(step_dur)
        rec_step += 1

        if rec_step % val_every == 0:
            vl = float(mse_loss(params, X_val, y_val))
            val_iter_list.append(rec_step)
            val_loss_list.append(vl)

        if rec_step >= n_iters:
            break

    total_time = time.perf_counter() - t_start
    avg_ms = 1e3 * float(np.mean(step_time_list))
    print(f"  → {rec_step} iters in {total_time:.1f}s  ({avg_ms:.1f} ms/step)")

    return {
        'iter':       np.array(iter_list),
        'train_loss': np.array(train_list),
        'ls_err':     np.array(ls_list),
        'val_iter':   np.array(val_iter_list),
        'val_loss':   np.array(val_loss_list),
        'val_metric': np.array([]),
        'step_times': np.array(step_time_list),
        'total_time': total_time,
    }


# ═══════════════════════════════════════════════════════════════
# 8.  Plot — identical layout to subspace.py plot_method_comparison
#     3 panels: (A) Train Loss  (B) Val Loss  (C) Inner LS Error
# ═══════════════════════════════════════════════════════════════

STYLES = {
    'full_gn':         dict(color='#d62728', lw=2, ls='-',
                            label='Full GN (P=I)'),
    'pure_subspace':   dict(color='#e07b39', lw=2, ls='-',
                            label='Pure Subspace'),
    'pure_subspace_d': dict(color='#9467bd', lw=2, ls='-.',
                            label='Pure Subspace k=d'),
    'hybrid':          dict(color='#8c564b', lw=2, ls='-',
                            label='Hybrid complement'),
    'grad_augmented':  dict(color='#17becf', lw=2, ls='--',
                            label='Grad-augmented'),
    'adamw':           dict(color='#3a7ebf', lw=2, ls=':',
                            label='Baseline: AdamW'),
}


def _label_with_time(key, res):
    s = dict(STYLES[key])
    t = res.get('total_time', 0.0)
    s['label'] = f"{s['label']}  ({t:.0f}s)"
    return s


def plot_method_comparison(results, n_iters, noise_floor,
                           subtitle="", save_path="full_gn_sanity.png"):
    """
    3 panels — (A) Train Loss, (B) Val Loss, (C) Inner LS Error.
    Identical layout to subspace.py plot_method_comparison.
    """
    fig, (ax_train, ax_val, ax_ls) = plt.subplots(1, 3, figsize=(18, 5))

    # ── (A) Train loss ────────────────────────────────────────
    for k, res in results.items():
        if len(res['iter']) > 0:
            style = _label_with_time(k, res)
            ax_train.plot(res['iter'], res['train_loss'], **style)

    ax_train.set_xlabel('Iteration')
    ax_train.set_ylabel('Training loss')
    ax_train.set_title('(A) Train loss')
    ax_train.set_yscale('log')
    ax_train.set_xlim(0, n_iters)
    ax_train.legend(fontsize=8)
    ax_train.grid(True, alpha=0.3)

    # ── (B) Val loss ──────────────────────────────────────────
    for k, res in results.items():
        if len(res['val_iter']) > 0:
            ax_val.plot(res['val_iter'], res['val_loss'],
                        **_label_with_time(k, res))

    ax_val.set_xlabel('Iteration')
    ax_val.set_ylabel('Validation loss')
    ax_val.set_title('(B) Val loss vs iteration')
    ax_val.set_yscale('log')
    ax_val.set_xlim(0, n_iters)
    ax_val.legend(fontsize=8)
    ax_val.grid(True, alpha=0.3)

    # ── (C) Inner LS error ────────────────────────────────────
    for k, res in results.items():
        if len(res['iter']) > 0 and not np.all(np.isnan(res['ls_err'])):
            style = dict(STYLES[k])
            ax_ls.plot(res['iter'], res['ls_err'], **style)

    ax_ls.set_xlabel('Iteration')
    ax_ls.set_ylabel('Inner LS Error')
    ax_ls.set_title('(C) Inner Optimization Error')
    ax_ls.set_yscale('log')
    ax_ls.set_xlim(0, n_iters)
    ax_ls.legend(fontsize=8)
    ax_ls.grid(True, alpha=0.3)

    plt.suptitle(
        f'GN variants sanity check — quadratic regression\n{subtitle}',
        fontsize=12, y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {save_path}")


# ═══════════════════════════════════════════════════════════════
# 9.  Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    key = random.PRNGKey(42)

    D         = 1000            # input dim  →  M = D+1
    N         = 500         # number of training samples
    NOISE_STD = 0.5
    DAMPING  = 0
    N_ITERS   = 300
    VAL_EVERY = 1
    REFRESH   = 1
    K         = int(0.004 * D)

    key, data_key = random.split(key)
    X_train, y_train, X_val, y_val, w_true, b_true = make_regression(
        data_key, n_train=N, n_val=int(N*0.2), d=D, noise_std=NOISE_STD)

    noise_floor = 0.5 * NOISE_STD ** 2
    print(f"\nDataset: d={D}, M={D+1}, noise_floor={noise_floor:.5f}\n")

    results = {}

    # ── Full GN  (P = I) ──────────────────────────────────────
    print("── Full GN  (λ=0, P=I) ──")
    key, run_key = random.split(key)
    results['full_gn'] = train(
        'full_gn', run_key, X_train, y_train, X_val, y_val, D,
        n_iters=N_ITERS, val_every=VAL_EVERY,
        damping=DAMPING)

    # ── Subspace GN  k=4 ──────────────────────────────────────
    print("\n── Pure Subspace GN  (k=4) ──")
    key, run_key = random.split(key)
    results['pure_subspace'] = train(
        'subspace_gn', run_key, X_train, y_train, X_val, y_val, D,
        n_iters=N_ITERS, val_every=VAL_EVERY,
        k=4, damping=DAMPING, refresh_every=REFRESH)

    # ── Subspace GN  k=D ────────────────────────────────────
    print("\n── Pure Subspace GN  (k=d) ──")
    key, run_key = random.split(key)
    results['pure_subspace_d'] = train(
        'subspace_gn', run_key, X_train, y_train, X_val, y_val, D,
        n_iters=N_ITERS, val_every=VAL_EVERY,
        k = D, damping=DAMPING, refresh_every=REFRESH)

    # ── Hybrid complement  (k=4) ─────────────────────────────
    print("\n── Hybrid complement  (k=4) ──")
    key, run_key = random.split(key)
    results['hybrid'] = train(
        'hybrid', run_key, X_train, y_train, X_val, y_val, D,
        n_iters=N_ITERS, val_every=VAL_EVERY,
        k=K, damping=DAMPING, refresh_every=REFRESH,
        adam_lr=1e-2, weight_decay=1e-4)

    # ── Grad-augmented  (k=4, gradient fills last column) ────
    print("\n── Grad-augmented  (k=4) ──")
    key, run_key = random.split(key)
    results['grad_augmented'] = train(
        'grad_augmented', run_key, X_train, y_train, X_val, y_val, D,
        n_iters=N_ITERS, val_every=VAL_EVERY,
        k=K, damping=DAMPING, refresh_every=REFRESH,
        adam_lr=1e-2, weight_decay=1e-4)

    # ── AdamW baseline ────────────────────────────────────────
    print("\n── AdamW baseline ──")
    key, run_key = random.split(key)
    results['adamw'] = train(
        'adamw', run_key, X_train, y_train, X_val, y_val, D,
        n_iters=N_ITERS, val_every=VAL_EVERY,
        adam_lr=1e-2, weight_decay=1e-4, batch_size=128)

    # ── Summary table ─────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  {'Method':<26} {'Final train':>12} {'Final val':>12} {'Time':>8}")
    print('─'*64)
    for key_m, res in results.items():
        label = STYLES[key_m]['label']
        ft = float(res['train_loss'][-1]) if len(res['train_loss']) else float('nan')
        fv = float(res['val_loss'][-1])   if len(res['val_loss'])   else float('nan')
        t  = res['total_time']
        print(f"  {label:<26} {ft:>12.6f} {fv:>12.6f} {t:>7.1f}s")
    print(f"  {'Noise floor':<26} {'':>12} {noise_floor:>12.6f}")
    print('─'*64)

    plot_method_comparison(
        results, n_iters=N_ITERS, noise_floor=noise_floor,
        subtitle=f'Linear model, d={D}, n={N}, k={K}, noise_std={NOISE_STD}',
        save_path='full_gn_sanity.png')