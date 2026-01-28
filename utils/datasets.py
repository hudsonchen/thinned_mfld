from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import jax
import jax.numpy as jnp

def get_data(q1_nn_apply,d, M, data, shape, rng_key):
    key_z, key_noise = jax.random.split(rng_key, 2)
    key_params = jax.random.PRNGKey(42)
    # Sample inputs
    d_input = d
    Z = jax.random.normal(key_z, shape=(*shape, d_input), dtype=jnp.float32)
    Z_all = Z.reshape((-1, d_input))
    teacher_params = sample_teacher_params_multimodal_per_neuron(
        key_params,
        d,
        M,
        num_modes=(M // 10)+1,
        mode_separation=2.0,
        within_mode_std=0.2,
        base_scale=0.8,
    )
    y_clean_all = jax.vmap(jax.vmap(q1_nn_apply, in_axes=(0, None)),
                    in_axes=(None, 0)
                )(Z_all, teacher_params).mean(0)
    y_clean = y_clean_all.reshape((*shape, -1))
    eps = 0.3 * jax.random.normal(key_noise, shape=y_clean.shape, dtype=jnp.float32)
    y = y_clean + eps

    z_mean, z_std = data["z_stats"]
    Z = (Z - z_mean) / z_std
    return (Z, y)


def load_student_teacher(batch_size, total_size, q1_nn_apply, d, M, standardize_Z=True, standardize_y=False):
    """
    Synthetic student-teacher regression dataset, batched like your load_boston().

    Returns dict with:
      Z, y:       (num_batches_tr, batch_size, d_in), (num_batches_tr, batch_size, 1)
      Z_test,y_test similarly
      z_stats: (mean, std) computed from train split (like your code)
      y_stats: (min, max) from train split (for optional [-1,1] scaling)

    Teacher:
      z ~ N(0, I)
      y = q1_nn_apply(z, (W1,b1,W2)) + noise,  noise ~ N(0, noise_std^2)
    """
    # RNG
    key = jax.random.PRNGKey(42)
    key_z, key_perm, key_noise, key_params = jax.random.split(key, 4)

    # Sample inputs
    d_input = d
    Z = jax.random.normal(key_z, shape=(total_size, d_input), dtype=jnp.float32)
    # Z = Z / jnp.linalg.norm(Z, axis=1, keepdims=True)

    # Sample teacher parameters
    # W1 = 0.8 * jax.random.normal(key_params, shape=(d_input, M), dtype=jnp.float32)
    # b1 = jnp.zeros((M,), dtype=jnp.float32)
    # W2 = 0.8 * jax.random.normal(jax.random.fold_in(key_params, 1), shape=(M,), dtype=jnp.float32)
    # teacher_params = jnp.concatenate([W1.T, b1[:, None], W2[:, None]], axis=1)

    teacher_params = sample_teacher_params_multimodal_per_neuron(
        key_params,
        d_input,
        M,
        num_modes=(M // 10)+1,
        mode_separation=2.0,
        within_mode_std=0.2,
        base_scale=0.8,
    )
    
    # Generate targets
    y_clean = jax.vmap(jax.vmap(q1_nn_apply, in_axes=(0, None)),
                    in_axes=(None, 0)
                )(Z, teacher_params).mean(0)
    eps = 0.3 * jax.random.normal(key_noise, shape=y_clean.shape, dtype=jnp.float32)
    y = y_clean + eps

    # Train/test split (numpy-style, deterministic)
    N_te = int(np.round(0.1 * total_size))
    N_tr = total_size - N_te
    perm = jax.random.permutation(key_perm, total_size)
    Z = Z[perm]
    y = y[perm]
    Z_tr, Z_te = Z[:N_tr], Z[N_tr:]
    y_tr, y_te = y[:N_tr], y[N_tr:]

    # Standardize using train stats
    z_mean = jnp.mean(Z_tr, axis=0, keepdims=True)
    z_std = jnp.std(Z_tr, axis=0, keepdims=True) + 1e-8
    if standardize_Z:
        Z_tr = (Z_tr - z_mean) / z_std
        Z_te = (Z_te - z_mean) / z_std

    y_min = jnp.min(y_tr, axis=0, keepdims=True)
    y_max = jnp.max(y_tr, axis=0, keepdims=True)
    if standardize_y:
        y_tr = 2 * (y_tr - y_min) / (y_max - y_min + 1e-8) - 1
        y_te = 2 * (y_te - y_min) / (y_max - y_min + 1e-8) - 1

    # Batch + pad (edge) like your load_boston
    def batchify(Z_, y_):
        N_ = y_.shape[0]
        num_batches = (N_ + batch_size - 1) // batch_size
        pad = num_batches * batch_size - N_
        if pad > 0:
            Z_ = jnp.pad(Z_, ((0, pad), (0, 0)), mode="edge")
            y_ = jnp.pad(y_, ((0, pad), (0, 0)), mode="edge")
        Z_ = Z_.reshape(num_batches, batch_size, Z_.shape[-1])
        y_ = y_.reshape(num_batches, batch_size, 1)
        return Z_, y_, num_batches

    Z_tr_b, y_tr_b, num_batches_tr = batchify(Z_tr, y_tr)
    Z_te_b, y_te_b, num_batches_te = batchify(Z_te, y_te)

    out = {
        "Z": Z_tr_b,
        "y": y_tr_b,
        "Z_test": Z_te_b,
        "y_test": y_te_b,
        "z_stats": (jnp.asarray(z_mean.squeeze(0)), jnp.asarray(z_std.squeeze(0))),
        "y_stats": (jnp.asarray(y_min.item()), jnp.asarray(y_max.item())),
        "batch_size": batch_size,
        "num_batches_tr": num_batches_tr,
        "num_batches_te": num_batches_te,
        "d_in": d_input,
        "noise_std": 0.3,
        "teacher_params": teacher_params,
    }

    return out


def sample_teacher_params_multimodal_per_neuron(
    key,
    d_input: int,
    M: int,
    num_modes: int = 8,
    mode_separation: float = 2.0,
    within_mode_std: float = 0.2,
    base_scale: float = 0.8,
):
    key_centers, key_assign, key_eps = jax.random.split(key, 3)
    Drow = d_input + 2

    centers = mode_separation * jax.random.normal(key_centers, (num_modes, Drow), dtype=jnp.float32)
    # assign a mode to each hidden unit
    idx = jax.random.randint(key_assign, (M,), 0, num_modes)
    mu_rows = centers[idx]  # (M, Drow)

    eps = within_mode_std * jax.random.normal(key_eps, (M, Drow), dtype=jnp.float32)
    teacher_params = mu_rows + eps  # (M, d_input+2)

    # scale W1/W2 similarly
    W1 = teacher_params[:, :d_input] * (base_scale)
    b1 = teacher_params[:, d_input:d_input+1]
    W2 = teacher_params[:, d_input+1:d_input+2] * (base_scale)
    return jnp.concatenate([W1, b1, W2], axis=1)
