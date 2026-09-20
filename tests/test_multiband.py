import numpy as np
from sklearn.preprocessing import StandardScaler

from wdm_cnn.preprocessing.multiband import make_3channel_scaler_input
from wdm_cnn.preprocessing.scalers import ArcsinhScaler, ShiftedLogScaler


def _sample_data(n=5, h=8, w=8, seed=0):
    rng = np.random.default_rng(seed)
    # Positive-valued, plausible flux-like data
    return rng.uniform(0.01, 10.0, size=(n, h, w)).astype(np.float32)


def _fitted_scalers():
    scaler_shl = ShiftedLogScaler(shift=0.1)
    scaler_arcsinh = ArcsinhScaler(soft=0.1)
    scaler_std = StandardScaler()
    scaler_std.fit(_sample_data().reshape(-1, 1))
    return scaler_shl, scaler_arcsinh, scaler_std


def test_output_shape():
    data = _sample_data(n=5, h=8, w=8)
    scaler_shl, scaler_arcsinh, scaler_std = _fitted_scalers()
    out = make_3channel_scaler_input(data, scaler_shl, scaler_arcsinh, scaler_std)
    assert out.shape == (5, 3, 8, 8)
    assert out.dtype == np.float32


def test_each_image_channel_is_normalized():
    data = _sample_data(n=4, h=6, w=6)
    scaler_shl, scaler_arcsinh, scaler_std = _fitted_scalers()
    out = make_3channel_scaler_input(data, scaler_shl, scaler_arcsinh, scaler_std)

    means = out.mean(axis=(2, 3))
    stds = out.std(axis=(2, 3))
    np.testing.assert_allclose(means, 0.0, atol=1e-5)
    np.testing.assert_allclose(stds, 1.0, atol=1e-3)


def test_chunking_gives_identical_result():
    """Chunked and unchunked calls should be mathematically identical,
    since per-image normalization is independent across the batch."""
    data = _sample_data(n=10, h=6, w=6)
    scaler_shl, scaler_arcsinh, scaler_std = _fitted_scalers()

    out_unchunked = make_3channel_scaler_input(
        data, scaler_shl, scaler_arcsinh, scaler_std, chunk_size=None
    )
    out_chunked = make_3channel_scaler_input(
        data, scaler_shl, scaler_arcsinh, scaler_std, chunk_size=3
    )
    np.testing.assert_allclose(out_unchunked, out_chunked, atol=1e-6)