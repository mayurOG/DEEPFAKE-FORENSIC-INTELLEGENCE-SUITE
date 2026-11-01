import numpy as np
from PIL import Image

from utils.comparator import (
    compute_hist_correlation,
    compute_image_similarity,
    compute_psnr,
    compute_ssim,
    generate_difference_heatmap,
    plot_3d_difference_surface,
    plot_3d_fft_difference,
    plot_expert_logits_comparison,
    plot_forensic_radar_comparison,
    plot_similarity_gauge,
)


def test_similarity_identical_images():
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    im = Image.fromarray(arr)

    sim = compute_image_similarity(im, im)
    assert sim["ssim"] >= 99.0
    assert sim["overall_similarity"] >= 95.0
    assert sim["badge"] == "MATCH"
    assert sim["psnr_db"] >= 90.0


def test_similarity_modified_images():
    rng = np.random.default_rng(42)
    arr_a = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    arr_b = arr_a.copy()
    # Modify a central region (simulating face swap)
    arr_b[30:80, 30:80] = 0

    im_a = Image.fromarray(arr_a)
    im_b = Image.fromarray(arr_b)

    sim = compute_image_similarity(im_a, im_b)
    assert sim["overall_similarity"] < 95.0
    assert sim["ssim"] < 100.0


def test_difference_heatmap():
    im_a = Image.new("RGB", (64, 64), color=(100, 100, 100))
    im_b = Image.new("RGB", (64, 64), color=(200, 200, 200))
    diff_map = generate_difference_heatmap(im_a, im_b)
    assert diff_map.size == (64, 64)
    assert isinstance(diff_map, Image.Image)


def test_plotly_comparator_figures():
    im_a = Image.new("RGB", (64, 64), color=(100, 100, 100))
    im_b = Image.new("RGB", (64, 64), color=(120, 120, 120))

    fig_surf = plot_3d_difference_surface(im_a, im_b)
    assert len(fig_surf.data) > 0

    fig_fft = plot_3d_fft_difference(im_a, im_b)
    assert len(fig_fft.data) > 0

    fig_gauge = plot_similarity_gauge(85.5, "MODIFIED")
    assert len(fig_gauge.data) > 0

    res_a = {"p_fake": 0.05, "quality": {"srm_noise_energy": 5.0}, "evidence": [{"name": "neural", "logit": -3.0, "status": "used"}]}
    res_b = {"p_fake": 0.92, "quality": {"srm_noise_energy": 25.0}, "evidence": [{"name": "neural", "logit": 4.2, "status": "used"}]}

    fig_radar = plot_forensic_radar_comparison(res_a, res_b)
    assert len(fig_radar.data) == 2

    fig_bar = plot_expert_logits_comparison(res_a, res_b)
    assert len(fig_bar.data) == 2
