import numpy as np

from img2svg import verify


def test_identical_images_score_zero():
    a = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    st = verify.compare(a, a.copy())
    assert st["mean"] == 0.0
    assert st["max"] == 0.0


def test_delta_e2000_matches_the_sharma_reference_pairs():
    # from Sharma, Wu & Dalal (2005), the standard CIEDE2000 test data
    cases = [
        ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
        ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
        ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
        ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
        ((50.0000, 2.5000, 0.0000), (50.0000, 0.0000, -2.5000), 4.3065),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ]
    for lab1, lab2, want in cases:
        got = float(verify.delta_e2000(np.array([lab1]), np.array([lab2]))[0])
        assert abs(got - want) < 1e-3, f"{lab1} vs {lab2}: {got} != {want}"


def test_delta_e2000_is_symmetric():
    rng = np.random.default_rng(3)
    a = verify.srgb_to_lab(rng.integers(0, 255, (64, 3), dtype=np.uint8))
    b = verify.srgb_to_lab(rng.integers(0, 255, (64, 3), dtype=np.uint8))
    assert np.allclose(verify.delta_e2000(a, b), verify.delta_e2000(b, a), atol=1e-9)


def test_white_is_lab_100():
    lab = verify.srgb_to_lab(np.array([[255, 255, 255]], dtype=np.uint8))
    assert abs(lab[0, 0] - 100.0) < 1e-3
    assert abs(lab[0, 1]) < 1e-2 and abs(lab[0, 2]) < 1e-2
