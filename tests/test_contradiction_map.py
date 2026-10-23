import numpy as np

from detector.contradiction_map import contradiction_map

FPS = 25.0


def speech_env(n, seed=0):
    """Syllable-like bursts with pauses."""
    rng = np.random.default_rng(seed)
    env = np.zeros(n)
    i = 0
    while i < n:
        on = int(rng.integers(4, 10))
        env[i:i + on] = rng.uniform(0.5, 1.0) * (0.6 + 0.4 * np.sin(np.linspace(0, np.pi, min(on, n - i))))
        i += on + int(rng.integers(2, 6))
    return env[:n] + 0.02 * rng.random(n)


def mouth_from(env, delay=0, noise=0.05, seed=1):
    rng = np.random.default_rng(seed)
    m = np.roll(env, delay) * 1.0
    return m + rng.normal(0, noise, len(m))


def test_localizes_a_swapped_segment():
    n = int(15 * FPS)
    env = speech_env(n)
    mouth = mouth_from(env, delay=2)
    # seconds 5-8: mouth follows an unrelated signal (e.g. inserted synthetic speech)
    mouth[int(5 * FPS):int(10 * FPS)] = speech_env(int(5 * FPS), seed=99)
    out = contradiction_map(mouth, env, FPS)
    assert out["status"] == "ok" and out["intervals"]
    iv = out["intervals"][0]
    assert 3.0 <= iv["t0"] <= 7.0 and 8.0 <= iv["t1"] <= 11.5
    flagged = [w for w in out["windows"] if w["status"] == "contradiction"]
    assert all(w["t1"] > 4.0 and w["t0"] < 10.5 for w in flagged)


def test_benign_constant_dubbing_latency_is_not_a_contradiction():
    n = int(12 * FPS)
    env = speech_env(n)
    mouth = mouth_from(env, delay=int(0.24 * FPS))      # 240 ms offset, whole clip
    out = contradiction_map(mouth, env, FPS)
    assert out["status"] == "ok" and out["intervals"] == []
    assert abs(out["global_lag_s"]) > 0.15 and out["benign_latency"]


def test_missing_modalities_and_uninformative_windows():
    assert contradiction_map(None, np.ones(50), FPS)["status"] == "no_face"
    assert contradiction_map(np.ones(50), None, FPS)["status"] == "no_audio"
    assert contradiction_map(np.ones(5), np.ones(5), FPS)["status"] == "too_short"
    n = int(10 * FPS)
    env = np.zeros(n) + 1e-4   # silence: nothing to compare
    mouth = np.random.default_rng(0).random(n)
    out = contradiction_map(mouth, env, FPS)
    assert out["intervals"] == [] and out["coverage"] == 0.0
    assert all(w["status"] == "uninformative" for w in out["windows"])
    # face lost half the time -> those windows are uninformative, not contradictions
    env = speech_env(n); mouth = mouth_from(env)
    mouth[: n // 2] = np.nan
    out = contradiction_map(mouth, env, FPS)
    assert out["intervals"] == []
    assert any(w.get("why") == "face lost" for w in out["windows"])


def test_genuine_clip_has_no_false_alarm():
    fa = 0
    for seed in range(8):
        n = int(12 * FPS)
        env = speech_env(n, seed)
        out = contradiction_map(mouth_from(env, delay=1, seed=seed + 50), env, FPS)
        fa += bool(out["intervals"])
    assert fa <= 1
