from scripts.evaluate_emotion_ablation import compare_runs


def test_compare_runs_returns_metrics_per_run():
    annotations = [
        {"turn_count": 1, "expected": "anxious"},
        {"turn_count": 2, "expected": "grateful"},
    ]
    runs = {
        "baseline": [
            {"turn_count": 1, "emotion": "sad", "success": True},
            {"turn_count": 2, "emotion": "grateful", "success": True},
        ],
        "dynamic-eicl": [
            {"turn_count": 1, "emotion": "anxious", "success": True},
            {"turn_count": 2, "emotion": "grateful", "success": True},
        ],
    }

    result = compare_runs(runs, annotations)

    assert result["baseline"]["accuracy"] == 0.5
    assert result["dynamic-eicl"]["accuracy"] == 1.0
    assert "macro_f1" in result["dynamic-eicl"]
