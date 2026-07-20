"""Non-interactive end-to-end check of the ML pipeline.

`main.py` is an interactive CLI and cannot run in CI (it waits on stdin).
This script exercises the same pipeline functions from start to finish so the
CI workflow can verify that the whole chain still works on every push.
"""

from function import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main() -> None:
    X_train, X_test, y_train, y_test, scaler = prepare_data()

    for model_name in ("rf", "ada", "xgb"):
        model = train_model(model_name, X_train, y_train)
        accuracy, _, _ = evaluate_model(model, X_test, y_test)
        assert 0.0 <= accuracy <= 1.0, f"invalid accuracy for {model_name}"

    # Persistence round-trip: the reloaded model must predict like the original.
    save_model(model, scaler)
    loaded_model, loaded_scaler = load_model()
    assert (loaded_model.predict(X_test) == model.predict(X_test)).all(), (
        "reloaded model does not reproduce the original predictions"
    )
    assert loaded_scaler is not None

    print("Smoke test passed: prepare -> train -> evaluate -> save -> load")


if __name__ == "__main__":
    main()
