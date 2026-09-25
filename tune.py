import json
import os
import optuna

from train import train
from config import OPTUNA_STORAGE, OPTUNA_STUDY_NAME, BEST_PARAMS_PATH

N_TRIALS = 30
NUM_EPOCHS_PER_TRIAL = 5


def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    optimizer_name = trial.suggest_categorical('optimizer_name', ['adam', 'adamw', 'sgd'])
    weight_decay = trial.suggest_float('weight_decay', 0.0, 1e-3)

    valid_acc = train(
        num_epochs=NUM_EPOCHS_PER_TRIAL,
        batch_size=batch_size,
        lr=lr,
        optimizer_name=optimizer_name,
        weight_decay=weight_decay,
        save_checkpoint=False,
        save_losses=False,
        verbose=False,
    )
    return valid_acc


def run_search(n_trials=N_TRIALS):
    os.makedirs(os.path.dirname(BEST_PARAMS_PATH), exist_ok=True)

    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        storage=OPTUNA_STORAGE,
        direction='maximize',
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=n_trials)

    print(f"\nBest trial:")
    print(f"  valid_accuracy: {study.best_value:.4f}")
    print(f"  params: {study.best_params}")

    with open(BEST_PARAMS_PATH, 'w') as f:
        json.dump({
            'valid_accuracy': study.best_value,
            'params': study.best_params,
        }, f, indent=2)

    print(f"\nSaved best params to {BEST_PARAMS_PATH}")
    return study


if __name__ == "__main__":
    run_search(n_trials=10)