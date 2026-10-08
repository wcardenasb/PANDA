import os
from keras_tuner import BayesianOptimization, Objective
from sklearn.metrics import roc_auc_score
from panda.training import train_em
from experiments.data_loader import load_dataset

max_layers = 3

class MyTuner(BayesianOptimization):

    def __init__(self, train_data, val_data, input_dim, K, T, eta_type, **kwargs):
        super().__init__(**kwargs)
        self.train_data = train_data
        self.val_data = val_data
        self.input_dim = input_dim
        self.K = K
        self.T = T
        self.eta_type = eta_type

    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters
        n_layers = hp.Int("n_layers", 1, max_layers)
        all_units = [hp.Int(f"units_{i}", 10, 600, step=1) for i in range(max_layers)]
        nn_list = all_units[:n_layers]
        batch_size = hp.Int("batch_size", 8, 512, step=8)
        lr = hp.Choice("Learning_rate", [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5])
        d = hp.Int("d", 8, 128)

        model, acc, auc, f1, history, val_history = train_em(
            train=self.train_data, val=self.val_data, input_dim=self.input_dim, d=d,
            K=self.K, T=self.T, em_iters=25, m_steps=20, batch_size=batch_size,
            lr=lr, weight_decay=-1e-6, eta_type=self.eta_type,
            n_layers=n_layers, nn_list=nn_list
        )

        x_val, Z_val = self.val_data[0], self.val_data[3]
        Pi_val, _ = model.predict_proba(x_val)
        y_true = Z_val.numpy().argmax(axis=1)

        auc2 = roc_auc_score(y_true, Pi_val[:, 1]) if self.K == 2 else roc_auc_score(y_true, Pi_val, multi_class="ovr")
        print("AUC_2:", auc2)

        save_path = os.path.join(self.project_dir, trial.trial_id)
        os.makedirs(save_path, exist_ok=True)
        model.save(os.path.join(save_path, "model.keras"))

        self.oracle.update_trial(trial.trial_id, {"auc": auc})


def run_tuning(dataset_name, base_path, eta_type="full", max_trials=100, seed=42, p_errors=0.3):

    model_name = {"symmetric": "sym", "diagonal": "diag", "full": "full"}[eta_type]
    results_path = os.path.join(os.path.dirname(base_path), "..", "..", "results", "tuning", model_name)
    datasets = load_dataset(dataset_name, base_path, seed=seed, p_errors=p_errors)

    for data in datasets:
        dataset = data["name"] or dataset_name
        output_dir = os.path.join(results_path, dataset)

        tuner = MyTuner(
            train_data=data["train_data"], val_data=data["val_data"],
            input_dim=data["input_dim"], K=data["K"], T=data["T"],
            eta_type=eta_type, objective=Objective("auc", direction="max"),
            max_trials=max_trials, executions_per_trial=1,
            overwrite=True, directory=output_dir, project_name="tuning_results"
        )

        tuner.search()