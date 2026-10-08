import os
import json
import numpy as np


results_path = r'C:\Users\Usuario UTP\Documents\PANDA\results'
tuning_path = os.path.join(results_path, 'tuning')

models = {
    'sym': os.path.join(tuning_path, 'sym'),
    'diag': os.path.join(tuning_path, 'diag'),
    'full': os.path.join(tuning_path, 'full')
}

resumen_global = {}

for model, root_dir in models.items():

    if not os.path.isdir(root_dir):
        continue

    for db in os.listdir(root_dir):
        db_path = os.path.join(root_dir, db, 'tuning_results')

        if not os.path.isdir(db_path):
            continue

        best_auc, best_trial_id, best_hyperparams = -np.inf, None, None

        for trial in os.listdir(db_path):
            if not trial.startswith('trial_'):
                continue

            trial_json = os.path.join(db_path, trial, 'trial.json')

            if not os.path.exists(trial_json):
                continue

            with open(trial_json, 'r') as f:
                trial_data = json.load(f)

            score = trial_data.get('score')

            if score is not None and score > best_auc:
                best_auc = score
                best_trial_id = trial.split('_')[1]
                best_hyperparams = trial_data.get('hyperparameters', {}).get('values', {})

        if best_hyperparams is not None:
            resumen_global.setdefault(db, {})[model] = {
                'best_score': float(best_auc),
                'metric': 'AUC',
                'trial_id': best_trial_id,
                'hyperparameters': best_hyperparams
            }


output_path = os.path.join(results_path, 'resumen_global.json')

with open(output_path, 'w') as f:
    json.dump(resumen_global, f, indent=4)

print(f'✅ Resumen global guardado en: {output_path}')