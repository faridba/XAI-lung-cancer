#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from mrmr import mrmr_classif  # The library for mRMR
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from skorch import NeuralNetBinaryClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, 
                             average_precision_score, matthews_corrcoef, accuracy_score, f1_score,
                             cohen_kappa_score, jaccard_score, roc_auc_score)
import os
import joblib


# In[2]:


X_GSE89843, y_GSE89843 = pd.read_csv('C:/Users/farid/xai paper/GSE89843/Data/GSE89843_Normalized_MaxMean_Symbols.csv', index_col=0).T, pd.read_csv('C:/Users/farid/xai paper/GSE89843/Data/GSE89843_cancer_type.csv', index_col=0)
X_GSE207586, y_GSE207586 = pd.read_csv('C:/Users/farid/xai paper/GSE207586/Data/GSE207586_Normalized_MaxMean_Symbols.csv', index_col=0).T, pd.read_csv('C:/Users/farid/xai paper/GSE207586/Data/GSE207586_cancer_type.csv', index_col=0)

X_GSE89843.index = X_GSE89843.index.str.replace('.', '-').str.replace('HC', 'Healthy Control')
y_GSE89843.index = y_GSE89843['subject_id']

common_genes = sorted(list(set(X_GSE89843.columns) & set(X_GSE207586.columns)))
common_subjects = sorted(list(set(y_GSE89843['subject_id'].values) & set(np.array(X_GSE89843.index.tolist()))))

X_GSE89843 = X_GSE89843.loc[common_subjects, common_genes]
y_GSE89843 = y_GSE89843.loc[common_subjects]
# List of values you want to keep
keep_list = ['Healthy Control', 'NSCLC']

# Filter the DataFrame
# Replace 'Group' with the actual name of your column
y_GSE89843 = y_GSE89843[y_GSE89843['cancer_type'].isin(keep_list)]
X_GSE89843 = X_GSE89843.loc[y_GSE89843.index]
y_GSE89843['cancer_type'] = y_GSE89843['cancer_type'].replace('Healthy Control', 'Control')

a = []
for i in X_GSE207586.index.str.split('.'):
    a.append('-'.join(i[:-1]))
X_GSE207586.index = a
y_GSE207586.index = y_GSE207586['subject_id']

y_GSE207586 = y_GSE207586[~y_GSE207586.index.isin(y_GSE89843.index)]

X_GSE207586 = X_GSE207586.loc[y_GSE207586.index, common_genes]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device}")
# --- 1. Setup & Definitions ---

# Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device}")

# Helper: Calculate Metrics
def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1 score': f1_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'ROC_AUC': roc_auc_score(y_true, y_prob),
        'PR_AUC': average_precision_score(y_true, y_prob),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Kappa': cohen_kappa_score(y_true, y_pred),
        'Jaccard': jaccard_score(y_true, y_pred),

    }

# Helper: PyTorch Classes
class PyTorchDNN(nn.Module):
    def __init__(self, input_dim, n_layers=1, n_nodes=64, dropout_rate=0.3, use_bn=True):
        super(PyTorchDNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, n_nodes))
        if use_bn: layers.append(nn.BatchNorm1d(n_nodes))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_nodes, n_nodes))
            if use_bn: layers.append(nn.BatchNorm1d(n_nodes))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(n_nodes, 1))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class PyTorchCNN(nn.Module):
    def __init__(self, input_dim, n_layers=1, n_filters=32, kernel_size=3, dropout_rate=0.3, use_bn=True):
        super(PyTorchCNN, self).__init__()
        self.layers = nn.ModuleList()
        # Layer 1
        self.layers.append(nn.Conv1d(1, n_filters, kernel_size, padding=1))
        if use_bn: self.layers.append(nn.BatchNorm1d(n_filters))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool1d(2))
        self.layers.append(nn.Dropout(dropout_rate))
        # Extra Layers
        for _ in range(n_layers - 1):
            self.layers.append(nn.Conv1d(n_filters, n_filters, kernel_size, padding=1))
            if use_bn: self.layers.append(nn.BatchNorm1d(n_filters))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool1d(2))
            self.layers.append(nn.Dropout(dropout_rate))
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(n_filters, 1)
    def forward(self, x):
        for layer in self.layers: x = layer(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)

# Transformers
class ReshapeForPyTorchCNN(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X.astype(np.float32).reshape(X.shape[0], 1, X.shape[1])

class Float32Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X.astype(np.float32)

# --- 2. Model Configuration ---
models_config = {
    'SVM_Linear': {'model': SVC(kernel='linear', probability=True), 'params': {'model__C': [0.1, 1, 10]}},
    'SVM_RBF': {'model': SVC(kernel='rbf', probability=True), 'params': {'model__C': [1, 10], 'model__gamma': ['scale']}},
    'RF': {'model': RandomForestClassifier(), 'params': {'model__n_estimators': [100, 300], 'model__max_depth': [10, 20]}},
    'ExtraTrees': {'model': ExtraTreesClassifier(), 'params': {'model__n_estimators': [100, 300], 'model__max_depth': [10, 20]}},
    'XGBoost': {'model': XGBClassifier(eval_metric='logloss'), 'params': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1]}},
    'LogReg': {'model': LogisticRegression(max_iter=1000), 'params': {'model__C': [0.1, 1, 10]}},

    'DNN': {
        'model': NeuralNetBinaryClassifier(
            module=PyTorchDNN, criterion=nn.BCEWithLogitsLoss, optimizer=torch.optim.Adam,
            verbose=0, device=device
        ),
        'params': {
            'model__module__n_layers': [1, 2], 'model__module__n_nodes': [64],
            'model__module__dropout_rate': [0.3], 'model__module__use_bn': [True],
            'model__max_epochs': [30], 'model__batch_size': [32], 'model__lr': [0.001]
        }
    },
    'CNN': {
        'is_cnn': True,
        'model': NeuralNetBinaryClassifier(
            module=PyTorchCNN, criterion=nn.BCEWithLogitsLoss, optimizer=torch.optim.Adam,
            verbose=0, device=device
        ),
        'params': {
            'model__module__n_layers': [1, 2], 'model__module__n_filters': [32],
            'model__module__dropout_rate': [0.3], 'model__module__use_bn': [True],
            'model__max_epochs': [30], 'model__batch_size': [32], 'model__lr': [0.001]
        }
    }
}

# --- 3. Main Loop with mRMR ---
path_save = 'C:/Users/farid/xai paper'
path_save_fs = os.path.join(path_save, 'saved_features')
path_models_fs = os.path.join(path_save, 'saved_models')
os.makedirs(path_save_fs, exist_ok=True)
os.makedirs(path_models_fs, exist_ok=True)

results_list = []
predictions_store = {}
# Define K values
K_values = [20, 50, 100, 200, 300, 400]
mapping = {'NSCLC': 1, 'Control': 0}
# Ensure Datasets are DataFrames (mRMR needs column names)
# Assuming X_GSE89843 and X_GSE207586 are pandas DataFrames with gene names as columns
y_GSE89843['cancer_type'] = y_GSE89843['cancer_type'].map(mapping)
y_GSE207586['cancer_type'] = y_GSE207586['cancer_type'].map(mapping)

y_GSE89843 = y_GSE89843['cancer_type'].astype(np.float32)
y_GSE207586 = y_GSE207586['cancer_type'].astype(np.float32)

datasets = [
    ('GSE89843_to_GSE207586', X_GSE89843, y_GSE89843, X_GSE207586, y_GSE207586),
    ('GSE207586_to_GSE89843', X_GSE207586, y_GSE207586, X_GSE89843, y_GSE89843)
]

for direction, X_train_full, y_train, X_test_full, y_test in datasets:

    # --- 1. Loop over Feature Counts (K) ---
    for K in K_values:
        print(f"\n--- Processing {direction} | K={K} ---")

        # --- A. Check/Load mRMR Features ---
        feature_filename = os.path.join(path_save_fs, f"mrmr_features_{direction}_K{K}.pkl")


        if os.path.exists(feature_filename):
            print(f"   [Load] Loading saved features from {feature_filename}")
            selected_genes = joblib.load(feature_filename)
        else:
            print(f"   [Compute] Running mRMR selection (K={K})...")
            try:
                selected_genes = mrmr_classif(X=X_train_full, y=y_train, K=K, show_progress=False)
                joblib.dump(selected_genes, feature_filename)
                print(f"   [Save] Features saved to {feature_filename}")
            except Exception as e:
                print(f"   [Error] mRMR failed: {e}")
                continue

        # Subset data using the (loaded or computed) genes
        try:
            X_train_k = X_train_full[selected_genes]
            X_test_k = X_test_full[selected_genes]
        except KeyError as e:
            print(f"   [Error] Feature mismatch: {e}. Delete 'saved_features' folder and retry.")
            continue

        # --- 2. Loop over SMOTE ---
        for smote_name, smote_sampler in [('No_SMOTE', None), ('With_SMOTE', SMOTE(random_state=42))]:

            # --- 3. Loop over Models ---
            for name, config in models_config.items():

                # Define Model Filename
                model_filename = os.path.join(path_models_fs, smote_name, name, str(f'K{K}'), f"model_{direction}.pkl")
                os.makedirs(os.path.join(path_models_fs, smote_name, name, str(f'K{K}')), exist_ok=True)

                # --- B. Check/Load Model ---
                if os.path.exists(model_filename):
                    print(f"   [Load] Loading trained model: {name} ({smote_name})...")
                    try:
                        grid = joblib.load(model_filename)
                        # We don't need to fit, but we need the best estimator for prediction
                    except Exception as e:
                        print(f"   [Error] Failed to load {model_filename}: {e}. Retraining...")
                        # If load fails, force file deletion to trigger training below
                        if os.path.exists(model_filename): os.remove(model_filename)

                # --- C. Train Model (If not loaded) ---
                if not os.path.exists(model_filename):
                    print(f"   [Train] Training {name} ({smote_name})...")

                    # Build Pipeline
                    steps = [('scaler', StandardScaler())]
                    if smote_sampler: steps.append(('smote', smote_sampler))

                    if name == 'CNN': steps.append(('reshaper', ReshapeForPyTorchCNN()))
                    elif name == 'DNN': steps.append(('floater', Float32Transformer()))

                    steps.append(('model', config['model']))
                    pipeline = ImbPipeline(steps)

                    # Dynamic Input Injection
                    current_params = config['params'].copy()
                    if name in ['DNN', 'CNN']:
                        current_params['model__module__input_dim'] = [K] # Use K features

                    # Grid Search
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    grid = GridSearchCV(pipeline, current_params, cv=cv, scoring='average_precision', n_jobs=1)

                    try:
                        grid.fit(X_train_k, y_train)
                        # Save the Fitted Grid Object
                        joblib.dump(grid, model_filename)
                        print(f"   [Save] Model saved to {model_filename}")
                    except Exception as e:
                        print(f"   [Fail] Training failed on {name}: {str(e)}")
                        continue

                # --- D. Predict & Evaluate (Whether Loaded or New) ---
                try:
                    # 'grid' is now populated either from load or fit
                    y_pred = grid.predict(X_test_k)

                    if hasattr(grid.best_estimator_, "predict_proba"):
                        y_prob = grid.best_estimator_.predict_proba(X_test_k)[:, 1]
                    else:
                        y_prob = y_pred

                    mets = calculate_metrics(y_test, y_pred, y_prob)

                    row = {
                        'Direction': direction,
                        'Num_Features_K': K,
                        'SMOTE': smote_name,
                        'Model': name,
                        'Best_Params': grid.best_params_
                    }
                    row.update(mets)
                    results_list.append(row)

                    # Store for stats
                    pred_key = f"{direction}_{K}_{name}_{smote_name}"
                    predictions_store[pred_key] = y_pred

                except Exception as e:
                    print(f"   [Error] Prediction failed for {name}: {e}")

# Save Final Results
results_df = pd.DataFrame(results_list)
print("Workflow Complete. Results dataframe created.")


# In[4]:


import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

def perform_statistical_analysis_by_K(results_df, predictions_store, truth_dict):
    """
    1. Runs Wilcoxon (SMOTE vs No SMOTE) for EACH K independently.
    2. Runs McNemar (Model vs Model) for EACH K independently.
    """

    # Get all unique K values sorted
    unique_Ks = sorted(results_df['Num_Features_K'].unique())

    # -------------------------------------------------------
    # Part 1: SMOTE Impact Analysis (Per K)
    # -------------------------------------------------------
    smote_stats_list = []

    for k in unique_Ks:
        # Filter data for this specific K
        df_k = results_df[results_df['Num_Features_K'] == k]

        # Pivot: Align 'No_SMOTE' and 'With_SMOTE' scores for the same Model+Direction
        # We use PR_AUC (Precision-Recall AUC) as it's best for imbalance, 
        # but you can switch 'values' to 'ROC_AUC' if preferred.
        pivot = df_k.pivot_table(
            index=['Direction', 'Model'], 
            columns='SMOTE', 
            values='PR_AUC' 
        ).dropna()

        # We need pairs. If pivot has columns 'No_SMOTE' and 'With_SMOTE', we are good.
        if len(pivot) > 2 and 'No_SMOTE' in pivot.columns and 'With_SMOTE' in pivot.columns:
            # Wilcoxon Test
            stat, p_val = wilcoxon(pivot['No_SMOTE'], pivot['With_SMOTE'])

            # Calculate average gain (Positive = SMOTE helped)
            avg_diff = (pivot['With_SMOTE'] - pivot['No_SMOTE']).mean()

            smote_stats_list.append({
                'Num_Features_K': k,
                'Test': 'Wilcoxon Signed-Rank',
                'Avg_PR_AUC_Gain': avg_diff,
                'P_Value': p_val,
                'Significant (p<0.05)': p_val < 0.05,
                'Sample_Size': len(pivot)
            })
        else:
            smote_stats_list.append({
                'Num_Features_K': k,
                'Note': 'Not enough data for stats'
            })

    smote_stats_df = pd.DataFrame(smote_stats_list)

    # -------------------------------------------------------
    # Part 2: McNemar's Test (Model vs Model) (Per K)
    # -------------------------------------------------------
    # We will generate a dictionary of DataFrames, one matrix for each K
    mcnemar_matrices = {}

    for k in unique_Ks:
        # Identify models available for this K
        # Keys in prediction store: "{Direction}_{K}_{Model}_{SMOTE}"

        # Filter keys relevant to this K
        relevant_keys = [key for key in predictions_store.keys() if f"_{k}_" in key]
        model_names = sorted(list(set(relevant_keys))) # Unique list of models for this K

        n_models = len(model_names)
        if n_models < 2: continue

        # Create P-Value Matrix
        p_matrix = pd.DataFrame(index=model_names, columns=model_names)

        for i in range(n_models):
            for j in range(n_models):
                key1 = model_names[i]
                key2 = model_names[j]

                if i == j:
                    p_matrix.iloc[i, j] = np.nan
                    continue

                # Retrieve Truth (using the direction prefix in the key)
                # Key format example: "GSE89843_to_GSE207586_20_SVM_Linear_No_SMOTE"
                y_target = None
                for direction, y_true in truth_dict.items():
                    if key1.startswith(direction) and key2.startswith(direction):
                        y_target = y_true
                        break

                # If directions don't match, skip comparison
                if y_target is None:
                    continue

                # Get Predictions
                pred1 = predictions_store[key1]
                pred2 = predictions_store[key2]

                # Contingency Table
                correct_1 = (pred1 == y_target)
                correct_2 = (pred2 == y_target)

                b = np.sum(correct_1 & ~correct_2) # 1 correct, 2 wrong
                c = np.sum(~correct_1 & correct_2) # 1 wrong, 2 correct

                # McNemar
                if (b + c) > 0:
                    res = mcnemar([[0, b], [c, 0]], exact=True)
                    p_matrix.iloc[i, j] = res.pvalue
                else:
                    p_matrix.iloc[i, j] = 1.0 # No disagreement = No difference

        mcnemar_matrices[k] = p_matrix

    return smote_stats_df, mcnemar_matrices

# ==========================================
# EXECUTION & EXPORT BLOCK
# ==========================================

# 1. Setup Truth Dictionary (Ensure this matches your loop variables)
truth_dict = {
    'GSE89843_to_GSE207586': y_GSE207586.values if hasattr(y_GSE207586, 'values') else y_GSE207586,
    'GSE207586_to_GSE89843': y_GSE89843.values if hasattr(y_GSE89843, 'values') else y_GSE89843
}

# 2. Run Stats
results_df = pd.DataFrame(results_list)
smote_stats_df, mcnemar_results = perform_statistical_analysis_by_K(results_df, predictions_store, truth_dict)

# 3. Save to Excel
output_file = "RNASeq_Results_By_FeatureSize.xlsx"

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:

    # Sheet 1: All Raw Metrics
    results_df.to_excel(writer, sheet_name='Raw_Metrics', index=False)

    # Sheet 2: SMOTE Significance (The summary table you asked for)
    smote_stats_df.to_excel(writer, sheet_name='SMOTE_Significance_by_K', index=False)

    # Sheets 3+: McNemar Matrices for each K
    # (Since we have many K values, we create a sheet for each)
    workbook = writer.book
    green_fmt = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})

    for k, matrix in mcnemar_results.items():
        sheet_name = f"McNemar_K{k}"
        matrix.to_excel(writer, sheet_name=sheet_name)

        # Apply formatting
        ws = writer.sheets[sheet_name]
        ws.conditional_format(1, 1, matrix.shape[0], matrix.shape[1], {
            'type': 'cell', 'criteria': '<', 'value': 0.05, 'format': green_fmt
        })

print(f"Stats completed. Check '{output_file}' for the 'SMOTE_Significance_by_K' sheet.")


# In[5]:


from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

def get_deseq2_genes(raw_counts_df, metadata_df, target_col, design_factor, 
                     padj_cutoff=0.05, log2fc_cutoff=1.0, save_path=None):
    """
    Runs DESeq2 on raw counts to find significant genes.
    Args:
        raw_counts_df: (Samples x Genes) Dataframe of raw integers.
        metadata_df: (Samples x Metadata) Dataframe containing the target label.
        design_factor: The column name in metadata_df to compare (e.g., 'condition').
    Returns:
        list: List of significant gene names.
    """

    # Check if results already exist to save time
    if save_path and os.path.exists(save_path):
        print(f"   [DESeq2] Loading cached results from {save_path}")
        res_df = pd.read_csv(save_path, index_col=0)
    else:
        print(f"   [DESeq2] Running Differential Expression Analysis...")

        # 1. Initialize DESeq2 Object
        # Ensure raw_counts are integers
        counts = raw_counts_df.round().astype(int)

        dds = DeseqDataSet(
            counts=counts,
            metadata=metadata_df,
            design_factors=design_factor,
            n_cpus=4 # Adjust based on your CPU
        )

        # 2. Fit Dispersion and LFC
        dds.deseq2()

        # 3. Run Statistical Test (Wald Test)
        stat_res = DeseqStats(dds)
        stat_res.summary()
        res_df = stat_res.results_df

        # Save results
        if save_path:
            res_df.to_csv(save_path)

    # 4. Filter Genes
    # padj < 0.05 AND |log2FoldChange| > 1.0 (2-fold change)
    sig_genes = res_df[
        (res_df['padj'] < padj_cutoff) & 
        (abs(res_df['log2FoldChange']) > log2fc_cutoff)
    ]

    print(f"   [DESeq2] Found {len(sig_genes)} significant genes.")
    return res_df, sig_genes


# In[ ]:


from sklearn.model_selection import RepeatedStratifiedKFold
X_GSE89843_counts = pd.read_csv('C:/Users/farid/xai paper/GSE89843/Data/GSE89843_Raw_Counts_Symbols.csv', index_col=0).T
X_GSE207586_counts = pd.read_csv('C:/Users/farid/xai paper/GSE207586/Data/GSE207586_Raw_Counts_Symbols.csv', index_col=0).T
X_GSE89843_counts.index = X_GSE89843_counts.index.str.replace('.', '-').str.replace('HC', 'Healthy Control')
a = []
for i in X_GSE207586_counts.index.str.split('.'):
    a.append('-'.join(i[:-1]))
X_GSE207586_counts.index = a

common_genes = sorted(list(set(X_GSE89843_counts.columns) & set(X_GSE207586_counts.columns)))

X_GSE89843_counts = X_GSE89843_counts.loc[y_GSE89843.index, common_genes]
X_GSE207586_counts = X_GSE207586_counts.loc[y_GSE207586.index, common_genes]

y_GSE89843 = pd.DataFrame(y_GSE89843)
y_GSE207586 = pd.DataFrame(y_GSE207586)
n_repeats = 5
n_splits = 10
total_runs = n_repeats * n_splits
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

run_id = 0
deseq2_results_GSE89843_all = []
for train_index, val_index in rskf.split(X_GSE89843_counts, y_GSE89843):
    run_id += 1
    print(f"\n--- Run {run_id}/{total_runs} ---")

    # Subset Training Data (we ONLY look at training folds to find genes)
    X_train = X_GSE89843_counts.iloc[train_index]
    y_train = y_GSE89843.iloc[train_index]
    deseq2_results_GSE89843, _ = get_deseq2_genes(X_train,
                                               y_train,
                                               target_col='cancer_type',
                                               design_factor='cancer_type')
    deseq2_results_GSE89843_all.append(deseq2_results_GSE89843)


# In[ ]:


b = []
for i in deseq2_results_GSE89843_all[0].columns:
    b.append(pd.concat(deseq2_results_GSE89843_all, axis=1)[i].mean(axis=1))

b = pd.concat(b, axis=1)
b.columns = deseq2_results_GSE89843_all[0].columns


# In[ ]:


b.loc[['H3-3A', 'CDC42', 'CYTH2', 'CDC37', 'NME4', 'CTSW', 'REST', 'MS4A1', 'RGCC']].to_excel('deseq2_analysis.xlsx')


# In[ ]:


padj_cutoff = 0.05
log2fc_cutoff = 1.0
sig_genes = b[
        (b['padj'] < padj_cutoff) & 
        (abs(b['log2FoldChange']) > log2fc_cutoff)
    ]


# In[ ]:


sig_genes


# In[29]:


stats_results_gse20189 = pd.read_csv('C:/Users/farid/OneDrive/Documents/GSE20189_stats_table.csv', index_col=0)
stats_results_gse20189.loc[['SLC14A1', 'JUND', 'ARHGAP6', 'MPI', 'SCGB1A1', 'IGKC', 'ZCCHC4', 'FTH1', 'GRB10', 'FZD1']].to_excel('limma_analysis.xlsx')


# In[ ]:




