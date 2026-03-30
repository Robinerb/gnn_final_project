import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import QED
from optimize_qed import optimize_molecule 

def evaluate_optimization(num_samples=5):
    df = pd.read_csv('data/zinc_subset.csv')
    test_indices = df[df['qed'] < 0.5].head(num_samples).index.tolist()
    
    final_data = []
    for idx in test_indices:
        orig_smiles = df.loc[idx, 'smiles']
        orig_qed = df.loc[idx, 'qed']
        opt_smiles, final_pred_qed = optimize_molecule(idx=idx, return_score=True)
        
        # calculate real QED
        mol = Chem.MolFromSmiles(opt_smiles)
        real_qed_after = QED.qed(mol) if mol else 0.0
        
        final_data.append({
            "Original_SMILES": orig_smiles,
            "Original_QED": orig_qed,
            "Optimized_SMILES": opt_smiles,
            "Model_Predicted_QED": final_pred_qed,
            "RDKit_Real_QED": real_qed_after,
            "Hallucination_Gap": final_pred_qed - real_qed_after
        })
    
    results_df = pd.DataFrame(final_data)
    results_df.to_csv("results/optimization_results_final.csv", index=False)

if __name__ == "__main__":
    evaluate_optimization()