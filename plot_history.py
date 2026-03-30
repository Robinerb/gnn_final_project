import pandas as pd
import matplotlib.pyplot as plt

def plot_training_results():
    # load the history files
    try:
        smiles_df = pd.read_csv('results/smiles_history.csv')
        selfies_df = pd.read_csv('results/selfies_history.csv')
    except FileNotFoundError:
        print("Error: History files not found.")
        return

    # plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(smiles_df['epoch'], smiles_df['loss'], label='SMILES VAE', color='red', linewidth=2)
    plt.plot(selfies_df['epoch'], selfies_df['loss'], label='SELFIES VAE', color='green', linewidth=2)
    
    plt.title('Training Loss Comparison: SMILES vs. SELFIES', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average VAE Loss (Reconstruction + KL)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('figures/loss_comparison_plot.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_training_results()