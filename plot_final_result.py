import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hallucination_gap(csv_file='results/optimization_results_final.csv'):
    df = pd.read_csv(csv_file)
    
    # prepare data for plotting
    plot_data = []
    for i, row in df.iterrows():
        plot_data.append({'Mol': f'Mol {i+1}', 'QED': row['Model_Predicted_QED'], 'Type': 'Model Prediction'})
        plot_data.append({'Mol': f'Mol {i+1}', 'QED': row['RDKit_Real_QED'], 'Type': 'Ground Truth (RDKit)'})
    
    plot_df = pd.DataFrame(plot_data)
    
    # plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Mol', y='QED', hue='Type', data=plot_df, palette=['#4A90E2', '#D0021B'])
    
    plt.title('The Hallucination Gap: Predicted vs. Real QED', fontsize=14, pad=15)
    plt.ylabel('QED Score', fontsize=12)
    plt.xlabel('Optimized Molecules', fontsize=12)
    plt.ylim(0, 1.1)
    
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

    plt.tight_layout()
    plt.savefig('figures/hallucination_gap.png', dpi=300)

if __name__ == "__main__":
    plot_hallucination_gap()