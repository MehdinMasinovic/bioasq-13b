import matplotlib.pyplot as plt
import numpy as np

# Metrics
flat_metrics = ["accuracy","EbP","EbR","EbF","MaP","MaR","MaF","MiP","MiR","MiF"]
flat_trad = [0.16902605292209646,0.24881725566053395,0.475378940647961,0.2755868083632468,0.11700287620720116,0.3257178402296632,0.2744170617470586,0.1875,0.4419642857142857,0.26329787234042556]
flat_neural = [0.18350678224606362,0.2405941151962519,0.5004033826285365,0.2894156098469736,0.13257054156192166,0.3251678285710037,0.28002625970325873,0.20733863837312114,0.4527027027027027,0.28441479684657367]

hier_metrics = ["hP", "hR", "hF", "LCA-P", "LCA-R", "LCA-F"]
hier_trad = [0.366101,0.673475,0.418427,0.237381,0.448031,0.276247]
hier_neural = [0.361674,0.678686,0.428603,0.251874,0.45836,0.296766]

def plot_metrics(metrics, traditional, neural_nlp, title):
    x = np.arange(len(metrics))  # label locations
    width = 0.35  # bar width

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, traditional, width, label='Traditional IR')
    bars2 = ax.bar(x + width/2, neural_nlp, width, label='Neural NLP')

    # Add labels
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, max(max(traditional), max(neural_nlp)) + 0.1)  # Set y-axis limit

    # Add bar labels
    for bar in bars1 + bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# Plot flat metrics
plot_metrics(flat_metrics, flat_trad, flat_neural, 'Flat Metrics Comparison')
# Plot hierarchical metrics
plot_metrics(hier_metrics, hier_trad, hier_neural, 'Hierarchical Metrics Comparison')