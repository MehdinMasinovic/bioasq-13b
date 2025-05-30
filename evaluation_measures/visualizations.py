import matplotlib.pyplot as plt
import numpy as np

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

# Metrics
flat_metrics = ["accuracy","EbP","EbR","EbF","MaP","MaR","MaF","MiP","MiR","MiF"]
flat_trad = [0.16902605292209646,0.24881725566053395,0.475378940647961,0.2755868083632468,0.11700287620720116,0.3257178402296632,0.2744170617470586,0.1875,0.4419642857142857,0.26329787234042556]
flat_neural = [0.18350678224606362,0.2405941151962519,0.5004033826285365,0.2894156098469736,0.13257054156192166,0.3251678285710037,0.28002625970325873,0.20733863837312114,0.4527027027027027,0.28441479684657367]

flat_tradcutoff = [0.16506977381720306,0.24484687372082295,0.4114062378373234,0.27128201537426505,0.14255343447255633,0.25153240582595027,0.22629271500819695,0.21779964221824688,0.36754716981132074,0.2735186745296265]
flat_hybrid = [0.1668717956999727,0.24597697408471444,0.4452653504096683,0.2752395276758952,0.13376528998093726,0.2948038160448893,0.25466311402029523,0.2084752175558078,0.4118086696562033,0.2768148706355187]
flat_tad = [0.13800416893530035,0.17942371243861613,0.5090385308835936,0.2343318981453979,0.09525686117776037,0.34576303304785494,0.27617978619101063,0.15019011406844107,0.47234678624813153,0.22791200865488642]

hier_metrics = ["hP", "hR", "hF", "LCA-P", "LCA-R", "LCA-F"]
hier_trad = [0.366101,0.673475,0.418427,0.237381,0.448031,0.276247]
hier_neural = [0.361674,0.678686,0.428603,0.251874,0.45836,0.296766]

hier_tad = [0.284825,0.68901,0.36299,0.190819,0.479814,0.251021]
hier_tradcutoff = [0.380848,0.588349,0.419022,0.24609,0.387734,0.275501]
hier_hybrid = [0.368385,0.612914,0.410633,0.24368,0.418993,0.27899]

# Plot flat metrics
plot_metrics(flat_metrics, flat_tad, flat_hybrid, 'Flat Metrics Comparison')
# Plot hierarchical metrics
plot_metrics(hier_metrics, hier_tad, hier_hybrid, 'Hierarchical Metrics Comparison')


def plot3(flat_labels, hier_labels, flats, hiers, focus_labels, legend):
    # Choose only focus labels from flat and hierarchical metrics
    flat_indices = [flat_labels.index(label) for label in focus_labels if label in flat_labels]
    hier_indices = [hier_labels.index(label) for label in focus_labels if label in hier_labels]
    flats_focused = [[flat[i] for i in flat_indices] for flat in flats]
    hiers_focused = [[hier[i] for i in hier_indices] for hier in hiers]
    print(flats_focused, hiers_focused)
    x = np.arange(len(focus_labels)) 
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = []
    for i, (flat, hier) in enumerate(zip(flats_focused, hiers_focused)):
        bars.append(ax.bar(x - width/2 + i*width/len(flats_focused), flat + hier, width/len(flats_focused), label=legend[i]))
    # Add labels
    ax.set_ylabel('Score')
    ax.set_title('Models compared by F scores')
    ax.set_xticks(x)
    ax.set_xticklabels(focus_labels)
    ax.legend()
    ax.set_ylim(0, max(max(max(flat) for flat in flats_focused), max(max(hier) for hier in hiers_focused)) + 0.1) 
    plt.tight_layout()
    plt.show()


plot3(flat_metrics, hier_metrics, [flat_tad,flat_neural,flat_tradcutoff,flat_hybrid], [hier_tad,hier_neural,hier_tradcutoff,hier_hybrid], 
        ['MiF','EbF','hF','LCA-F'], ['Traditional IR', 'Neural NLP', 'Trad + Cutoff', 'Trad + Rerank'])
