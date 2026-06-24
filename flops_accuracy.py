import matplotlib.pyplot as plt
import numpy as np
 
# ---------- DATA ----------
# Each entry: (FLOPs in GFLOPs, accuracy %, label, dataset, model_family)
# model_family: 'baseline_published', 'baseline_bnext', 'phasedfd', 'dualdfd', 'fulldfd'
 
# DFFD data (Table 6, confirmed final)
dffd_data = [
    # Published baselines
    (18.0, 99.64, 'Xception', 'baseline_published'),       # AUC used as proxy since acc not reported - skip on accuracy plot
    (15.5, 99.67, 'VGG16', 'baseline_published'),          # same - skip
    # BNext baselines (unfrozen)
    (0.89, 98.95, 'BNext-T', 'baseline_bnext'),
    (1.91, 99.01, 'BNext-S', 'baseline_bnext'),
    (3.39, 98.75, 'BNext-M', 'baseline_bnext'),
    # Proposed
    (4.904, 99.21, 'PhaseDFD-T', 'phasedfd'),
    (10.944, 99.37, 'PhaseDFD-S', 'phasedfd'),
    (20.218, 99.40, 'PhaseDFD-M', 'phasedfd'),
    (4.855, 99.31, 'DualDFD-T', 'dualdfd'),
    (10.895, 99.43, 'DualDFD-S', 'dualdfd'),
    (20.170, 99.46, 'DualDFD-M', 'dualdfd'),
    (4.904, 99.13, 'FullDFD-T', 'fulldfd'),
    (10.944, 99.44, 'FullDFD-S', 'fulldfd'),
    (20.218, 99.38, 'FullDFD-M', 'fulldfd'),
]
 
# CIFAKE data (Table 7, confirmed final)
cifake_data = [
    # Published baselines from [40]
    (4.8, 95.00, 'ResNet', 'baseline_published'),
    (7.63, 96.00, 'VGGNet', 'baseline_published'),
    (5.6, 98.00, 'DenseNet', 'baseline_published'),
    # BNext (unfrozen)
    (0.89, 97.29, 'BNext-T', 'baseline_bnext'),
    (1.91, 96.96, 'BNext-S', 'baseline_bnext'),
    (3.39, 97.35, 'BNext-M', 'baseline_bnext'),
    # Proposed
    (4.904, 98.55, 'PhaseDFD-T', 'phasedfd'),
    (10.944, 98.75, 'PhaseDFD-S', 'phasedfd'),
    (20.218, 98.65, 'PhaseDFD-M', 'phasedfd'),
    (4.855, 98.49, 'DualDFD-T', 'dualdfd'),
    (10.895, 98.54, 'DualDFD-S', 'dualdfd'),
    (20.170, 98.58, 'DualDFD-M', 'dualdfd'),
    (4.904, 98.37, 'FullDFD-T', 'fulldfd'),
    (10.944, 98.68, 'FullDFD-S', 'fulldfd'),
    (20.218, 98.54, 'FullDFD-M', 'fulldfd'),
]
 
# Style mapping
styles = {
    'baseline_published': {'color': '#888888', 'marker': 's', 'size': 80, 'label': 'Published baselines'},
    'baseline_bnext':     {'color': '#2c7fb8', 'marker': 'D', 'size': 80, 'label': 'BNext (RGB only)'},
    'phasedfd':           {'color': '#d7301f', 'marker': 'o', 'size': 100, 'label': 'PhaseDFD (proposed)'},
    'dualdfd':            {'color': '#fdae61', 'marker': '^', 'size': 100, 'label': 'DualDFD (proposed)'},
    'fulldfd':            {'color': '#5e3c99', 'marker': 'v', 'size': 100, 'label': 'FullDFD (proposed)'},
}
 
 
def plot_pareto(ax, data, title, ylim, exclude_no_acc=True):
    """Plot one panel with Pareto-frontier markers."""
    families_seen = set()
    for flops, acc, label, family in data:
        if acc is None:
            continue
        s = styles[family]
        legend_label = s['label'] if family not in families_seen else None
        families_seen.add(family)
        ax.scatter(flops, acc,
                   c=s['color'], marker=s['marker'],
                   s=s['size'], edgecolors='black', linewidths=0.6,
                   label=legend_label, zorder=3, alpha=0.95)
 
    # Compute and draw Pareto frontier (best accuracy at each FLOPs level)
    pts = sorted([(f, a) for f, a, _, _ in data if a is not None])
    pareto = []
    best_acc = -np.inf
    for f, a in pts:
        if a > best_acc:
            pareto.append((f, a))
            best_acc = a
    if len(pareto) >= 2:
        px, py = zip(*pareto)
        ax.plot(px, py, '--', color='black', alpha=0.35,
                linewidth=1.0, zorder=1, label='Pareto frontier')
 
    ax.set_xscale('log')
    ax.set_xlabel('FLOPs (G)', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
 
 
# ---------- BUILD FIGURE ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
 
# CIFAKE panel first so legend collects all marker types
plot_pareto(axes[1], cifake_data, '(b) CIFAKE', ylim=(94.5, 99.0))
 
# DFFD panel — note Xception/VGG16 don't report accuracy, so we skip them via None substitution
dffd_plot = [(f, a if name not in ('Xception', 'VGG16') else None, name, fam)
             for f, a, name, fam in dffd_data]
plot_pareto(axes[0], dffd_plot, '(a) DFFD', ylim=(98.5, 99.7))
 
# Single shared legend at top - collect from CIFAKE panel which has all marker families
handles, labels = axes[1].get_legend_handles_labels()
# Deduplicate by label, preserve order
seen, dedup_h, dedup_l = set(), [], []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l)
        dedup_h.append(h)
        dedup_l.append(l)
fig.legend(dedup_h, dedup_l, loc='upper center', ncol=6,
           bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9)
 
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('/sweet/binary_deepfake_detection/figures/flops_vs_accuracy.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('/sweet/binary_deepfake_detection/figures/flops_vs_accuracy.pdf',
            bbox_inches='tight', facecolor='white')
print("Figure D saved.")
 