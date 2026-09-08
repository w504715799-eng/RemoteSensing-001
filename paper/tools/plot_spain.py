"""Plot the published descriptive Spain results; no raw data or model access."""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_spain import read_verified, tables

SCIENCE_SHA = 'f09fc42555cb00deac5125e478b61817200de9a796bd668e5abcc15fbfb07fe4'


def main():
    root = Path(__file__).resolve().parents[2]
    products = tables(read_verified(root / 'paper/tables/spain-science-v1.json', SCIENCE_SHA))
    output = root / 'paper/figures'
    output.mkdir(exist_ok=True)
    colors = dict(lr='#2563eb', three_model='#ea580c', k5='#15803d',
                  neighborhood='#9333ea', random='#6b7280')
    labels = dict(lr='LR', three_model='Three-model', k5='K5',
                  neighborhood='Neighborhood w3', random='Random expectation')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for column, subset in enumerate(('spain_crops', 'spain_urban')):
        for row, window in enumerate(('R1', 'R9')):
            ax = axes[row, column]
            for method in colors:
                values = [r for r in products['curves'] if r['subset'] == subset
                          and r['window'] == window and r['method'] == method]
                ax.plot([r['coverage'] for r in values], [r['mean_risk'] for r in values],
                        label=labels[method], color=colors[method], linewidth=1.6)
            ax.set(title=f'{subset.removeprefix("spain_").title()} — {window}',
                   xlabel='Retained coverage', ylabel='Equal-ROI mean risk')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(output / 'spain_risk_coverage.pdf')
    fig.savefig(output / 'spain_risk_coverage.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, subset in zip(axes, ('spain_crops', 'spain_urban'), strict=True):
        values = [r for r in products['transfer'] if r['subset'] == subset
                  and r['coverage'] is not None]
        ax.scatter([r['coverage'] for r in values], [r['roi_max_r9'] for r in values],
                   alpha=.8, color='#15803d', edgecolors='white')
        ax.axhline(.05, color='#6b7280', linestyle='--', linewidth=1,
                   label='Internal target (reference only)')
        ax.set(title=f'{subset.removeprefix("spain_").title()} (n={len(values)})',
               xlabel='Coverage at frozen K5 threshold', ylabel='ROI maximum retained R9 risk',
               xlim=(0, 1), ylim=(0, .1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(output / 'spain_threshold_transfer.pdf')
    fig.savefig(output / 'spain_threshold_transfer.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
