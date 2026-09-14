"""Domain and surface figures for the thesis.

3D is used only where the quantity is a smooth field over two continuous
indices and the shape is the point; colour there encodes height, so the z axis
already carries the scale and no colour bar is added that would repeat it.
Where the grid is small and sparse, a flat annotated map is used instead: a
bar field would occlude cells and give no way to read a value.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs
from experiments.common import read_csv
from src.envs.cartpole_continuous import CartPoleCompetentAgent, CartPoleDynamics
from src.envs.gridworld_mdp import make_choice_gridworld
from src.metrics.diagnostics import compute_action_margin
from src.planning.dp import value_iteration

ELEVATION, AZIMUTH = 28, -57
BOX_ASPECT = (1.18, 1, 0.66)


def _style_3d(ax, **kwargs):
    """Apply the shared typography to a 3D projection."""
    ax.set(**kwargs)
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_box_aspect(BOX_ASPECT)
    ax.zaxis.labelpad = 2
    ax.tick_params(labelsize=fs.SMALL_PT - 1.5, pad=-1.5)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor('white')
        axis.pane.set_edgecolor(fs.GRID)
        axis.pane.set_alpha(1.0)
        axis._axinfo['grid'].update(color=fs.GRID, linewidth=0.4)


def _captions(fig, texts, y=0.015):
    """Panel captions under each column, matching the flat figures."""
    for x, text in zip(np.linspace(.5 / len(texts), 1 - .5 / len(texts),
                                   len(texts)), texts):
        fig.text(x, y, text, ha='center', va='bottom', fontsize=fs.SMALL_PT,
                 color=fs.INK)


def gridworld_surfaces(report=None):
    env = make_choice_gridworld()
    value, q, _ = value_iteration(env)
    margin = compute_action_margin(q)
    size = env.height
    x, y = np.meshgrid(np.arange(size), np.arange(size))

    fig = plt.figure(figsize=(fs.TEXT_WIDTH_IN, 3.05))
    panels = [(121, value.reshape(size, size), r'$V^*(s)$', 'viridis'),
              (122, margin.reshape(size, size), r'$m(s)$', 'plasma')]
    for position, z, zlabel, cmap in panels:
        ax = fig.add_subplot(position, projection='3d')
        ax.plot_surface(x, y, z, cmap=cmap, edgecolor='#303030',
                        linewidth=.35, antialiased=True)
        ax.contour(x, y, z, zdir='z', offset=float(z.min()), cmap=cmap,
                   levels=7, linewidths=.6)
        _style_3d(ax, xlabel='Cột', ylabel='Hàng', zlabel=zlabel,
                  xticks=range(size), yticks=range(size))
    fig.subplots_adjust(left=.0, right=.90, bottom=.10, top=1., wspace=.30)
    _captions(fig, ['(a) Miền giá trị tối ưu', '(b) Miền action-value margin'])
    fs.save(fig, 'surface_gridworld_value_margin', report)


def cartpole_surface(report=None):
    agent = CartPoleCompetentAgent(CartPoleDynamics())
    theta = np.linspace(-.20, .20, 81)
    theta_dot = np.linspace(-1.25, 1.25, 81)
    th, thd = np.meshgrid(theta, theta_dot)
    margin = np.empty_like(th)
    action = np.empty_like(th)
    for i in range(th.shape[0]):
        for j in range(th.shape[1]):
            state = np.array([0., 0., th[i, j], thd[i, j]])
            q = agent.get_action_values(state)
            margin[i, j] = abs(q[1] - q[0])
            action[i, j] = int(q[1] >= q[0])

    fig = plt.figure(figsize=(fs.TEXT_WIDTH_IN, 3.15))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(th, thd, margin, cmap='cividis', edgecolor='none',
                    rcount=45, ccount=45, antialiased=True)
    ax.contour(th, thd, margin, zdir='z', offset=0, levels=9, cmap='cividis',
               linewidths=.55)
    _style_3d(ax, xlabel=r'$\theta$ (rad)', ylabel=r'$\dot\theta$ (rad/s)')

    ax = fig.add_subplot(122)
    palette = matplotlib.colors.ListedColormap(['#56B4E9', '#E69F00'])
    ax.pcolormesh(th, thd, action, shading='auto', cmap=palette)
    ax.contour(th, thd, margin, levels=[.05, .20], colors='#333333',
               linewidths=.65)
    ax.contour(th, thd, action, levels=[.5], colors='white', linewidths=1.8)
    ax.set(xlabel=r'Góc $\theta$ (rad)',
           ylabel=r'Vận tốc góc $\dot\theta$ (rad/s)')
    ax.grid(False)
    ax.text(-.185, 1.05, '$a = 1$', color='#7A3D00', fontsize=fs.SMALL_PT)
    ax.text(-.185, -1.16, '$a = 0$', color='#00435F', fontsize=fs.SMALL_PT)
    fig.subplots_adjust(left=.0, right=.96, bottom=.22, top=1., wspace=.38)
    _captions(fig, [r'(a) Bề mặt margin tại $x=\dot{x}=0$',
                    '(b) Miền quyết định và đồng mức margin'])
    fs.save(fig, 'surface_cartpole_margin_region', report)


def empirical_error_map(report=None):
    """Measured flip rate and policy loss over the (margin, error) grid.

    Drawn flat rather than as a 3D bar field. The grid is small and sparse, so
    a bar field occludes cells and leaves no way to read a value; an annotated
    map prints every number and keeps the empty combinations visibly empty.
    """
    rows = read_csv('exp1_gridworld_results.csv')
    margins = np.array([float(r['margin']) for r in rows])
    errors = np.array([float(r['actual_l1']) for r in rows])
    flips = np.array([int(r['decision_flip']) for r in rows])
    losses = np.array([float(r['policy_loss']) for r in rows])
    # Fixed bins declared from observed ranges; aggregate all repetitions/seeds.
    edges = np.unique(np.quantile(margins, np.linspace(0, 1, 7)))
    levels = np.array(sorted(set(errors.round(12))))
    centers = (edges[:-1] + edges[1:]) / 2
    flip_grid = np.full((len(levels), len(centers)), np.nan)
    loss_grid = np.full_like(flip_grid, np.nan)
    for i, level in enumerate(levels):
        for j in range(len(centers)):
            upper = (margins <= edges[j + 1] if j == len(centers) - 1
                     else margins < edges[j + 1])
            mask = (np.isclose(errors, level, atol=1e-10)
                    & (margins >= edges[j]) & upper)
            if mask.any():
                flip_grid[i, j] = flips[mask].mean() * 100
                loss_grid[i, j] = losses[mask].mean()

    fig, axes = fs.figure(3.25, nrows=1, ncols=2)
    panels = [(flip_grid, 'Đổi hành động (%)', 'magma_r', '{:.0f}', 2),
              (loss_grid, r'$J(\pi^*)-J(\widehat\pi)$', 'viridis_r', '{:.3f}', 3)]
    for ax, (grid, label, cmap, fmt, _) in zip(axes, panels):
        masked = np.ma.masked_invalid(grid)
        palette = plt.get_cmap(cmap).copy()
        palette.set_bad('#f2f2f2')
        image = ax.pcolormesh(np.arange(len(centers) + 1),
                              np.arange(len(levels) + 1), masked,
                              cmap=palette, shading='flat',
                              edgecolors=fs.GRID, linewidth=.4)
        ax.set(xticks=np.arange(len(centers)) + .5,
               xticklabels=[f'{c:.2f}' for c in centers],
               yticks=np.arange(len(levels)) + .5,
               yticklabels=[f'{v:.2f}' for v in levels],
               ylabel=r'Sai số thực tế $\|\widehat P-P\|_1$')
        ax.tick_params(direction='out', top=False, right=False,
                       labelsize=fs.SMALL_PT - 1.5)
        ax.minorticks_off()
        ax.grid(False)
        # Pick the label colour from the luminance of the cell it sits on, not
        # from a value threshold: a reversed colour map inverts that mapping.
        norm = image.norm
        palette_lookup = image.cmap
        for i in range(len(levels)):
            for j in range(len(centers)):
                if not np.isfinite(grid[i, j]):
                    continue
                red, green, blue, _ = palette_lookup(norm(grid[i, j]))
                luminance = .299 * red + .587 * green + .114 * blue
                ax.text(j + .5, i + .5, fmt.format(grid[i, j]).replace('.', ','),
                        ha='center', va='center', fontsize=fs.SMALL_PT - 2.5,
                        color='white' if luminance < .55 else fs.INK)
        bar = fig.colorbar(image, ax=ax, shrink=.9, pad=.03)
        bar.set_label(label, fontsize=fs.SMALL_PT - 1)
        bar.ax.tick_params(labelsize=fs.SMALL_PT - 2)
        bar.outline.set_linewidth(.6)
    fs.panel_label(axes[0], 'Margin (trung điểm nhóm)', '(a) Tỷ lệ đổi hành động')
    fs.panel_label(axes[1], 'Margin (trung điểm nhóm)', '(b) Tổn thất chính sách')
    fs.save(fig, 'grid_error_impact_map', report)


def first_order_law(report=None):
    """First-order margin sensitivity against the exact solve, and its threshold.

    Laid out like the reference journal's theory-versus-simulation panel. The
    plotted quantity is the margin shift, not the action-value shift: the
    margin is what decides the action, and it is the only one of the two whose
    first-order form needs the rival action's revisit term.
    """
    rows = read_csv('exp7_first_order_law.csv')
    states = sorted({int(r['state']) for r in rows},
                    key=lambda s: next(float(r['true_margin']) for r in rows
                                       if int(r['state']) == s))
    rank = {s: i for i, s in enumerate(states)}
    x = np.array([float(r['fraction_of_cap']) for r in rows])
    y = np.array([rank[int(r['state'])] for r in rows])
    measured = np.array([float(r['measured_margin_shift']) for r in rows])
    predicted = np.array([float(r['predicted_margin_shift']) for r in rows])

    fig = plt.figure(figsize=(fs.TEXT_WIDTH_IN, 3.15))
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(x, y, measured, marker='*', s=9, color='#0000CC',
               depthshade=False, label='Giải chính xác', zorder=3)
    ax.scatter(x, y, predicted, marker='o', s=13, facecolors='none',
               edgecolors='#CC0000', linewidths=.45, depthshade=False,
               label='Bậc nhất đầy đủ', zorder=4)
    _style_3d(ax, xlabel='Tỉ lệ trần khả thi', ylabel='State (theo margin)',
              zlabel=r'$\Delta m$')
    ax.legend(loc='upper left', fontsize=fs.SMALL_PT - 1.5, frameon=True,
              framealpha=1., edgecolor=fs.INK, fancybox=False,
              borderpad=.3, handletextpad=.2)

    # Two thresholds on the same axis: the naive one fires early at the states
    # where the rival revisit weight is large.
    ax = fig.add_subplot(122)
    flips = np.array([int(r['decision_flip']) for r in rows])
    edges = np.linspace(0, 2.2, 23)
    for column, style, label in [
            ('fraction_of_naive_critical', fs.series(1), 'Bỏ hành động cạnh tranh'),
            ('fraction_of_critical', fs.series(0), 'Bậc nhất đầy đủ')]:
        ratio = np.array([float(r[column]) for r in rows])
        centers, rates = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (ratio >= lo) & (ratio < hi)
            if mask.sum():
                centers.append((lo + hi) / 2)
                rates.append(100 * flips[mask].mean())
        ax.plot(centers, rates, label=label, markersize=3.2,
                **dict(style, linestyle=style['linestyle']))
    ax.axvline(1., color='#CC0000', linestyle='--', linewidth=1.1,
               label='Ngưỡng xấp xỉ')
    ax.set(xlabel=r'$t/t_{\mathrm{crit}}$', ylabel='Đổi hành động (%)',
           xlim=(0, 2.25), ylim=(-4, 104))
    ax.grid(True, color=fs.GRID, linewidth=.4)
    fs.legend(ax, loc='upper left', fontsize=fs.SMALL_PT - 2)
    fig.subplots_adjust(left=.0, right=.96, bottom=.22, top=1., wspace=.46)
    _captions(fig, ['(a) Độ nhạy margin bậc nhất',
                    '(b) Ngưỡng đổi quyết định'])
    fs.save(fig, 'first_order_law', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    fs.use()
    gridworld_surfaces(args.report)
    cartpole_surface(args.report)
    empirical_error_map(args.report)
    first_order_law(args.report)
