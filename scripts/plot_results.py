"""The six result figures, drawn from the committed CSVs in the shared style.

Every panel carries its caption beneath the axis, every series is separable in
greyscale, and closely spaced curves get a magnified inset rather than being
left for the reader to squint at.
"""
import numpy as np
from matplotlib.colors import LogNorm

import figstyle as fs
from experiments.common import MAGNITUDES, SEEDS, auc, mean_ci
from experiments.exp4_fork_signed_direction import PAIR_MAGNITUDES

NARROW = fs.series(1)   # margin below the median
WIDE = fs.series(0)     # margin at or above the median
DOWN = fs.series(1)     # reflection projected negatively on V*
UP = fs.series(0)       # reflection projected positively on V*


def _band(ax, x, per_seed, style, label, alpha=.15):
    """Plot the across-seed mean with a pointwise 95% Student-t band."""
    arr = np.asarray(per_seed, float)
    mean = arr.mean(axis=0)
    half = np.array([mean_ci(arr[:, i])['high'] - mean[i] for i in range(arr.shape[1])])
    ax.plot(x, mean, label=label, **style)
    ax.fill_between(x, mean - half, mean + half, color=style['color'],
                    alpha=alpha, linewidth=0)
    return mean


def _roc(labels, score):
    """Return the ROC curve of `score` against binary `labels`."""
    order = np.argsort(-np.asarray(score, float))
    y = np.asarray(labels, float)[order]
    positives, negatives = y.sum(), (1 - y).sum()
    if not positives or not negatives:
        return None
    tpr = np.concatenate([[0.], np.cumsum(y) / positives])
    fpr = np.concatenate([[0.], np.cumsum(1 - y) / negatives])
    return fpr, tpr


def grid_error_margin(ctx, report):
    """Flip rate and policy loss against realized error, split by margin."""
    fig, axes = fs.figure(3.35, nrows=1, ncols=2)

    def curves(ax, key, scale):
        drawn = {}
        for group, style, label in [('narrow', NARROW, 'Margin dưới trung vị'),
                                    ('wide', WIDE, 'Margin từ trung vị')]:
            per_seed = [[ctx.col(ctx.sub(ctx.grid, seed=s, group=group,
                                         requested_l1=m), key).mean() * scale
                         for m in MAGNITUDES] for s in SEEDS]
            drawn[group] = _band(ax, MAGNITUDES, per_seed, style, label)
        return drawn

    curves(axes[0], 'decision_flip', 100)
    axes[0].set_ylabel('Đổi hành động (%)')
    fs.panel_label(axes[0], r'Sai số thực tế $\|\widehat P-P\|_1$',
                   '(a) Quyết định tại dòng bị nhiễu')
    fs.legend(axes[0], loc='upper left')
    # The wide-margin group hugs zero at this scale; magnify it instead of
    # leaving a flat line the reader cannot resolve.
    fs.zoom_inset(axes[0], (.52, .12, .44, .34), (0.03, 0.72), (-0.15, 0.9),
                  lambda a: curves(a, 'decision_flip', 100), loc1=2, loc2=4)

    curves(axes[1], 'policy_loss', 1)
    axes[1].set_ylabel(r'$J(\pi^*)-J(\widehat\pi)$')
    fs.panel_label(axes[1], r'Sai số thực tế $\|\widehat P-P\|_1$',
                   '(b) Tổn thất từ trạng thái xuất phát')
    fs.legend(axes[1], loc='upper left')
    fs.save(fig, 'grid_error_margin', report)


def grid_matched_pairs(ctx, report):
    """The two reflection branches, and their paired per-seed difference.

    The requested magnitude is capped by the successor probabilities, so only
    two distinct errors are actually feasible here. Levels that collapse onto
    the same realized error are pooled rather than drawn as separate points.
    """
    fig, axes = fs.figure(3.35, nrows=1, ncols=2)
    achieved = [ctx.col(ctx.sub(ctx.pairs, requested_l1=m), 'l1_down').mean()
                for m in MAGNITUDES]
    levels = sorted({round(a, 6) for a in achieved})

    def by_level(seed, level, column):
        rows = [r for r in ctx.sub(ctx.pairs, seed=seed)
                if round(float(r['l1_down']), 6) == level]
        return ctx.col(rows, column).mean()

    positions = np.arange(len(levels))
    width = .3
    hatches = ['', '///']
    for offset, hatch, (side, style, label) in zip(
            (-width / 2, width / 2), hatches, [
                ('down', DOWN, r'$\delta\!\cdot\!V^*<0$'),
                ('up', UP, r'$\delta\!\cdot\!V^*>0$')]):
        per_seed = np.array([[by_level(s, lv, f'flip_{side}') * 100
                              for lv in levels] for s in SEEDS])
        mean = per_seed.mean(axis=0)
        half = [mean_ci(per_seed[:, i])['high'] - mean[i]
                for i in range(len(levels))]
        axes[0].bar(positions + offset, mean, width, label=label,
                    color=style['color'], edgecolor=fs.INK, linewidth=.6,
                    hatch=hatch, yerr=half, capsize=2.5,
                    error_kw=dict(elinewidth=.7, capthick=.7), zorder=3)
        # The positive branch is flat zero, which draws no visible bar. The
        # number has to be written or the reader sees a missing series.
        for x, value, error in zip(positions + offset, mean, half):
            axes[0].text(x, value + error + .25, f'{value:.2f}', ha='center',
                         va='bottom', fontsize=fs.SMALL_PT - 2, color=fs.INK,
                         zorder=5)
    axes[0].set(xticks=positions, xticklabels=[f'{lv:.2f}' for lv in levels],
                ylabel='Đổi hành động (%)', ylim=(0, 17.5))
    axes[0].tick_params(axis='x', which='minor', bottom=False, top=False)
    fs.panel_label(axes[0], r'Sai số thực tế đạt được $\|\widehat P-P\|_1$',
                   '(a) Quyết định tại state can thiệp')
    fs.legend(axes[0], loc='upper left')
    # The ceiling itself is a result: past 0.15 the requested magnitude buys
    # no extra perturbation, because the successor mass runs out.
    inset = axes[0].inset_axes((.65, .69, .32, .25))
    inset.plot(MAGNITUDES, MAGNITUDES, color='#555555', linestyle=':',
               linewidth=.8)
    inset.plot(MAGNITUDES, achieved, color=DOWN['color'], marker='o',
               markersize=2.4, linewidth=1.0)
    inset.set_xlabel('Yêu cầu', fontsize=fs.SMALL_PT - 2.5, labelpad=1)
    inset.set_ylabel('Đạt được', fontsize=fs.SMALL_PT - 2.5, labelpad=1)
    inset.tick_params(labelsize=fs.SMALL_PT - 3, direction='in',
                      top=True, right=True, pad=1.5)
    inset.grid(True, color=fs.GRID, linewidth=.35)

    # The inferential quantity is the within-seed difference, so plot it
    # directly instead of leaving the reader to subtract two overlapping bands.
    difference = np.array([ctx.col(ctx.sub(ctx.pairs, seed=s), 'loss_down').mean()
                           - ctx.col(ctx.sub(ctx.pairs, seed=s), 'loss_up').mean()
                           for s in SEEDS])
    jitter = np.linspace(-.22, .22, len(difference))
    axes[1].axvline(0, color='#555555', linestyle=':', linewidth=.9)
    axes[1].scatter(difference, jitter, s=12, facecolors='none',
                    edgecolors=UP['color'], linewidths=.7, zorder=3,
                    label=f'Từng seed (n = {len(SEEDS)})')
    interval = mean_ci(difference)
    axes[1].errorbar([interval['mean']], [0],
                     xerr=[[interval['mean'] - interval['low']],
                           [interval['high'] - interval['mean']]],
                     color=DOWN['color'], marker='D', markersize=5.5,
                     linewidth=1.6, capsize=3.5, zorder=4,
                     label='Trung bình, CI 95%')
    axes[1].set(ylim=(-.5, .5), yticks=[])
    axes[1].tick_params(axis='y', which='both', left=False, right=False)
    fs.panel_label(axes[1], r'$\Delta$ tổn thất: chiếu âm $-$ chiếu dương',
                   '(b) Hiệu ứng ghép cặp trong từng seed')
    fs.legend(axes[1], loc='upper left')
    fs.save(fig, 'grid_matched_pairs', report)


def grid_layout(ctx, report):
    """Optimal value with the greedy policy, and the action-value margin."""
    fig, axes = fs.figure(3.1, nrows=1, ncols=2)
    size = ctx.env.height
    panels = [(ctx.v, '(a) Giá trị tối ưu và chính sách', r'$V^*(s)$'),
              (ctx.margin, '(b) Action-value margin', r'$m(s)$')]
    for ax, (z, panel, bar) in zip(axes, panels):
        image = ax.imshow(z.reshape(size, size), cmap='cividis')
        ax.grid(False)
        ax.set(xticks=range(size), yticks=range(size), ylabel='Hàng')
        ax.tick_params(direction='out', top=False, right=False)
        ax.minorticks_off()
        fs.panel_label(ax, 'Cột', panel)
        colorbar = fig.colorbar(image, ax=ax, shrink=.82, pad=.03)
        colorbar.set_label(bar, fontsize=fs.SMALL_PT)
        colorbar.ax.tick_params(labelsize=fs.SMALL_PT - 1)
        colorbar.outline.set_linewidth(.6)
        midpoint = (z.min() + z.max()) / 2
        for s in range(size * size):
            row, col = divmod(s, size)
            if ax is axes[0]:
                text = 'G' if s == ctx.env.goal_state else '↑↓←→'[ctx.pi[s]]
                if (row, col) in ctx.env.hazards:
                    text = 'H\n' + text
                if s == 0:
                    text = 'S\n' + text
            else:
                text = f'{z[s]:.2f}'
            ax.text(col, row, text, ha='center', va='center',
                    fontsize=fs.SMALL_PT - 1.5,
                    color='white' if z[s] < midpoint else 'black')
    fs.save(fig, 'grid_layout', report)


def cart_margin(ctx, report):
    """Prediction error against margin, and how well each ranks the flips."""
    fig, axes = fs.figure(3.35, nrows=1, ncols=2)
    # Sixty thousand test points overplot into a solid blob as a scatter, so
    # the kept decisions are drawn as a density and only the rare flips, which
    # are the object of interest, keep individual markers.
    kept = ctx.sub(ctx.cart, decision_flip=0)
    flipped = ctx.sub(ctx.cart, decision_flip=1)
    # An explicit 2D histogram drawn with pcolormesh, not hexbin: a log-scaled
    # hexbin collection is dropped entirely by the PDF backend, which would
    # ship a figure whose main data layer is missing from the report.
    margin, error = ctx.col(kept, 'margin'), ctx.col(kept, 'pred_mse')
    x_edges = np.linspace(0., margin.max(), 41)
    y_edges = np.logspace(np.log10(error.min()), np.log10(error.max()), 37)
    counts, _, _ = np.histogram2d(margin, error, bins=[x_edges, y_edges])
    density = axes[0].pcolormesh(x_edges, y_edges,
                                 np.ma.masked_equal(counts, 0).T,
                                 norm=LogNorm(vmin=1, vmax=counts.max()),
                                 cmap='Blues', shading='flat', zorder=2)
    axes[0].scatter(ctx.col(flipped, 'margin'), ctx.col(flipped, 'pred_mse'),
                    s=16, marker='x', linewidths=.7, c=NARROW['color'],
                    label=f'Đổi hành động (n = {len(flipped)})', zorder=4)
    axes[0].set(yscale='log', ylabel='MSE dự báo trạng thái')
    bar = fig.colorbar(density, ax=axes[0], shrink=.85, pad=.03)
    bar.set_label('Số điểm giữ hành động', fontsize=fs.SMALL_PT - 1)
    bar.ax.tick_params(labelsize=fs.SMALL_PT - 2)
    bar.outline.set_linewidth(.6)
    fs.panel_label(axes[0], 'Margin tham chiếu', '(a) Tập kiểm tra độc lập')
    fs.legend(axes[0], loc='lower left')

    # The AUROC headline number deserves the curve behind it.
    span = np.linspace(0, 1, 201)
    for key, sign, style, label in [('margin', -1, NARROW, r'$-$Margin'),
                                    ('pred_mse', 1, WIDE, 'MSE dự báo')]:
        interpolated = []
        for seed in SEEDS:
            rows = ctx.sub(ctx.cart, seed=seed)
            curve = _roc(ctx.col(rows, 'decision_flip'), sign * ctx.col(rows, key))
            if curve:
                interpolated.append(np.interp(span, curve[0], curve[1]))
        if interpolated:
            _band(axes[1], span, interpolated, dict(style, marker=''), label)
    axes[1].plot([0, 1], [0, 1], color='#555555', linestyle=':', linewidth=.8,
                 label='Ngẫu nhiên')
    axes[1].set(xlim=(0, 1), ylim=(0, 1.02), ylabel='Tỷ lệ dương thật')
    fs.panel_label(axes[1], 'Tỷ lệ dương giả', '(b) ROC trung bình theo seed')
    fs.legend(axes[1], loc='lower right')
    fs.save(fig, 'cart_margin', report)


def cart_evaluation(ctx, report):
    """Per-seed discrimination, and the paired return of the two policies."""
    fig, axes = fs.figure(3.35, nrows=1, ncols=2)
    offsets = np.linspace(-.16, .16, len(SEEDS))
    for i, (key, label) in enumerate([('auc_margin', r'$-$Margin'),
                                      ('auc_mse', 'MSE dự báo')]):
        data = np.asarray(ctx.summary_series[key], float)
        style = fs.series(1 - i)
        finite = np.isfinite(data)
        axes[0].scatter(np.full(len(data), i)[finite] + offsets[finite],
                        data[finite], s=11, facecolors='none',
                        edgecolors=style['color'], linewidths=.7, zorder=3)
        axes[0].plot([i - .27, i + .27], [np.nanmean(data)] * 2,
                     color=style['color'], linewidth=1.6, zorder=4)
    axes[0].axhline(.5, color='#555555', linestyle=':', linewidth=.8)
    axes[0].set(xticks=[0, 1], xticklabels=[r'$-$Margin', 'MSE dự báo'],
                ylabel='AUROC', ylim=(.35, 1.04), xlim=(-.5, 1.5))
    axes[0].tick_params(axis='x', which='minor', bottom=False, top=False)
    fs.panel_label(axes[0], '', '(a) Phân biệt hành động thay đổi')

    pairs = np.array([[ctx.col(ctx.sub(ctx.roll, seed=s, policy=p),
                               'return_value').mean()
                       for p in ['reference', 'learned_model']] for s in SEEDS])
    for low, high in pairs:
        axes[1].plot([0, 1], [low, high], color='#9a9a9a', linewidth=.6,
                     alpha=.85, zorder=2)
    means = pairs.mean(axis=0)
    interval = [mean_ci(pairs[:, i]) for i in range(2)]
    axes[1].errorbar([0, 1], means,
                     yerr=[[means[i] - interval[i]['low'] for i in range(2)],
                           [interval[i]['high'] - means[i] for i in range(2)]],
                     color=DOWN['color'], marker='D', markersize=5,
                     linewidth=1.6, capsize=3, zorder=4,
                     label=f'Trung bình {len(SEEDS)} seed, CI 95%')
    axes[1].set(xticks=[0, 1], xlim=(-.35, 1.35),
                xticklabels=['Mô hình thật', 'Mô hình học'],
                ylabel=f'Return trung bình ({ctx.episodes} episode)')
    axes[1].tick_params(axis='x', which='minor', bottom=False, top=False)
    fs.panel_label(axes[1], '', '(b) Đánh giá trên CartPole-v1')
    fs.legend(axes[1], loc='lower left')
    fs.save(fig, 'cart_evaluation', report)


def baseline_learning(ctx, report):
    """Greedy-policy return against the real-interaction budget."""
    fig, ax = fs.figure(3.9)
    steps = ctx.col(ctx.sub(ctx.base, algorithm='Q-learning', noise_level=0.,
                            seed=SEEDS[0]), 'real_steps')

    def curves(target):
        for i, (name, noise, label) in enumerate(ctx.baseline_labels):
            rows = ctx.sub(ctx.base, algorithm=name, noise_level=noise)
            per_seed = [ctx.col(ctx.sub(rows, seed=s), 'greedy_return')
                        for s in SEEDS]
            # No smoothing: every checkpoint is plotted as evaluated.
            _band(target, steps, per_seed, dict(fs.series(i), marker=''), label,
                  alpha=.11)
        target.axhline(ctx.optimal_return, color='#333333', linestyle=(0, (6, 2)),
                       linewidth=.9, label='Tối ưu (quy hoạch động)')

    curves(ax)
    ax.set(xlabel='Số tương tác thật',
           ylabel='Return chiết khấu của chính sách greedy')
    fs.legend(ax, loc='lower left', ncol=1)
    # Q-learning and noise-free Dyna-Q converge almost on top of each other,
    # and their difference is the comparison the thesis actually reports. The
    # inset sits in the empty lower band so it crosses no curve.
    window = steps >= steps.max() * .8
    finals = np.array([[np.mean([ctx.col(ctx.sub(ctx.base, algorithm=n,
                                                 noise_level=v, seed=s),
                                         'greedy_return')[i] for s in SEEDS])
                        for i in np.flatnonzero(window)]
                       for n, v, _ in ctx.baseline_labels[:2]])
    low, high = finals.min(), max(ctx.optimal_return, finals.max())
    pad = (high - low) * .30 + 1e-3
    # Sitting directly beneath the magnified window keeps both connectors
    # short and vertical instead of cutting a diagonal across the axes.
    fs.zoom_inset(ax, (.58, .08, .40, .34),
                  (steps[window][0], steps.max()), (low - pad, high + pad),
                  curves, loc1=2, loc2=1)
    fs.save(fig, 'baseline_learning', report)


def fork_signed_direction(ctx, report):
    """Margin-closing minus margin-opening repair value across 25 Fork MDPs."""
    fig, axes = fs.figure(3.35, nrows=1, ncols=2)
    order = np.argsort(ctx.fork_per_mdp)
    values = np.asarray(ctx.fork_per_mdp)[order]
    positions = np.arange(len(values))
    colours = [DOWN['color'] if v > 0 else UP['color'] for v in values]
    axes[0].axvline(0, color='#555555', linestyle=':', linewidth=.9)
    axes[0].scatter(values, positions, s=13, facecolors='none',
                    edgecolors=colours, linewidths=.8, zorder=3)
    interval = ctx.summary['fork_difference']
    axes[0].errorbar([interval['mean']], [len(values) + 2.2],
                     xerr=[[interval['mean'] - interval['low']],
                           [interval['high'] - interval['mean']]],
                     color=DOWN['color'], marker='D', markersize=5,
                     linewidth=1.6, capsize=3.5, zorder=4,
                     label='Gộp, CI bootstrap cụm 95%')
    axes[0].set(ylim=(-2, len(values) + 5), yticks=[])
    axes[0].tick_params(axis='y', which='both', left=False, right=False)
    fs.panel_label(axes[0], r'$\Delta$ giá trị sửa: đóng $-$ mở margin',
                   f'(a) Từng MDP (n = {len(values)})')
    fs.legend(axes[0], loc='lower right')

    # The contrast only exists where a decision actually moves, so it grows
    # with severity and then stops when the feasibility cap binds.
    for flag, style, label in [(1, DOWN, 'Hành động greedy'),
                               (0, UP, 'Hành động đối thủ')]:
        per_seed = [[ctx.col(ctx.sub(ctx.fork, mdp_id=i, is_greedy_action=flag,
                                     requested_magnitude=m),
                             'repair_difference').mean()
                     for m in PAIR_MAGNITUDES] for i in ctx.fork_ids]
        _band(axes[1], PAIR_MAGNITUDES, per_seed, style, label)
    axes[1].axhline(0, color='#555555', linestyle=':', linewidth=.9)
    axes[1].set_ylabel(r'$\Delta$ giá trị sửa')
    fs.panel_label(axes[1], r'Độ lớn yêu cầu $\|\widehat P-P\|_1$',
                   '(b) Theo mức sai số')
    fs.legend(axes[1], loc='upper left')
    fs.save(fig, 'fork_signed_direction', report)


def mujoco_signal_audit(ctx, report):
    """The preregistered kill-gate across twelve MuJoCo cells."""
    fig, axes = fs.figure(3.9, nrows=1, ncols=2)
    hosts = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
    ordered = [r for host in hosts for r in ctx.mj if r['env_id'] == host]
    positions = np.arange(len(ordered))[::-1]
    for i, (row, y) in enumerate(zip(ordered, positions)):
        style = fs.series(hosts.index(row['env_id']))
        mean = float(row['delta2_mean'])
        low, high = float(row['delta2_low']), float(row['delta2_high'])
        axes[0].errorbar([mean], [y], xerr=[[mean - low], [high - mean]],
                         color=style['color'], marker=style['marker'],
                         markersize=3.6, linewidth=1.0, capsize=2,
                         label=row['env_id'] if i % 4 == 0 else None, zorder=3)
    axes[0].axvline(0, color='#555555', linestyle=':', linewidth=.9)
    axes[0].set(yticks=positions,
                yticklabels=[f"s{r['seed']} {int(r['step']) // 1000}k" for r in ordered],
                ylim=(-1, len(ordered)))
    axes[0].tick_params(axis='y', labelsize=fs.SMALL_PT - 2, which='both',
                        right=False)
    fs.panel_label(axes[0], r'$\rho(S_{\mathrm{cont}})-\rho(\sigma_{\mathrm{dyn}})$',
                   '(a) Cổng quyết định trên 12 ô')
    fs.legend(axes[0], loc='lower left', fontsize=fs.SMALL_PT - 1)

    # The negative gate is only half the story: each error type still has a
    # signal that ranks it, and it is the matching one.
    signals = ['sigma_dyn', 'sigma_Q', 'sigma_grad_aQ', 'S_cont']
    pretty = [r'$\sigma_{\mathrm{dyn}}$', r'$\sigma_Q$',
              r'$\sigma_{\nabla_a Q}$', r'$S_{\mathrm{cont}}$']
    labels = [('y_A', 'Sai lệch\nhành động'), ('y_B', 'Sai số\nđộng học'),
              ('y_C', 'Sai số\ngiá trị')]
    spots = np.arange(len(labels))
    width = .2
    for i, signal in enumerate(signals):
        style = fs.series(i)
        means = [np.mean([float(r[f'rho_{signal}_{lab}']) for r in ctx.mj])
                 for lab, _ in labels]
        axes[1].bar(spots + (i - 1.5) * width, means, width, label=pretty[i],
                    color=style['color'], edgecolor=fs.INK, linewidth=.5,
                    zorder=3)
    axes[1].set(xticks=spots, xticklabels=[name for _, name in labels],
                ylabel=r'Spearman $\rho$ trung bình 12 ô', ylim=(0, 1.02))
    axes[1].tick_params(axis='x', which='minor', bottom=False, top=False,
                        labelsize=fs.SMALL_PT - 1)
    fs.panel_label(axes[1], '', '(b) Mỗi loại sai số có tín hiệu riêng')
    fs.legend(axes[1], loc='upper left', ncol=2, fontsize=fs.SMALL_PT - 1)
    fs.save(fig, 'mujoco_signal_audit', report)


def cartpole_holdout(ctx, report):
    """The same diagnosis scored on trajectories instead of a uniform box."""
    fig, axes = fs.figure(3.5, nrows=1, ncols=2)
    scores = [('true_margin', -1, r'Margin tham chiếu (đặc quyền)'),
              ('unprivileged_score', 1,
               r'$\sigma_{\mathrm{dyn}}/(\widehat m+\varepsilon)$ (không đặc quyền)'),
              ('sigma_dyn', 1, r'$\sigma_{\mathrm{dyn}}$ (không đặc quyền)')]
    span = np.linspace(0, 1, 201)
    for i, (name, sign, label) in enumerate(scores):
        interpolated = []
        for seed in SEEDS:
            rows = ctx.sub(ctx.hold, seed=seed)
            curve = _roc(ctx.col(rows, 'decision_flip'), sign * ctx.col(rows, name))
            if curve:
                interpolated.append(np.interp(span, curve[0], curve[1]))
        _band(axes[0], span, interpolated, dict(fs.series(i), marker=''), label)
    axes[0].plot([0, 1], [0, 1], color='#555555', linestyle=':', linewidth=.8)
    axes[0].set(xlim=(0, 1), ylim=(0, 1.02), ylabel='Tỷ lệ dương thật')
    fs.panel_label(axes[0], 'Tỷ lệ dương giả', '(a) ROC trên quỹ đạo held-out')
    fs.legend(axes[0], loc='lower right', fontsize=fs.SMALL_PT - 1.5)

    # The uniform box and the visited distribution are two different questions;
    # putting the same estimator on both is the point of this figure.
    box = [auc(ctx.col(ctx.sub(ctx.cart, seed=s), 'decision_flip'),
               -ctx.col(ctx.sub(ctx.cart, seed=s), 'margin')) for s in SEEDS]
    groups = [('Hộp đều\nđặc quyền', box),
              ('Quỹ đạo\nđặc quyền',
               [auc(ctx.col(ctx.sub(ctx.hold, seed=s), 'decision_flip'),
                    -ctx.col(ctx.sub(ctx.hold, seed=s), 'true_margin')) for s in SEEDS]),
              ('Quỹ đạo\nkhông đặc quyền',
               [auc(ctx.col(ctx.sub(ctx.hold, seed=s), 'decision_flip'),
                    ctx.col(ctx.sub(ctx.hold, seed=s), 'unprivileged_score')) for s in SEEDS])]
    offsets = np.linspace(-.17, .17, len(SEEDS))
    for i, (label, data) in enumerate(groups):
        style = fs.series(0 if i == 2 else i)
        data = np.asarray(data, float)
        axes[1].scatter(np.full(len(data), i) + offsets, data, s=10,
                        facecolors='none', edgecolors=style['color'],
                        linewidths=.7, zorder=3)
        axes[1].plot([i - .28, i + .28], [data.mean()] * 2, color=style['color'],
                     linewidth=1.8, zorder=4)
    axes[1].axhline(.5, color='#555555', linestyle=':', linewidth=.8)
    axes[1].set(xticks=range(len(groups)),
                xticklabels=[label for label, _ in groups],
                ylabel='AUROC', ylim=(.45, 1.02), xlim=(-.55, len(groups) - .45))
    axes[1].tick_params(axis='x', which='minor', bottom=False, top=False,
                        labelsize=fs.SMALL_PT - 1.5)
    fs.panel_label(axes[1], '', '(b) Hộp đều so với phân phối thăm')
    fs.save(fig, 'cartpole_holdout', report)


def make_all(ctx, report=None):
    fs.use()
    for figure in [grid_error_margin, grid_matched_pairs, grid_layout,
                   cart_margin, cart_evaluation, baseline_learning,
                   fork_signed_direction, mujoco_signal_audit, cartpole_holdout]:
        figure(ctx, report)
