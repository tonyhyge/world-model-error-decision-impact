"""Shared IEEE-style figure grammar for every thesis figure.

All figures are drawn at the exact text width of the report so that LaTeX
includes them at 1:1 and no font is ever resampled. Text uses a Times-metric
serif with full Vietnamese coverage; maths uses the Times-like STIX set that
ships with matplotlib.
"""
import glob
import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'figures'

# \the\textwidth of main.tex (424.06033 TeX pt) measured with pdflatex.
TEXT_WIDTH_IN = 424.06033 / 72.27

# The report sets 12pt body text; IEEE keeps figure text near 0.8 of the body.
BASE_PT = 10.0
SMALL_PT = 9.0

# TeX Gyre Termes is a Times clone carrying the Vietnamese diacritics that the
# STIXGeneral shipped with matplotlib lacks. Anyone able to compile the report
# already has it, and the chain degrades to fonts present on macOS/Windows.
SERIF_STACK = ['TeX Gyre Termes', 'Times New Roman', 'Times', 'DejaVu Serif']

INK = '#1a1a1a'
GRID = '#c9c9c9'

# Okabe-Ito: colourblind-safe, and every series also carries its own dash
# pattern and marker so the figures survive greyscale printing.
PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00']
DASHES = ['-', '--', '-.', ':', (0, (5, 1, 1, 1)), (0, (3, 1, 3, 1, 1, 1))]
MARKERS = ['o', 's', '^', 'D', 'v', 'P']


def series(index):
    """Return the colour/linestyle/marker triple reserved for one series."""
    return dict(color=PALETTE[index % len(PALETTE)],
                linestyle=DASHES[index % len(DASHES)],
                marker=MARKERS[index % len(MARKERS)])


def _register_serif():
    """Make TeX Gyre Termes visible to matplotlib if a TeX Live tree exists."""
    pattern = '/usr/local/texlive/*/texmf-dist/fonts/opentype/public/tex-gyre/texgyretermes-*.otf'
    for path in sorted(glob.glob(pattern)):
        try:
            fm.fontManager.addfont(path)
        except Exception:  # A broken font must never block figure generation.
            pass


def use():
    """Install the shared rcParams. Safe to call from every script."""
    _register_serif()
    # A serif face with full Vietnamese coverage can still lack symbols such
    # as U+2032 and U+22EE, and matplotlib silently draws a blank box for
    # them. Fail loudly instead so no figure ships with a missing glyph.
    warnings.filterwarnings('error', message='Glyph .* missing from font')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': SERIF_STACK,
        'mathtext.fontset': 'stix',
        'font.size': BASE_PT,
        'axes.labelsize': BASE_PT,
        'axes.titlesize': BASE_PT,
        'xtick.labelsize': SMALL_PT,
        'ytick.labelsize': SMALL_PT,
        'legend.fontsize': SMALL_PT,
        'figure.titlesize': BASE_PT,
        # Closed frame on all four sides, as in the reference journal figures.
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.linewidth': 0.7,
        'axes.edgecolor': INK,
        'axes.labelcolor': INK,
        'text.color': INK,
        # Ticks point inward and are mirrored on the top and right spines.
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.color': INK,
        'ytick.color': INK,
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        # Solid light grid rather than a faint wash.
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.4,
        'grid.linestyle': '-',
        'grid.alpha': 1.0,
        # Boxed legend sitting inside the axes.
        'legend.frameon': True,
        'legend.framealpha': 1.0,
        'legend.edgecolor': INK,
        'legend.fancybox': False,
        'legend.borderpad': 0.35,
        'legend.labelspacing': 0.3,
        'legend.handlelength': 2.4,
        'lines.linewidth': 1.2,
        'lines.markersize': 4.0,
        'lines.markeredgewidth': 0.8,
        'patch.linewidth': 0.7,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 600,
        'figure.dpi': 150,
    })


def figure(height_in, width_frac=1.0, **kwargs):
    """Create a figure whose width is a fraction of the report text width."""
    return plt.subplots(figsize=(TEXT_WIDTH_IN * width_frac, height_in),
                        layout='constrained', **kwargs)


def legend(ax, **kwargs):
    """Boxed legend with the square corners used by the reference figures."""
    options = dict(frameon=True, framealpha=1.0, edgecolor=INK, fancybox=False)
    options.update(kwargs)
    handle = ax.legend(**options)
    handle.get_frame().set_linewidth(0.6)
    return handle


def panel_label(ax, xlabel, panel):
    """Set the axis label with an IEEE-style '(a) ...' caption beneath it.

    Folding the panel caption into the xlabel keeps it inside the layout
    engine's bookkeeping, so it can never collide with or overflow the figure.
    """
    ax.set_xlabel(f'{xlabel}\n\n{panel}' if xlabel else f'\n{panel}')


def zoom_inset(ax, bounds, xlim, ylim, draw, loc1=2, loc2=4):
    """Add a magnified inset over `xlim`/`ylim`, wired to the parent region.

    `bounds` is the inset rectangle in axes coordinates and `draw` receives the
    inset axes so the caller can replay the same artists at the zoomed scale.
    """
    axins = ax.inset_axes(bounds)
    draw(axins)
    axins.set(xlim=xlim, ylim=ylim)
    axins.tick_params(labelsize=SMALL_PT - 1.5, direction='in',
                      top=True, right=True)
    axins.grid(True, color=GRID, linewidth=0.35)
    for spine in axins.spines.values():
        spine.set_linewidth(0.6)
    mark_inset(ax, axins, loc1=loc1, loc2=loc2,
               facecolor='none', edgecolor=INK, linewidth=0.5, alpha=0.8)
    return axins


def save(fig, name, report=None):
    """Write the vector figure plus a raster preview, and mirror to the report."""
    FIGURES.mkdir(exist_ok=True)
    pdf = FIGURES / (name + '.pdf')
    fig.savefig(pdf)
    fig.savefig(FIGURES / (name + '.png'))
    plt.close(fig)
    if report:
        target = Path(report) / 'Figures'
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, target / (name + '.pdf'))
