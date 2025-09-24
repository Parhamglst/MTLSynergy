import re
import math
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Allow optional leading spaces before "Epoch"
EPOCH_RE = re.compile(r'^[^\S\r\n]*Epoch\s+(\d+):', re.MULTILINE)
VAL_RE = re.compile(r'Val Loss:\s*\[(.*?)\]', re.DOTALL)
WEIGHTED_RE = re.compile(r'Weighted Val Loss\s*=\s*([\-+0-9.eE]+)')

def parse_sessions(text: str):
    """
    Parse the log into sessions. A new session starts when epochs reset (e.g., Epoch 1)
    or when the epoch number decreases. Returns list of sessions; each session is dict with:
      - 'val_losses': list[list[float]]
      - 'weighted': list[float]
    """
    epoch_marks = list(EPOCH_RE.finditer(text))
    sessions = []
    current = {'val_losses': [], 'weighted': []}
    prev_epoch = None

    def flush():
        nonlocal current
        if current['val_losses']:
            sessions.append(current)
        current = {'val_losses': [], 'weighted': []}

    for i, m in enumerate(epoch_marks):
        epoch_num = int(m.group(1))
        start = m.end()
        end = epoch_marks[i + 1].start() if i + 1 < len(epoch_marks) else len(text)
        block = text[start:end]

        # Start new session if epochs reset or decrease
        if prev_epoch is not None and epoch_num <= prev_epoch and current['val_losses']:
            flush()
        prev_epoch = epoch_num

        # Extract first Val Loss array in this block
        val_match = VAL_RE.search(block)
        if not val_match:
            continue
        vals_str = val_match.group(1)
        nums = re.findall(r'[-+]?\d*\.?\d+(?:e[+-]?\d+)?', vals_str, flags=re.IGNORECASE)
        vals = [float(x) for x in nums] if nums else []
        if not vals:
            continue

        # Extract the first Weighted Val Loss in this block (optional)
        w_match = WEIGHTED_RE.search(block)
        weighted = float(w_match.group(1)) if w_match else None

        current['val_losses'].append(vals)
        if weighted is not None:
            current['weighted'].append(weighted)

    flush()
    return sessions

def plot_session(session, out_dir: Path, idx: int):
    val_losses = session['val_losses']
    weighted = session['weighted']
    if not val_losses:
        return

    epochs = list(range(1, len(val_losses) + 1))

    # Use the minimum number of components across epochs for consistent plotting
    num_components = min(len(v) for v in val_losses)
    val_losses = [v[:num_components] for v in val_losses]

    plt.style.use('seaborn-v0_8-darkgrid')

    ncols = 4 if num_components >= 4 else num_components
    nrows = math.ceil(num_components / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    axes = axes.flatten()

    for comp in range(num_components):
        series = [v[comp] for v in val_losses]
        ax = axes[comp]
        ax.plot(epochs, series, marker='o', linewidth=1.5)
        ax.set_title(f'Val Loss {comp + 1}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

    for j in range(num_components, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Session {idx} - Val Loss components')
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    png_path = out_dir / f'val_losses_session_{idx}.png'
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {png_path}')

    # Weighted Val Loss (plot whatever length is available)
    if weighted:
        k = min(len(weighted), len(epochs))
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(epochs[:k], weighted[:k], marker='o', color='tab:red', linewidth=1.8)
        ax2.set_title(f'Session {idx} - Weighted Val Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Weighted Val Loss')
        fig2.tight_layout()
        wpng = out_dir / f'weighted_val_loss_session_{idx}.png'
        fig2.savefig(wpng, dpi=150)
        plt.close(fig2)
        print(f'Wrote {wpng}')

def main():
    ap = argparse.ArgumentParser(description='Plot Val Losses per session from slurm log.')
    # Fix default: look in project root beside this script
    ap.add_argument('logfile', nargs='?', default=str(Path(__file__).resolve().parent / 'slurm-65944783-cleaned.out'),
                    help='Path to the slurm output log (default: ./slurm-65944783-cleaned.out)')
    ap.add_argument('--out', default='plots', help='Output directory for PNGs (default: ./plots)')
    args = ap.parse_args()

    log_path = Path(args.logfile)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        print(f'Log file not found: {log_path}')
        return

    text = log_path.read_text(encoding='utf-8', errors='ignore')
    sessions = parse_sessions(text)

    if not sessions:
        print('No sessions found.')
        return

    for i, sess in enumerate(sessions, start=1):
        plot_session(sess, out_dir, i)

    print(f'Done. Sessions plotted: {len(sessions)}')

if __name__ == '__main__':
    main()