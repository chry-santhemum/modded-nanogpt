import re
import math
import matplotlib.pyplot as plt

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#000000",
    "#ff1493",
    "#00ced1",
    "#ffd700",
    "#4b0082",
]

runs = {
    #'Muon (old best, 3500 steps)': ('311d7833-8dfc-43ea-a55c-fd313a11c4a8', '#d04a1f'),
    'Muon (best, 3375 steps)': '51ece938-03c5-4343-8dcc-3f3336b07008',
    # 'AdamW (best, 5625 steps)': 'a63a68d1-24aa-4a22-af9a-224e43209ea4',
    # 'MuonH (best, 3325 steps)': '20260430_muonh/9319c798-6643-464a-b407-b05468e468f5',
    # 'AdamH (best, 4875 steps)': '20260430_adamh/7533dd87-107f-4a4f-8229-acbec0fb00ac',
    'Muon² (best, 3325 steps)': '20260501_muonsq/bb903816-ea27-4f5f-8028-c963d38c6a7f',
    # 'NorMuonH (best, 3250 steps)': '20260430_normuonh/f45b5dcf-16bb-4e83-b5c7-4ef4981f0e9f',
    # 'NorMuon + u/w-floor (best, 3250 steps)': '20260501_skylight001/f78af80a-2ba3-4cf7-b9f7-e6e56ff2c54d',
    # 'NorMuon (best, 3300 steps)': 'e0d0185f-ccb8-426d-8265-a4e762ec69f6',
    'Nor-Contra-Muon (best, 3225 steps)': '20260501_contra_muon/08cd60f9-99e2-4e28-b1ac-19136dd42a05',
    'FOOF (cooldown 0.5 to 2)': 'logs/20260503_111602_255438/20260503_115418_851187.txt',
}

assert len(runs) <= len(colors)
pattern = re.compile(r'step:(\d+)/(\d+)\s+val_loss:([0-9.]+)')
out = 'figure.png'

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(5.5, 4), dpi=600)

max_step = 0
results = []
for i, (label, logfile) in enumerate(runs.items()):
    steps, losses = [], []
    logfile = logfile.rstrip('.txt')
    if logfile.startswith("logs/"):
        path = f'/workspace/modded-nanogpt/{logfile}.txt'
    else:
        path = f'results/{logfile}.txt'
    with open(path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(3))
                if step == 0:
                    # there may be multiple runs in the logfile, take the last one
                    steps, losses = [], []
                steps.append(step)
                losses.append(loss)
    if not steps:
        raise RuntimeError(f'No loss curve found in {path}')

    max_step = max(max_step, max(steps))
    results.append((label, steps, losses, colors[i]))

    ax.plot(
        steps,
        losses,
        marker='o',
        markersize=1.5,
        linewidth=0.75,
        label=label,
        color=colors[i],
    )

ax.axhline(3.28, color='gray', linestyle='--', linewidth=1.0)
ax.annotate(
    'target=3.28',
    xy=(0, 3.28),
    xytext=(8, 6),
    textcoords='offset points',
    color='gray',
    fontsize=9,
)

ax.set_title('Modded-NanoGPT Optimization Benchmark as of 2026/05/03', pad=12, fontsize=12)
ax.set_xlabel('Training steps @ 0.5M bsz', fontsize=11)
ax.set_ylabel('Validation loss', fontsize=11)
ax.legend(
    frameon=True,
    fontsize=5,
    markerscale=0.5,
    handlelength=1.0,
    handletextpad=0.4,
    borderpad=0.3,
    labelspacing=0.25,
)
ax.set_xlim(0, math.ceil(max_step / 1000) * 1000)
ax.set_ylim(3.15, 4.0)
ax.tick_params(axis='both', which='major', labelsize=10)

fig.tight_layout()
fig.savefig(out, bbox_inches='tight')
print(out)

# Generate zoomed-in figure
zoom_min_step = 3000
zoom_max_step = 3500
zoom_losses = [
    loss
    for _, steps, losses, _ in results
    for step, loss in zip(steps, losses)
    if zoom_min_step <= step <= zoom_max_step
]

fig, ax = plt.subplots(figsize=(5.5, 4), dpi=600)
for label, steps, losses, color in results:
    ax.plot(
        steps,
        losses,
        marker='o',
        markersize=1.5,
        linewidth=0.75,
        label=label,
        color=color,
    )

ax.axhline(3.28, color='gray', linestyle='--', linewidth=1.0)
ax.annotate(
    'target=3.28',
    xy=(zoom_min_step, 3.28),
    xytext=(8, 6),
    textcoords='offset points',
    color='gray',
    fontsize=9,
)

ax.set_title('Modded-NanoGPT Optimization Benchmark as of 2026/05/03', pad=12, fontsize=12)
ax.set_xlabel('Training steps @ 0.5M bsz', fontsize=11)
ax.set_ylabel('Validation loss', fontsize=11)
ax.legend(
    frameon=True,
    fontsize=5,
    markerscale=0.5,
    handlelength=1.0,
    handletextpad=0.4,
    borderpad=0.3,
    labelspacing=0.25,
)
ax.set_xlim(zoom_min_step, zoom_max_step)
if zoom_losses:
    zoom_margin = 0.01
    ax.set_ylim(min(zoom_losses) - zoom_margin, max(zoom_losses) + zoom_margin)
ax.tick_params(axis='both', which='major', labelsize=10)

fig.tight_layout()
fig.savefig('zoomed_figure.png', bbox_inches='tight')
print('zoomed_figure.png')
