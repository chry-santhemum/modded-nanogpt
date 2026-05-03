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
    'MuonH (best, 3325 steps)': '20260430_muonh/9319c798-6643-464a-b407-b05468e468f5',
    # 'AdamH (best, 4875 steps)': '20260430_adamh/7533dd87-107f-4a4f-8229-acbec0fb00ac',
    'Muon² (best, 3325 steps)': '20260501_muonsq/bb903816-ea27-4f5f-8028-c963d38c6a7f',
    'NorMuonH (best, 3275 steps)': '20260430_normuonh/f45b5dcf-16bb-4e83-b5c7-4ef4981f0e9f',
    # 'FOOF (default)': 'c718fa98-d45f-4b24-a01f-a6a0559c6a31',
    # 'FOOF (cooldown 0.5)': 'logs/20260503_035759_598139/20260503_035812_697861',
    # 'FOOF (beta 0.9)': 'logs/20260503_041609_179584/20260503_041622_100851',
    # 'FOOF (beta 0.99)': 'logs/20260503_041609_179584/20260503_043431_070366.txt',
    # 'FOOF (cooldown 0.5 both)': 'logs/20260503_045252_672097/20260503_045305_632893.txt',
    'FOOF (best, lr 0.035)': 'logs/20260503_151004_618338/20260503_151016_365663.txt',
    'FOOF (best, lr 0.05)': 'logs/20260503_151004_618338/20260503_152922_930647.txt',
    'FOOF (first_fw)': 'logs/20260503_194643_272898/20260503_194655_656643.txt',
}
assert len(runs) <= len(colors)
pattern = re.compile(r'step:(\d+)/(\d+)\s+val_loss:([0-9.]+)')
out = 'figure.png'

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(5.5, 4), dpi=600)

max_step = 0
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

ax.set_title('Modded-NanoGPT Optimization Benchmark as of 2026/05/01', pad=12, fontsize=12)
ax.set_xlabel('Training steps @ 0.5M bsz', fontsize=11)
ax.set_ylabel('Validation loss', fontsize=11)
ax.legend(frameon=True)
ax.set_xlim(0, math.ceil(max_step / 1000) * 1000)
ax.set_ylim(3.15, 4.0)
ax.tick_params(axis='both', which='major', labelsize=10)

fig.tight_layout()
fig.savefig(out, bbox_inches='tight')
print(out)
