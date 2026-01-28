import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Style: sans-serif + transparent background
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "dejavusans"   # makes math match the sans font
mpl.rcParams["figure.facecolor"] = (0, 0, 0, 0)   # transparent
mpl.rcParams["savefig.facecolor"] = (0, 0, 0, 0)  # transparent
mpl.rcParams["savefig.transparent"] = True

# --- Equation (SLI)
eq = r"$\mathrm{SLI}=\mathrm{RPI}_{\mathrm{exp}}-\mathrm{RPI}_{\mathrm{yok}}$"

# Create a tiny canvas that will tightly fit the equation
fig, ax = plt.subplots(figsize=(6, 1.2), dpi=300)
ax.axis("off")

# Draw equation centered
ax.text(
    0.5, 0.5, eq,
    ha="center", va="center",
    fontsize=42,  # adjust to match your slide
    color="white" # change to "black" if you want dark text
)

# Tight crop
plt.tight_layout(pad=0)

out_path = "imgs/sli_equation.png"
fig.savefig(out_path, transparent=True, bbox_inches="tight", pad_inches=0)
plt.close(fig)

print(f"Saved: {out_path}")
