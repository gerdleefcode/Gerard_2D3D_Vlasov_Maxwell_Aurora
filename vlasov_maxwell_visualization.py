from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm
from matplotlib.patches import Circle


# ------------------------------------------------------------
# Visual style
# ------------------------------------------------------------
plt.style.use("dark_background")


# ------------------------------------------------------------
# Output location: Desktop
# ------------------------------------------------------------
def get_desktop_dir() -> Path:
    candidates = [
        Path.home() / "Desktop",
        Path.home() / "OneDrive" / "Desktop",
        Path.cwd(),
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path.home()


desktop_dir = get_desktop_dir()
gif_path = desktop_dir / "vlasov_maxwell_aurora.gif"
png_path = desktop_dir / "vlasov_maxwell_aurora_poster.png"


# ------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------
rng = np.random.default_rng(7)

xmin, xmax = -6.5, 6.5
ymin, ymax = -4.5, 4.5

N = 2200
dt = 0.02
substeps = 2
frames = 180

q_over_m = 0.9

# Particle state: solar-wind-like inflow + some energetic particles
x = rng.normal(loc=-4.8, scale=0.45, size=N)
y = rng.normal(loc=0.0, scale=1.25, size=N)

# A small population near the reconnection region
hot = N // 12
x[:hot] = rng.normal(loc=-1.3, scale=0.35, size=hot)
y[:hot] = rng.normal(loc=0.0, scale=0.8, size=hot)

vx = rng.normal(loc=2.2, scale=0.65, size=N)
vy = rng.normal(loc=0.0, scale=0.55, size=N)

# Cosmic-ray-like fast particles
hi = rng.choice(N, size=max(1, N // 35), replace=False)
vx[hi] += rng.uniform(1.5, 3.5, size=hi.size)
vy[hi] += rng.normal(0.0, 1.2, size=hi.size)


# ------------------------------------------------------------
# Field definitions
# ------------------------------------------------------------
def visual_b_field(X, Y):
    """
    Static field-line geometry for the main display.
    This is for the visual streamplot, not the Lorentz-force update.
    """
    eps = 0.25
    r2 = X**2 + Y**2 + eps

    bx = -Y / (r2 ** 1.15)
    by = X / (r2 ** 1.15)

    tail = np.exp(-((X + 2.2) ** 2) / (2 * 1.1**2) - Y**2 / (2 * 0.95**2))
    bx += 0.55 * tail * (-Y)
    by += 0.55 * tail * (X + 2.2)

    return bx, by


def em_fields(x, y, t):
    """
    Analytic electromagnetic fields for the particle pusher.

    Bz drives the in-plane gyration through v x B.
    Ex and Ey give acceleration, reconnection-like jets, and solar-wind forcing.
    """
    r2 = x**2 + y**2 + 0.55

    # Out-of-plane magnetic field
    Bz = 1.8 * (x**2 - y**2) / (r2**2)
    Bz += 1.6 * np.exp(-((x + 1.5) ** 2 + y**2) / (2 * 0.75**2)) * np.sin(1.5 * t + 2.3 * y)
    Bz += 0.35 * np.exp(-((x - 2.6) ** 2 + (y + 0.7) ** 2) / (2 * 1.6**2))

    # Electric field: solar-wind bias + reconnection region + wave-like perturbation
    Ex = -0.18 * np.exp(-((x + 4.2) ** 2 + y**2) / (2 * 2.6**2))
    Ex += 0.42 * np.exp(-((x + 1.35) ** 2 + y**2) / (2 * 0.85**2)) * (-y)

    Ey = 0.42 * np.exp(-((x + 1.35) ** 2 + y**2) / (2 * 0.85**2)) * (x + 1.35)
    Ey += 0.10 * np.sin(0.85 * x - 1.7 * t) * np.exp(-y**2 / 5.5)

    return Ex, Ey, Bz


# Grid for background fields
nx, ny = 220, 150
Xg, Yg = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))


# ------------------------------------------------------------
# Boris pusher
# ------------------------------------------------------------
def boris_push(x, y, vx, vy, t, dt):
    Ex, Ey, Bz = em_fields(x, y, t)

    v = np.column_stack((vx, vy, np.zeros_like(vx)))
    E = np.column_stack((Ex, Ey, np.zeros_like(Ex)))
    B = np.column_stack((np.zeros_like(Bz), np.zeros_like(Bz), Bz))

    v_minus = v + 0.5 * q_over_m * dt * E
    t_vec = 0.5 * q_over_m * dt * B

    t_mag2 = np.sum(t_vec * t_vec, axis=1, keepdims=True)
    s_vec = 2.0 * t_vec / (1.0 + t_mag2)

    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime, s_vec)
    v_new = v_plus + 0.5 * q_over_m * dt * E

    x_new = x + v_new[:, 0] * dt
    y_new = y + v_new[:, 1] * dt

    return x_new, y_new, v_new[:, 0], v_new[:, 1]


def reinject_particles(x, y, vx, vy):
    """
    When particles leave the simulation box, re-inject them from the solar-wind side.
    """
    mask = (x < xmin - 0.5) | (x > xmax + 0.5) | (y < ymin - 0.5) | (y > ymax + 0.5)
    idx = np.where(mask)[0]
    n = idx.size
    if n == 0:
        return

    x[idx] = rng.normal(loc=-5.1, scale=0.35, size=n)
    y[idx] = rng.normal(loc=0.0, scale=1.25, size=n)
    vx[idx] = rng.normal(loc=2.15, scale=0.7, size=n)
    vy[idx] = rng.normal(loc=0.0, scale=0.55, size=n)

    # occasional bursty energetic injection
    burst = rng.random(n) < 0.18
    if np.any(burst):
        burst_idx = idx[burst]
        vx[burst_idx] += rng.uniform(1.5, 3.5, size=burst_idx.size)
        vy[burst_idx] += rng.normal(0.0, 1.1, size=burst_idx.size)


# ------------------------------------------------------------
# Figure layout
# ------------------------------------------------------------
fig = plt.figure(figsize=(15, 8), facecolor="#050816")
gs = fig.add_gridspec(
    2,
    2,
    width_ratios=[1.7, 1.0],
    height_ratios=[1.0, 1.0],
    wspace=0.18,
    hspace=0.22,
)

ax_main = fig.add_subplot(gs[:, 0])
ax_phase = fig.add_subplot(gs[0, 1])
ax_energy = fig.add_subplot(gs[1, 1])

for ax in (ax_main, ax_phase, ax_energy):
    ax.set_facecolor("#050816")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#8fb3ff")

ax_main.set_xlim(xmin, xmax)
ax_main.set_ylim(ymin, ymax)
ax_main.set_aspect("equal", adjustable="box")
ax_main.set_xlabel("x")
ax_main.set_ylabel("y")

ax_phase.set_xlabel("x")
ax_phase.set_ylabel("vx")
ax_phase.set_title("Distribution function f(x, vx)")

ax_energy.set_xlabel("Frame")
ax_energy.set_ylabel("Mean kinetic energy")
ax_energy.set_title("Energy growth")


# ------------------------------------------------------------
# Background fields
# ------------------------------------------------------------
Bz0 = em_fields(Xg, Yg, 0.0)[2]
bz_lim = max(2.0, np.percentile(np.abs(Bz0), 99))
bz_norm = TwoSlopeNorm(vmin=-bz_lim, vcenter=0.0, vmax=bz_lim)

bz_img = ax_main.imshow(
    Bz0,
    extent=[xmin, xmax, ymin, ymax],
    origin="lower",
    cmap="RdBu_r",
    norm=bz_norm,
    alpha=0.42,
    interpolation="bilinear",
    zorder=0,
)

bx_vis, by_vis = visual_b_field(Xg, Yg)
ax_main.streamplot(
    Xg,
    Yg,
    bx_vis,
    by_vis,
    color=(1.0, 1.0, 1.0, 0.22),
    density=1.45,
    linewidth=0.7,
    arrowsize=0.8,
    minlength=0.15,
    zorder=1,
)

# Earth-like body
earth = Circle((0.35, 0.0), 0.55, color="#09192e", ec="#88c0ff", lw=2.0, zorder=4)
atm = Circle((0.35, 0.0), 0.82, fill=False, ec="#4fc3ff", lw=1.0, alpha=0.28, zorder=3)
ax_main.add_patch(atm)
ax_main.add_patch(earth)
ax_main.text(0.35, -0.92, "Earth", color="white", ha="center", va="top", fontsize=10, zorder=5)


# Particle density glow
dens0, _, _ = np.histogram2d(x, y, bins=(160, 100), range=[[xmin, xmax], [ymin, ymax]])
dens_norm = PowerNorm(gamma=0.45)
dens_img = ax_main.imshow(
    dens0.T,
    extent=[xmin, xmax, ymin, ymax],
    origin="lower",
    cmap="magma",
    norm=dens_norm,
    alpha=0.55,
    interpolation="bilinear",
    zorder=2,
)


# Particle scatter
speed0 = np.hypot(vx, vy)
sc = ax_main.scatter(
    x,
    y,
    c=speed0,
    s=9,
    cmap="turbo",
    norm=Normalize(vmin=0.0, vmax=7.5),
    edgecolors="none",
    alpha=0.95,
    zorder=3,
)

# Phase-space panel
vmin_phase, vmax_phase = -3.5, 8.5
phase_hist0, _, _ = np.histogram2d(
    x,
    vx,
    bins=(120, 120),
    range=[[xmin, xmax], [vmin_phase, vmax_phase]],
)

phase_img = ax_phase.imshow(
    phase_hist0.T,
    extent=[xmin, xmax, vmin_phase, vmax_phase],
    origin="lower",
    aspect="auto",
    cmap="viridis",
    norm=PowerNorm(gamma=0.45),
    interpolation="bilinear",
)

ax_phase.set_xlim(xmin, xmax)
ax_phase.set_ylim(vmin_phase, vmax_phase)

phase_text = ax_phase.text(
    0.02,
    0.98,
    "",
    transform=ax_phase.transAxes,
    ha="left",
    va="top",
    color="white",
    fontsize=9,
    bbox=dict(facecolor=(0, 0, 0, 0.35), edgecolor="none", boxstyle="round,pad=0.3"),
)


# Energy panel
history_frame = []
history_energy = []
energy_line, = ax_energy.plot([], [], color="#7cffcb", lw=2.2)
energy_dot, = ax_energy.plot([], [], "o", color="#ffcc66", ms=6)

ax_energy.set_xlim(0, frames)
ax_energy.set_ylim(0, 10.0)


# Title and status text
fig.suptitle(
    "Vlasov-Maxwell-inspired plasma visualisation: solar wind, auroras, energetic bursts",
    color="white",
    fontsize=15,
    y=0.98,
)

main_text = ax_main.text(
    0.02,
    0.98,
    "",
    transform=ax_main.transAxes,
    ha="left",
    va="top",
    color="white",
    fontsize=10,
    bbox=dict(facecolor=(0, 0, 0, 0.35), edgecolor="none", boxstyle="round,pad=0.35"),
)


# Save a poster frame immediately
fig.savefig(png_path, dpi=220, facecolor=fig.get_facecolor())
print(f"Saved poster image to: {png_path}")


# ------------------------------------------------------------
# Animation update
# ------------------------------------------------------------
sim_state = {"t": 0.0}


def update(frame):
    for _ in range(substeps):
        x_new, y_new, vx_new, vy_new = boris_push(x, y, vx, vy, sim_state["t"], dt)

        x[:] = x_new
        y[:] = y_new
        vx[:] = vx_new
        vy[:] = vy_new

        reinject_particles(x, y, vx, vy)
        sim_state["t"] += dt

    # Background magnetic field
    Bz = em_fields(Xg, Yg, sim_state["t"])[2]
    bz_img.set_data(Bz)

    # Density glow
    dens, _, _ = np.histogram2d(x, y, bins=(160, 100), range=[[xmin, xmax], [ymin, ymax]])
    dens_img.set_data(dens.T)
    dens_img.set_clim(0, max(1.0, np.percentile(dens, 99.5)))

    # Scatter particles
    speed = np.hypot(vx, vy)
    sc.set_offsets(np.column_stack((x, y)))
    sc.set_array(speed)

    # Phase-space f(x, vx)
    hist, _, _ = np.histogram2d(
        x,
        vx,
        bins=(120, 120),
        range=[[xmin, xmax], [vmin_phase, vmax_phase]],
    )
    phase_img.set_data(hist.T)
    phase_img.set_clim(0, max(1.0, np.percentile(hist, 99.0)))

    # Energy plot
    mean_e = 0.5 * np.mean(vx**2 + vy**2)
    history_frame.append(frame)
    history_energy.append(mean_e)
    energy_line.set_data(history_frame, history_energy)
    energy_dot.set_data([frame], [mean_e])

    ax_energy.set_ylim(0, max(10.0, 1.25 * max(history_energy)))

    # Text updates
    main_text.set_text(
        f"t = {sim_state['t']:.2f}\n"
        f"mean speed = {np.mean(speed):.2f}\n"
        f"max speed = {np.max(speed):.2f}"
    )
    phase_text.set_text(
        f"N = {N}\n"
        f"<vx> = {np.mean(vx):.2f}\n"
        f"current mean energy = {mean_e:.2f}"
    )

    return (
        bz_img,
        dens_img,
        sc,
        phase_img,
        energy_line,
        energy_dot,
        main_text,
        phase_text,
    )


anim = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=33,
    blit=False,
    repeat=False,
)

# ============================================================
# 3D ROTATING AURORA / MAGNETOSPHERE VISUALIZATION
# Paste this BEFORE the final plt.show()
# ============================================================

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import LinearSegmentedColormap

gif3d_path = desktop_dir / "vlasov_maxwell_aurora_3d.gif"
png3d_path = desktop_dir / "vlasov_maxwell_aurora_3d_poster.png"

aurora_cmap = LinearSegmentedColormap.from_list(
    "aurora",
    ["#03111f", "#08f7b0", "#71f0ff", "#c86dff"]
)

# 3D scene
fig3 = plt.figure(figsize=(11, 11), facecolor="#02040b")
ax3 = fig3.add_subplot(111, projection="3d")
fig3.patch.set_facecolor("#02040b")
ax3.set_facecolor("#02040b")
ax3.set_axis_off()
ax3.set_box_aspect((1, 1, 1))
ax3.set_xlim(-3.8, 3.8)
ax3.set_ylim(-3.8, 3.8)
ax3.set_zlim(-3.8, 3.8)
ax3.view_init(elev=22, azim=35)

# Earth sphere
u = np.linspace(0, 2 * np.pi, 64)
v = np.linspace(0, np.pi, 34)
earth_r = 0.98
xs = earth_r * np.outer(np.cos(u), np.sin(v))
ys = earth_r * np.outer(np.sin(u), np.sin(v))
zs = earth_r * np.outer(np.ones_like(u), np.cos(v))

ax3.plot_surface(
    xs,
    ys,
    zs,
    color="#0a1d35",
    edgecolor="none",
    linewidth=0,
    shade=True,
    alpha=1.0,
)

# Atmosphere shell
ax3.plot_surface(
    1.05 * xs,
    1.05 * ys,
    1.05 * zs,
    color="#4fc3ff",
    edgecolor="none",
    linewidth=0,
    shade=False,
    alpha=0.05,
)

# Magnetic dipole field lines
phi_values = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
L_shells = [1.25, 1.55, 1.9, 2.3, 2.75]
field_lines = []

theta_line = np.linspace(0.18, np.pi - 0.18, 220)
for L in L_shells:
    for phi in phi_values:
        r = L * (np.sin(theta_line) ** 2)
        x_line = r * np.sin(theta_line) * np.cos(phi)
        y_line = r * np.sin(theta_line) * np.sin(phi)
        z_line = r * np.cos(theta_line)

        line, = ax3.plot(
            x_line,
            y_line,
            z_line,
            color=(0.55, 0.78, 1.0, 0.18),
            lw=1.0,
        )
        field_lines.append(line)

# Solar-wind-like tail / current sheet
tail_x = np.linspace(-3.4, -0.55, 220)
tail_y = 0.08 * np.sin(4.0 * tail_x)
tail_z = 0.12 * np.sin(2.5 * tail_x + 0.5)
tail_line, = ax3.plot(
    tail_x,
    tail_y,
    tail_z,
    color=(0.7, 0.95, 1.0, 0.12),
    lw=1.0,
)

# Starfield
n_stars = 320
star_phi = rng.uniform(0, 2 * np.pi, n_stars)
star_cost = rng.uniform(-1, 1, n_stars)
star_theta = np.arccos(star_cost)
star_r = rng.uniform(8.0, 11.0, n_stars)

star_x = star_r * np.sin(star_theta) * np.cos(star_phi)
star_y = star_r * np.sin(star_theta) * np.sin(star_phi)
star_z = star_r * np.cos(star_theta)

ax3.scatter(
    star_x,
    star_y,
    star_z,
    s=rng.uniform(2, 8, n_stars),
    color="white",
    alpha=0.22,
    depthshade=False,
    linewidths=0,
)

# Aurora points: northern lights around the north polar oval
n_aurora = 1400
phi_a = rng.uniform(0, 2 * np.pi, n_aurora)
base_theta = np.deg2rad(15) + rng.normal(0, np.deg2rad(2.5), n_aurora)
base_r = 1.22 + rng.normal(0, 0.05, n_aurora)
phase_a = rng.uniform(0, 2 * np.pi, n_aurora)

# Initial aurora geometry
theta_a = base_theta + 0.05 * np.sin(3.0 * phi_a + phase_a)
r_a = base_r + 0.08 * np.sin(2.0 * phi_a + 0.3 * phase_a)

aur_x = r_a * np.sin(theta_a) * np.cos(phi_a)
aur_y = r_a * np.sin(theta_a) * np.sin(phi_a)
aur_z = r_a * np.cos(theta_a)

aurora_intensity = 0.25 + 0.75 * (0.5 + 0.5 * np.sin(3.0 * phi_a + phase_a))

aurora_scatter = ax3.scatter(
    aur_x,
    aur_y,
    aur_z,
    c=aurora_intensity,
    cmap=aurora_cmap,
    s=12,
    alpha=0.92,
    depthshade=False,
    linewidths=0,
)

# Optional faint southern aurora glow for symmetry
n_south = 450
phi_s = rng.uniform(0, 2 * np.pi, n_south)
south_theta = np.pi - (np.deg2rad(15) + rng.normal(0, np.deg2rad(2.0), n_south))
south_r = 1.18 + rng.normal(0, 0.04, n_south)
south_phase = rng.uniform(0, 2 * np.pi, n_south)

south_x = south_r * np.sin(south_theta) * np.cos(phi_s)
south_y = south_r * np.sin(south_theta) * np.sin(phi_s)
south_z = south_r * np.cos(south_theta)
south_intensity = 0.12 + 0.22 * (0.5 + 0.5 * np.sin(2.0 * phi_s + south_phase))

south_scatter = ax3.scatter(
    south_x,
    south_y,
    south_z,
    c=south_intensity,
    cmap=aurora_cmap,
    s=8,
    alpha=0.24,
    depthshade=False,
    linewidths=0,
)

# Text overlay
info3d = ax3.text2D(
    0.02,
    0.96,
    "",
    transform=ax3.transAxes,
    color="white",
    fontsize=10,
    bbox=dict(facecolor=(0, 0, 0, 0.35), edgecolor="none", boxstyle="round,pad=0.3"),
)

fig3.suptitle(
    "3D Rotating Aurora / Magnetosphere",
    color="white",
    fontsize=16,
    y=0.93,
)

# Save poster frame
fig3.savefig(png3d_path, dpi=220, facecolor=fig3.get_facecolor())
print(f"Saved 3D poster image to: {png3d_path}")

# Animation
t3 = {"time": 0.0}

def update_3d(frame):
    t = t3["time"]
    t3["time"] += 0.05

    # Gentle camera rotation
    ax3.view_init(
        elev=22 + 6 * np.sin(0.3 * t),
        azim=35 + 1.5 * frame,
    )

    # Aurora motion
    theta_a = (
        base_theta
        + 0.06 * np.sin(4.0 * phi_a + 1.4 * t + phase_a)
        + 0.015 * np.sin(11.0 * phi_a - 0.8 * t)
    )
    r_a = base_r + 0.10 * np.sin(2.0 * phi_a - 1.1 * t + 0.5 * phase_a)

    aur_x = r_a * np.sin(theta_a) * np.cos(phi_a)
    aur_y = r_a * np.sin(theta_a) * np.sin(phi_a)
    aur_z = r_a * np.cos(theta_a)

    aurora_intensity = (
        0.20
        + 0.80
        * (0.5 + 0.5 * np.sin(3.0 * phi_a - 1.2 * t + phase_a))
        * np.clip((aur_z - 0.5) / 1.0, 0.0, 1.0)
    )

    aurora_scatter._offsets3d = (aur_x, aur_y, aur_z)
    aurora_scatter.set_array(aurora_intensity)

    # Subtle movement for southern glow too
    south_theta = np.pi - (
        np.deg2rad(15)
        + 0.03 * np.sin(3.0 * phi_s + 1.1 * t + south_phase)
    )
    south_r = 1.18 + 0.05 * np.sin(2.0 * phi_s - 0.9 * t)

    south_x = south_r * np.sin(south_theta) * np.cos(phi_s)
    south_y = south_r * np.sin(south_theta) * np.sin(phi_s)
    south_z = south_r * np.cos(south_theta)

    south_intensity = 0.10 + 0.25 * (0.5 + 0.5 * np.sin(2.0 * phi_s + south_phase - 0.6 * t))
    south_scatter._offsets3d = (south_x, south_y, south_z)
    south_scatter.set_array(south_intensity)

    info3d.set_text(
        f"t = {t:.2f}\n"
        f"Camera azimuth = {35 + 1.5 * frame:.1f}°\n"
        f"Aurora intensity = {np.mean(aurora_intensity):.2f}"
    )

    return aurora_scatter, south_scatter, info3d

anim3 = FuncAnimation(
    fig3,
    update_3d,
    frames=150,
    interval=33,
    blit=False,
    repeat=False,
)

try:
    writer = PillowWriter(fps=24)
    anim3.save(gif3d_path, writer=writer, dpi=140)
    print(f"Saved 3D animation to: {gif3d_path}")
except Exception as e:
    print("3D GIF save failed.")
    print("Error:", e)
    print("Try: pip install pillow")

# ------------------------------------------------------------
# Save animation to Desktop
# ------------------------------------------------------------
try:
    writer = PillowWriter(fps=24)
    anim.save(gif_path, writer=writer, dpi=140)
    print(f"Saved animation to: {gif_path}")
except Exception as e:
    print("GIF save failed.")
    print("Error:", e)
    print("You may need: pip install pillow")
    print("The PNG poster image was still saved successfully.")


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()