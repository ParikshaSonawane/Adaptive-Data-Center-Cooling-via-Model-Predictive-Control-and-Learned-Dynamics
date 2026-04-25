"""
dashboard.py
============
Interactive Streamlit dashboard for the Data Center Cooling simulation.

Features:
  🔥 Live animated heatmap of the data center grid
  🎛️ Control panel (controller selector, start/stop, speed)
  📊 Real-time temperature, energy, and reward plots
  📌 Key metrics panel
  🧪 Scenario comparison mode
"""

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import streamlit as st

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from env.cooling_env import (
    DataCenterEnv, TARGET_TEMP, MAX_TEMP, AMBIENT_TEMP, GRID_ROWS, GRID_COLS
)
from models.dynamics_model import (
    LinearDynamicsModel, LINEAR_PATH, NEURAL_PATH, NeuralDynamicsModel
)
from controllers.mpc_controller import get_controller


# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Center Cooling AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  .main { background: #0e1117; }

  /* Title block */
  .title-block {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 12px;
    padding: 20px 30px;
    margin-bottom: 20px;
    border-left: 4px solid #00d4ff;
  }
  .title-block h1 { color: #00d4ff; font-family: 'IBM Plex Mono', monospace;
                    font-size: 1.8rem; margin: 0; }
  .title-block p  { color: #8ab4c7; margin: 4px 0 0; font-size: 0.9rem; }

  /* Metric cards */
  .metric-card {
    background: #1a2332;
    border: 1px solid #2d3f55;
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
  }
  .metric-label { color: #6b8ca4; font-size: 0.75rem; text-transform: uppercase;
                  letter-spacing: 1px; font-family: 'IBM Plex Mono', monospace; }
  .metric-value { font-size: 2rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace;
                  line-height: 1.1; margin: 4px 0; }
  .metric-good   { color: #00d4ff; }
  .metric-warn   { color: #ffc107; }
  .metric-danger { color: #ff4444; }

  /* Status badge */
  .status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  .status-running { background: #0d3322; color: #00ff88; border: 1px solid #00ff88; }
  .status-stopped { background: #3a1a1a; color: #ff6666; border: 1px solid #ff6666; }
  .status-ready   { background: #1a2a3a; color: #66aaff; border: 1px solid #66aaff; }

  /* Section header */
  .section-header {
    color: #8ab4c7;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #2d3f55;
  }

  /* Controller comparison table */
  .compare-table { width: 100%; border-collapse: collapse; }
  .compare-table th { background: #1a2d42; color: #8ab4c7; padding: 10px;
                      font-size: 0.78rem; text-transform: uppercase;
                      font-family: 'IBM Plex Mono', monospace; border-bottom: 2px solid #2d3f55; }
  .compare-table td { padding: 10px 12px; border-bottom: 1px solid #1e2d3d;
                      font-family: 'IBM Plex Mono', monospace; font-size: 0.88rem; }
  .compare-table tr:hover td { background: #1a2332; }
  .good-val  { color: #00ff88; font-weight: 600; }
  .mid-val   { color: #ffc107; }
  .bad-val   { color: #ff4444; }

  div[data-testid="stExpander"] { border: 1px solid #2d3f55; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "running":       False,
        "step":          0,
        "env":           None,
        "controller":    None,
        "ctrl_name":     "none",
        "model":         None,
        "history":       {"temps": [], "energy": [], "rewards": [], "max_temps": [], "steps": []},
        "current_grid":  np.full((GRID_ROWS, GRID_COLS), TARGET_TEMP),
        "last_action":   np.zeros(4),
        "episode_done":  False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(kind: str = 'linear'):
    path = LINEAR_PATH if kind == 'linear' else NEURAL_PATH
    if os.path.exists(path):
        cls = LinearDynamicsModel if kind == 'linear' else NeuralDynamicsModel
        return cls.load(path)
    return None


def reset_simulation(ctrl_name: str, model_kind: str = 'linear', seed: int = 42):
    env = DataCenterEnv(seed=seed)
    model = load_model(model_kind)
    if ctrl_name == 'mpc' and model is None:
        st.warning("No trained dynamics model found yet. Falling back to Random Control.")
        ctrl_name = 'random'
    ctrl = get_controller(ctrl_name, dynamics_model=model, seed=seed)
    state = env.reset()
    ctrl.reset()

    st.session_state.env          = env
    st.session_state.controller   = ctrl
    st.session_state.ctrl_name    = ctrl_name
    st.session_state.model        = model
    st.session_state.step         = 0
    st.session_state.current_grid = env.get_grid()
    st.session_state.last_action  = np.zeros(4)
    st.session_state.episode_done = False
    st.session_state.history      = {"temps": [], "energy": [], "rewards": [], "max_temps": [], "steps": []}
    st.session_state._state        = state


def do_step():
    """Advance simulation by one step."""
    env  = st.session_state.env
    ctrl = st.session_state.controller
    if env is None or ctrl is None:
        return

    state  = st.session_state._state
    action = ctrl.select_action(state)
    next_state, reward, done, info = env.step(action)

    st.session_state._state       = next_state
    st.session_state.current_grid = env.get_grid()
    st.session_state.last_action  = action
    st.session_state.step        += 1
    st.session_state.episode_done = done

    h = st.session_state.history
    h["steps"].append(st.session_state.step)
    h["temps"].append(info["mean_temp"])
    h["max_temps"].append(info["max_temp"])
    h["energy"].append(info["energy"])
    h["rewards"].append(reward)


def temp_color_class(t: float) -> str:
    if t < TARGET_TEMP + 5:   return "metric-good"
    if t < TARGET_TEMP + 15:  return "metric-warn"
    return "metric-danger"


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap figure
# ─────────────────────────────────────────────────────────────────────────────
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "dc_heat",
    [(0,   "#1a3a5c"),   # cold – deep blue
     (0.3, "#00b4d8"),   # cool – cyan
     (0.5, "#06d6a0"),   # target – green
     (0.7, "#ffd166"),   # warm – amber
     (1.0, "#ef233c")],  # hot  – red
)

def make_heatmap(grid: np.ndarray, action: np.ndarray, step: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    im = ax.imshow(
        grid,
        cmap=CMAP,
        vmin=AMBIENT_TEMP,
        vmax=MAX_TEMP * 0.75,
        interpolation='bilinear',
        aspect='equal',
    )

    # Zone overlay borders
    half_r, half_c = GRID_ROWS // 2, GRID_COLS // 2
    zone_slices = [
        (0, 0, half_r, half_c),
        (0, half_c, half_r, GRID_COLS - half_c),
        (half_r, 0, GRID_ROWS - half_r, half_c),
        (half_r, half_c, GRID_ROWS - half_r, GRID_COLS - half_c),
    ]
    for z, (r0, c0, rh, cw) in enumerate(zone_slices):
        power = action[z] if len(action) > z else 0
        alpha = 0.08 + 0.25 * power
        rect = Rectangle((c0 - 0.5, r0 - 0.5), cw, rh,
                          linewidth=1.5, edgecolor='#00d4ff',
                          facecolor='#00d4ff', alpha=alpha)
        ax.add_patch(rect)
        ax.text(c0 + cw/2 - 0.5, r0 + rh/2 - 0.5,
                f"Z{z+1}\n{power:.0%}",
                color='white', fontsize=7.5, ha='center', va='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.4))

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Temperature (°C)", color='#8ab4c7', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='#8ab4c7')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8ab4c7', fontsize=8)
    cbar.outline.set_edgecolor('#2d3f55')

    ax.set_xticks(range(GRID_COLS))
    ax.set_yticks(range(GRID_ROWS))
    ax.set_xticklabels([str(i) for i in range(GRID_COLS)], color='#4a6070', fontsize=7)
    ax.set_yticklabels([str(i) for i in range(GRID_ROWS)], color='#4a6070', fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3f55')

    ax.set_title(f"Server Grid — Step {step}", color='#8ab4c7',
                 fontsize=10, fontfamily='monospace', pad=10)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Time-series chart
# ─────────────────────────────────────────────────────────────────────────────
def make_timeseries(history: dict) -> plt.Figure:
    steps   = history["steps"]
    if not steps:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    fig.patch.set_facecolor('#0e1117')

    plots = [
        ("temps",    "Mean Temp (°C)",   "#00b4d8", axes[0]),
        ("energy",   "Energy / Step",    "#ffc107", axes[1]),
        ("rewards",  "Reward",           "#06d6a0", axes[2]),
    ]

    for key, ylabel, color, ax in plots:
        vals = history[key]
        ax.set_facecolor('#111b27')
        ax.plot(steps, vals, color=color, linewidth=1.6, alpha=0.9)
        ax.fill_between(steps, vals, alpha=0.12, color=color)
        if key == "temps":
            ax.axhline(TARGET_TEMP, color='#ffffff', linewidth=0.8,
                       linestyle='--', alpha=0.4, label=f'Target {TARGET_TEMP}°C')
            ax.axhline(MAX_TEMP * 0.9, color='#ff4444', linewidth=0.8,
                       linestyle=':', alpha=0.5, label='Critical')
            ax.legend(fontsize=7, framealpha=0.3, labelcolor='white')
            ax.set_ylim(bottom=AMBIENT_TEMP - 2)
        ax.set_ylabel(ylabel, color='#8ab4c7', fontsize=8, fontfamily='monospace')
        ax.tick_params(colors='#4a6070', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2d3f55')
        ax.grid(axis='y', color='#2d3f55', linewidth=0.5, alpha=0.6)

    axes[2].set_xlabel("Simulation Step", color='#8ab4c7', fontsize=8, fontfamily='monospace')
    fig.tight_layout(h_pad=0.8)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Comparison run helper
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running scenario comparison …")
def run_comparison(steps: int = 150) -> pd.DataFrame:
    model = load_model('linear')
    rows  = []
    seed  = 0

    for ctrl_name in ['none', 'random', 'mpc']:
        if ctrl_name == 'mpc' and model is None:
            continue
        env   = DataCenterEnv(seed=seed)
        ctrl  = get_controller(ctrl_name, dynamics_model=model, seed=seed)
        state = env.reset(); ctrl.reset()
        temps, energies, rewards = [], [], []
        for _ in range(steps):
            action = ctrl.select_action(state)
            state, reward, done, info = env.step(action)
            temps.append(info["mean_temp"])
            energies.append(info["energy"])
            rewards.append(reward)
            if done: break
        rows.append({
            "Controller":   ctrl.name,
            "Avg Temp °C":  round(np.mean(temps), 2),
            "Max Temp °C":  round(np.max(temps),  2),
            "Temp Std":     round(np.std(temps),   3),
            "Total Energy": round(np.sum(energies), 1),
            "Avg Reward":   round(np.mean(rewards), 3),
            "_temps":       temps,
            "_energies":    energies,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">⚙️ Simulation Config</div>', unsafe_allow_html=True)

    ctrl_choice = st.selectbox(
        "Controller",
        ["No Control", "Random Control", "MPC Control"],
        index=2,
        help="Select the cooling strategy"
    )
    CTRL_MAP = {"No Control": "none", "Random Control": "random", "MPC Control": "mpc"}
    ctrl_key  = CTRL_MAP[ctrl_choice]

    model_choice = st.selectbox(
        "Dynamics Model",
        ["Linear (Ridge)", "Neural (MLP)"],
        help="Backbone model for MPC predictions"
    )
    model_key = 'linear' if 'Linear' in model_choice else 'neural'

    seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42)
    speed = st.slider("Speed (steps/sec)", 1, 20, 5)

    st.markdown('<div class="section-header" style="margin-top:16px">🎮 Controls</div>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    if col_a.button("▶ Start", use_container_width=True, type="primary"):
        if st.session_state.env is None or st.session_state.ctrl_name != ctrl_key:
            reset_simulation(ctrl_key, model_key, int(seed))
        st.session_state.running = True

    if col_b.button("⏹ Stop", use_container_width=True):
        st.session_state.running = False

    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.running = False
        reset_simulation(ctrl_key, model_key, int(seed))
        st.rerun()

    st.markdown('<div class="section-header" style="margin-top:16px">ℹ️ Info</div>',
                unsafe_allow_html=True)
    st.caption("""
    **No Control** – No cooling applied. Grid overheats quickly.

    **Random Control** – Random cooling each step. Unstable but prevents runaway heating.

    **MPC Control** – AI-optimised cooling using a learned model. Efficient and stable.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <h1>🏭 Data Center Cooling — Model Predictive Control</h1>
  <p>Real-time simulation of AI-driven thermal management using learned dynamics</p>
</div>
""", unsafe_allow_html=True)

# ── Metric bar ──────────────────────────────────────────────────────────────
h = st.session_state.history
mean_temp = h["temps"][-1]   if h["temps"]   else TARGET_TEMP
max_temp  = h["max_temps"][-1] if h["max_temps"] else TARGET_TEMP
energy    = h["energy"][-1]  if h["energy"]  else 0.0
reward    = h["rewards"][-1] if h["rewards"] else 0.0
step      = st.session_state.step
status_cls = "status-running" if st.session_state.running else (
             "status-stopped"  if step > 0 else "status-ready")
status_txt = "RUNNING" if st.session_state.running else (
             "STOPPED" if step > 0 else "READY")

def tc(v): return temp_color_class(v)

m1, m2, m3, m4, m5, m6 = st.columns(6)
for col, label, value, cls in [
    (m1, "Status",      status_txt,          f"status-badge {status_cls}"),
    (m2, "Step",        f"{step}",            "metric-value metric-good"),
    (m3, "Mean °C",     f"{mean_temp:.1f}",   f"metric-value {tc(mean_temp)}"),
    (m4, "Max °C",      f"{max_temp:.1f}",    f"metric-value {tc(max_temp)}"),
    (m5, "Energy",      f"{energy:.1f}",      "metric-value metric-warn"),
    (m6, "Reward",      f"{reward:.2f}",      "metric-value metric-good"),
]:
    if label == "Status":
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div style="margin-top:6px"><span class="{cls}">{value}</span></div></div>',
            unsafe_allow_html=True)
    else:
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="{cls}">{value}</div></div>',
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main content ─────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.markdown('<div class="section-header">🔥 Live Server Grid</div>', unsafe_allow_html=True)
    heatmap_placeholder = st.empty()
    fig_hm = make_heatmap(
        st.session_state.current_grid,
        st.session_state.last_action,
        st.session_state.step
    )
    heatmap_placeholder.pyplot(fig_hm, use_container_width=True)
    plt.close(fig_hm)

    # Zone power bars
    st.markdown('<div class="section-header" style="margin-top:12px">🌬️ Cooling Zone Power</div>',
                unsafe_allow_html=True)
    zone_cols = st.columns(4)
    for z, col in enumerate(zone_cols):
        pwr = float(st.session_state.last_action[z]) if len(st.session_state.last_action) > z else 0.0
        col.metric(f"Zone {z+1}", f"{pwr:.0%}", delta=None)

with right_col:
    st.markdown('<div class="section-header">📈 Time Series</div>', unsafe_allow_html=True)
    chart_placeholder = st.empty()

    if h["temps"]:
        fig_ts = make_timeseries(h)
        if fig_ts:
            chart_placeholder.pyplot(fig_ts, use_container_width=True)
            plt.close(fig_ts)
    else:
        chart_placeholder.info("Start the simulation to see live charts.")

# ── Scenario comparison ───────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📊  Controller Comparison (All Scenarios)", expanded=False):
    if st.button("▶ Run Comparison (150 steps each)"):
        run_comparison.clear()

    df = run_comparison(150)
    if df.empty:
        st.info("Train a dynamics model to include MPC in the comparison view.")
    else:
        display_df = df[["Controller","Avg Temp °C","Max Temp °C","Temp Std","Total Energy","Avg Reward"]]

        # Styled table
        def style_row(row):
            styles = []
            for col in row.index:
                if col == "Controller": styles.append(''); continue
                val = row[col]
                if col == "Total Energy":
                    c = '#00ff88' if val == df["Total Energy"].min() else ('#ffc107' if val < df["Total Energy"].max() else '#ff4444')
                elif col in ("Avg Temp °C", "Max Temp °C", "Temp Std"):
                    c = '#00ff88' if val == df[col].min() else ('#ffc107' if val < df[col].max() else '#ff4444')
                elif col == "Avg Reward":
                    c = '#00ff88' if val == df[col].max() else ('#ffc107' if val > df[col].min() else '#ff4444')
                else:
                    c = 'white'
                styles.append(f'color: {c}; font-family: monospace')
            return styles

        styled = display_df.style.apply(style_row, axis=1).set_properties(
            **{'background-color': '#111b27', 'border': '1px solid #2d3f55'})
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Overlay temperature chart
        fig_cmp, ax = plt.subplots(figsize=(9, 3.5))
        fig_cmp.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#111b27')
        colors = {'No Control': '#ff4444', 'Random Control': '#ffc107', 'MPC Control': '#00ff88'}
        for _, row in df.iterrows():
            ax.plot(row["_temps"], label=row["Controller"],
                    color=colors[row["Controller"]], linewidth=1.8, alpha=0.9)
        ax.axhline(TARGET_TEMP, color='white', linewidth=0.8, linestyle='--', alpha=0.4, label='Target')
        ax.set_xlabel("Step", color='#8ab4c7', fontsize=9)
        ax.set_ylabel("Mean Temperature (°C)", color='#8ab4c7', fontsize=9)
        ax.set_title("Temperature Trajectory — All Controllers", color='#8ab4c7',
                     fontsize=10, fontfamily='monospace')
        ax.legend(fontsize=9, framealpha=0.3, labelcolor='white')
        ax.tick_params(colors='#4a6070', labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor('#2d3f55')
        ax.grid(axis='y', color='#2d3f55', linewidth=0.5, alpha=0.6)
        fig_cmp.tight_layout()
        st.pyplot(fig_cmp, use_container_width=True)
        plt.close(fig_cmp)

# ─────────────────────────────────────────────────────────────────────────────
# Auto-advance loop
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.running and not st.session_state.episode_done:
    do_step()
    h = st.session_state.history

    # Refresh heatmap
    fig_hm2 = make_heatmap(
        st.session_state.current_grid,
        st.session_state.last_action,
        st.session_state.step
    )
    heatmap_placeholder.pyplot(fig_hm2, use_container_width=True)
    plt.close(fig_hm2)

    # Refresh chart
    if h["temps"]:
        fig_ts2 = make_timeseries(h)
        if fig_ts2:
            chart_placeholder.pyplot(fig_ts2, use_container_width=True)
            plt.close(fig_ts2)

    time.sleep(1.0 / speed)
    st.rerun()

elif st.session_state.episode_done:
    st.warning("⚠️  Episode ended (critical overheating detected). Press Reset to restart.")
