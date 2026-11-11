# Requirements – `dashboard02_smc`

This file lists what you need to install and run the sliding mode control dashboard,
and which functional requirements it implements. The user guide is in
[Readme.md](Readme.md).

## 1. System

| Item | Requirement |
| --- | --- |
| Operating system | Windows 10/11, Linux or macOS |
| Python | **3.11 or newer** (the pinned NumPy 2.5, SciPy 1.18 and pandas 3.1 need ≥ 3.11). Tested with Python 3.12.3. |
| Browser | A recent Chrome, Edge, Firefox or Safari (the dashboard is a local web page) |
| Hardware | Any recent PC. One default run (280 s simulated, 50 Hz controller) usually takes 5–15 s on one CPU core; the app uses about 300–500 MB of RAM with several stored runs. |
| Network | Only `localhost`; no internet access is needed after installation. |

## 2. Python packages

`dashboard02_smc/requirements.txt` installs the simulator's pinned dependencies
(root `requirements.txt`) plus Streamlit and Plotly:

| Package | Version | Used for |
| --- | --- | --- |
| numpy | 2.5.4 | simulator and dashboard numerics |
| scipy | 1.18.1 | `solve_ivp` integration, moving-average filter of the chattering index |
| pandas | 3.1.0 | time series and tables |
| matplotlib | 3.11.2 | simulator plots (`plot_results.py`, not used by the dashboard) |
| streamlit | 1.66.1 | web user interface |
| plotly | 7.1.0 | interactive figures |
| other pinned packages | see root `requirements.txt` | dependencies of the above |

Streamlit and Plotly are only needed by the dashboards; the command-line simulator
(`python run_simulation.py`) works without them. The first dashboard (`dashboard/`)
uses the same versions, so one virtual environment serves both.

> The root `requirements.txt` is saved as UTF-16. pip reads it correctly; if another
> tool cannot, convert it to UTF-8.

## 3. Installation

Run these commands from the project root (the folder that contains `usv_model.py`).

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r dashboard02_smc\requirements.txt
```

**Linux / macOS**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r dashboard02_smc/requirements.txt
```

If the project already has a virtual environment with the first dashboard's
requirements installed, nothing else is needed.

## 4. Check the installation (optional)

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s dashboard02_smc\tests -t .
```

105 tests; they take about 2 minutes and end with `OK`.

## 5. Launch

```powershell
.\.venv\Scripts\python.exe -m dashboard02_smc            # or: streamlit run dashboard02_smc/app.py
```

Open <http://localhost:8501> if the browser does not open by itself. Useful options:

| Option | Effect |
| --- | --- |
| `--server.port 8602` | use another port (e.g. to run both dashboards at once) |
| `--server.headless true` | do not open a browser (e.g. on a remote machine) |

## 6. Functional requirements implemented

| # | Requirement | Where in the dashboard |
| --- | --- | --- |
| R1 | New, self-contained folder `dashboard02_smc` using the functional base simulator; Streamlit technology | this folder; uses `usv_model`, `controllers.smc_figure_eight`, `disturbances` |
| R2 | Separate sliding mode controllers for surge and yaw | sidebar section *3 - Controller* (surge \| yaw pairs) |
| R3 | Slider **or** input box for every parameter: $\lambda_i$, $k_{s,i}$, $\phi_i$, $k_i$ ($k_{pos}$, $k_\psi$) | *Sliders* toggle in the sidebar |
| R4 | Disturbances: none, constant or sinusoidal (plus the directional, wind-like and combined models of the first dashboard) | sidebar section *4 - Disturbance* |
| R5 | Figure-eight trajectory configurable in frequency and velocity | sidebar section 1 (frequency + peak speed; also amplitude-based modes) |
| R6 | Results processed only after pressing the simulation button | *Run simulation*; parameter changes only show a "settings changed" notice |
| R7 | Current run in a bright colour, previous run in pale gray, same axes and scales, legends with the parameters of each run | every figure; LaTeX run cards below the plots |
| R8 | XY trajectory plot | tab *XY trajectory* |
| R9 | Controller errors | tab *Controller errors* |
| R10 | Control inputs | tab *Control inputs* |
| R11 | s-surface in the phase plane ($\dot e$ vs $e$) with a shaded $\pm\phi_i$ boundary layer | tab *Phase plane (s-surface)* |
| R12 | Performance indices for the errors and for chattering | tab *Performance indices* |
| R13 | Equations in LaTeX, not plain text | sidebar, run cards, *Equations* tab, index definitions |
| R14 | No obstacles / barrier functions | not used (CBF and governor disabled) |
| R15 | `Requirements.md` and `Readme.md` for the users | this file and [Readme.md](Readme.md) |
| R16 | Estimate of the switching gains needed for the given disturbance (sliding condition), shown as a hint when the disturbances change | *Gain hint* boxes in *4 - Disturbance*, one-line check in *3 - Controller*; button *Use suggested* |
| R17 | Sidebar structure of the first dashboard (sections, surge \| yaw parameter pairs, all disturbance models), plus the time window, the phase plane and the gain hints | sidebar; *Time window of the plots*; tab *Phase plane (s-surface)* |
| R18 | Disturbance information in the XY trajectory, subtle; phase plane with the s-surface and time window in the main view | first tab *Trajectory & sliding surfaces* |

Additional features: digital (sampled) controller to study chattering, scenario
presets, design hints from SMC theory, initial offset (reaching phase), comparison
with any stored run, time window, JSON/CSV export, reproducible runs from Python.
