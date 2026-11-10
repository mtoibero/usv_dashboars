# IAcquaBot USV simulation dashboard

Interactive Streamlit front end to configure, run and compare trajectory-tracking
simulations of the IAcquaBot USV **without editing the simulator sources**.
It drives the existing simulation engine (`usv_model.simulate`, the controllers in
`controllers.py` and the disturbances in `disturbances.py`); no dynamics or control
law is re-implemented. Scope: trajectory tracking under disturbances. Obstacles, CBF
filtering and the reference governor are intentionally not part of the dashboard.

## Installation

From the project root, in the same virtual environment as the simulator
(Windows PowerShell example, as in the main README):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r dashboard\requirements.txt
```

Linux/macOS: `python -m pip install -r dashboard/requirements.txt`.

`dashboard/requirements.txt` includes the simulator's pinned `requirements.txt` and
adds `streamlit==1.66.1` and `plotly==7.1.0`. The command-line simulator does not need
these two packages.

## Launching

From the project root, either command works:

```powershell
.\.venv\Scripts\python.exe -m dashboard      # uses this interpreter's Streamlit
streamlit run dashboard/app.py               # equivalent
```

The browser opens at <http://localhost:8501>. Streamlit options can be appended,
e.g. `python -m dashboard --server.port 8600`. Stop with `Ctrl+C`.

The command-line simulator is unchanged: `python run_simulation.py` still runs the
full default scenario (obstacles, CBF, governor, CSV, figures, GIF).

## Using the dashboard

1. **Configure** in the sidebar groups: *Simulation settings*, *Reference trajectory*,
   *Controller* and *Disturbance*. Only the parameters of the selected reference,
   controller and disturbance are shown. Changing a widget never runs a simulation.
2. The settings are **validated live**. Errors are listed under the button and disable
   **Run simulation**; warnings (e.g. a reference faster than the thrusters can
   sustain) are shown but do not block.
3. **Run simulation** takes an immutable snapshot of the settings, removes the previous
   results from the screen, and shows a spinner with elapsed time. The default
   280 s SMC run takes about 15 s.
4. Results appear in tabs: *XY trajectory*, *Tracking errors*, *Vessel states*,
   *Control signals* (commanded vs delivered Tu/Tr, the four thruster forces, PWM),
   *Disturbance*, *Metrics* (pass/fail indicators with an adjustable tolerance),
   *Compare runs* and *Run details & export*.
5. If you change any setting after a run, a **yellow banner** says the displayed
   results belong to *Run #N* and lists the changed parameters. A green banner
   confirms that results and sidebar agree.
6. **Previous run overlay.** The *XY trajectory* tab draws the previous run in light
   gray, under the dark dashed reference and the current run (original colours), in
   the same equal-scale axes. Only these two runs are drawn. If the previous run used a
   different reference, its path is added as a gray dotted line. Below the plot, the
   previous run's description, changed parameters, and main metrics are shown next to
   the current ones. A checkbox hides the overlay. Nothing changes in the run history.
7. **Comparison.** Every completed run is kept in the session (last 8) with its own
   configuration. The metric cards show the change against the previous run. *Compare
   runs* lets you pick any two stored runs and shows the configuration differences,
   a side-by-side metrics table, overlaid XY trajectories and overlaid errors and
   control efforts.
8. **Reproducibility.** *Run details & export* shows the exact configuration, lets you
   download it as JSON together with the time series as CSV, and can load any stored
   run's settings back into the sidebar.

### Parameters exposed

| Group | Parameters (units) | Default / source |
| --- | --- | --- |
| Simulation | total time: `laps x period` or custom [s]; output interval dt [s]; initial state on the reference or custom x0, y0 [m], psi0 [deg], u0, v0 [m/s], r0 [deg/s] | 280 s, `DT = 0.02`, `figure_eight_initial_state()` |
| Reference | type (figure-eight); amplitude A [m]; period T [s] **or** desired peak speed [m/s]; number of laps | `FIGURE_EIGHT_*` constants |
| SMC | lookahead L [m]; k_pos, k_psi [1/s]; reaching gains lambda_u, lambda_r [1/s]; switching gains k_s,u [m/s^2], k_s,r [rad/s^2]; boundary layers phi_u [m/s], phi_r [rad/s] | `smc_figure_eight()` signature |
| Simple feedback | k_pos, k_psi [1/s]; k_u [N s/m]; k_r [N m s/rad] | gains of `tau_figure_eight` |
| Disturbance | none; constant dTu, dTv [N], dTr [N m]; sinusoidal A, f [Hz], phase [rad], bias per axis; directional magnitude [N] + direction [deg]; wind-like (fixed profile); combined (wind + directional + sinusoidal) | `DISTURBANCE_OPTIONS` in `run_simulation.py` |

Notation: equations, widget labels, tooltips and validation messages use LaTeX
(rendered by Streamlit with KaTeX), e.g. $v_{cmd}$, $\lambda_u$, $k_{s,u}$, $\phi_u$. Vectors
are bold ($\boldsymbol{p}_s$), disturbances are $\delta T_u, \delta T_v, \delta T_r$ (columns `dTu`, `dTv`, `dTr`
in code and CSV). Plotly charts cannot render LaTeX inside Streamlit, so plot labels use
the same symbols with HTML subscripts and Unicode Greek letters (e<sub>s</sub>, ψ<sub>ref</sub>). Tables
use plain text. The sidebar opens 400 px wide so that the controller equations fit;
it can be resized by dragging its edge.

The dt setting only changes the output resolution. The integrator is adaptive (RK45,
rtol 1e-8) and evaluates the controller continuously, as in the CLI.

## Architecture

```
dashboard/
  app.py        Streamlit UI only: widgets, session state, layout      (imports streamlit)
  config.py     SimulationConfig (frozen dataclasses), defaults read from the simulator,
                validation, form <-> config conversion, config diff
  engine.py     config -> existing controller/disturbance factories -> usv_model.simulate
                -> run_simulation.controller_output_history / reference_history -> DataFrame
  metrics.py    error columns and metrics (reuses analysis_utils), pass/fail indicators
  figures.py    Plotly figures built from stored results
  session.py    bounded in-memory run history and staleness check
  tests/        unit tests + Streamlit AppTest end-to-end tests
```

Data flow of one run:

```
widgets -> session_state -> form_to_config() -> validate_config() -> [Run]
        -> run_from_config(snapshot) -> SimulationResult(config, data, metrics)
        -> RunHistory -> figures / metrics / comparison
```

Design decisions:

* **GUI separation.** Only `app.py` imports Streamlit. `engine.run_from_config()` can
  be used from scripts and notebooks (see *Reproduce this run from Python* in the app).
* **Snapshots.** `SimulationConfig` is frozen and holds the *effective* values in SI
  units (resolved duration, initial state and period). Each result keeps the config
  object that produced it.
* **No cached simulations.** Every click integrates again. Results are identified by a
  run number and a unique run id. The staleness banner compares the displayed run's
  config with the current sidebar values.
* **Defaults follow the simulator.** They are read from `smc_figure_eight`'s signature,
  `FIGURE_EIGHT_*`, `TRACKING_LOOKAHEAD`, `DT` and `DISTURBANCE_MODE`. The
  disturbance presets mirror `DISTURBANCE_OPTIONS`, and a test checks they are
  identical.

### Changes made to the existing simulator

All changes are backward-compatible: the new arguments are optional keyword arguments
whose defaults are the previous constants.

* `controllers.py`: `figure_eight_lookahead_reference`, `figure_eight_initial_state`
  and `smc_figure_eight` accept `amplitude` and `period`. Previously the reference
  inside the SMC always used the module constants. The body of `tau_figure_eight` moved
  into `_feedback_figure_eight_output`. `tau_figure_eight` keeps its signature and
  behaviour. A new factory, `feedback_figure_eight(...)`, exposes its gains and the same
  `output/required_tau/...` interface as the SMC.
* `run_simulation.py`: `reference_history` and `save_csv` accept optional `amplitude`
  and `period`. `main()` is unchanged.

Verification: the default `python run_simulation.py` scenario (SMC + CBF + governor)
gives a byte-identical CSV and identical PNG figures before and after these changes,
and `tau_figure_eight` gives bit-identical trajectories.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v              # all tests (~35 s)
.\.venv\Scripts\python.exe -m unittest discover -s dashboard\tests -t . -v   # dashboard only
```

The UI tests are skipped automatically if Streamlit is not installed.

| Requirement | Test(s) |
| --- | --- |
| Dashboard starts | `test_app.test_app_starts_without_results` (also checked in a real browser) |
| Existing functionality operational | original `test_*.py` files; `test_engine.test_matches_the_command_line_simulator` runs the unmodified `run_simulation.main()` |
| Controller parameter reaches the simulation | `test_engine.test_smc_parameters_reach_smc_figure_eight`, `test_app.test_controller_parameter_reaches_the_simulation` |
| Invalid configurations handled | `test_config.ValidationTests`, `test_engine.test_invalid_config_never_reaches_the_simulator`, `test_app.test_invalid_configuration_blocks_the_run` |
| Dashboard = direct Python simulator | `test_engine.test_matches_direct_call_of_the_python_simulator` (bit-identical), `test_matches_the_command_line_simulator` (CLI CSV, 4 disturbance modes, tolerance 1e-12) |
| Previous run overlaid in the XY plot (only one, gray, same axes) | `test_figures.XYOverlayTests`, `test_app.test_xy_plot_overlays_only_the_previous_run` |
| No reuse of outdated parameters/results | `test_app.test_repeated_runs_never_reuse_outdated_results`, `test_failed_run_keeps_old_results_clearly_labelled`, `test_widget_changes_do_not_run_simulations`, `test_switching_display_modes_does_not_change_the_experiment` |

## Metrics

* **Lookahead-point error** `||p_s,ref - p_s||`: the output regulated by the SMC and the
  quantity of `plot_tracking_error` and `analysis_utils.position_error`.
* **Hull-centre error** `||p_ref - p||`: controller-independent; use it when comparing
  controllers or lookahead distances.
* Max, final and RMS values over the whole run, plus "steady-state" values over the
  final 25 % of the run.
* Heading error relative to the path tangent, RMS surge-speed error, RMS control effort,
  and the total variation rate of Tu/Tr as a chattering indicator.
* Thruster saturation share (`analysis_utils.saturation_duty_cycle`).
* Indicators: solver success, final and steady-state lookahead error within the
  tolerance, and saturation below 5 % of the time.

## Limitations

* Only the figure-eight reference exists in `controllers.py`, so it is the only option.
  `manual_tau` and `tau_example` are open-loop test inputs and are not offered.
* The wind-like disturbance is a fixed profile in `disturbances.py` and has no
  parameters. No new disturbance models were added.
* Runs are synchronous. The spinner shows elapsed time, but `solve_ivp` gives no
  progress information. Runs longer than 900 s of wall-clock time are aborted. Very
  thin boundary layers or large gains make the SMC integration much slower.
* The run history lives only in the browser session (last 8 runs) and is lost on
  page reload. Use the JSON/CSV downloads to keep experiments.
* Plots are decimated to about 4000 points per signal (metrics use all samples). The
  chattering indicator is computed at the output rate dt and cannot see faster
  switching.
* For the simple feedback controller, the lookahead-point error is evaluated with the
  default L = 1 m although that controller tracks the hull centre. Compare
  controllers with the hull-centre error.
* The dashboard runs locally for a single user. It is not designed for deployment.
* Validation messages contain LaTeX markup (e.g. `$\phi_u$`), so they read slightly less
  cleanly when the engine raises `ConfigValidationError` in a plain Python script.
* Pre-existing and unchanged: with the reference governor enabled, the CLI's
  `save_csv` writes one more data column than header labels (see main README). The
  dashboard does not use the governor and is not affected.
