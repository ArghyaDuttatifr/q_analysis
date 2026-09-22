import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from functools import partial
import os
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pcolors

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Q-Factor Analysis", layout="wide", page_icon="📈")
st.title("📈 Resonator Q-Factor & Background Analysis")

# ==========================================
# SIDEBAR: CONTROLS & INPUTS
# ==========================================
st.sidebar.header("1. Data Upload")

# Dynamic session state key for widget reset
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

uploaded_files = st.sidebar.file_uploader(
    "Upload S12 Sweep Files (.txt, .csv)", 
    accept_multiple_files=True, 
    type=['txt', 'csv'],
    key=f"uploader_{st.session_state['uploader_key']}"
)

# Reset Button
if st.sidebar.button("🗑️ Reset / Clear All Files", use_container_width=True):
    st.session_state["uploader_key"] += 1
    # Clear stored analysis data
    for key in ['master_results', 'individual_plots_data', 'bg_combined']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

st.sidebar.header("2. Filtering & Options")
FILE_PREFIX = st.sidebar.text_input("File Prefix Filter", value="S12")
col1, col2 = st.sidebar.columns(2)

# Default T_min = 2.0 K, T_max = 15.0 K
T_min = col1.number_input("T min (K)", value=2.0, step=0.1)
T_max = col2.number_input("T max (K)", value=15.0, step=0.1)

# Default Background Subtraction = False
bg_correction = st.sidebar.checkbox("Enable Background Subtraction", value=False, 
    help="Uses the previous temperature's fitted background before fitting the next one.")

restrict_rows = st.sidebar.checkbox("Restrict Row Range")
if restrict_rows:
    row_col1, row_col2 = st.sidebar.columns(2)
    start_row = row_col1.number_input("Start Row (1-indexed)", value=1, min_value=1)
    end_row = row_col2.number_input("End Row", value=1600, min_value=1)
    row_range = (int(start_row), int(end_row))
else:
    row_range = None

with st.sidebar.expander("⚙️ Advanced Fit Settings"):
    RELAXATION_RANGE = st.number_input("Relaxation Range (GHz)", value=0.001, format="%.4f") * 1e9
    N_TRIALS = st.slider("Number of Trials", min_value=10, max_value=100, value=50)
    CHI2_NORM = st.number_input("Chi-Square Norm", value=1600, step=100)
    USE_SMOOTHED_REFERENCE = st.checkbox("Use Smoothed Reference (for noisy data)", value=False)
    
    st.markdown("**Parameter Bounds [QL, Qe, A, B]**")
    b_min = st.text_input("Lower Bounds", "1, 100, -0.5, -0.5")
    b_max = st.text_input("Upper Bounds", "20000, 100000, 0.5, 0.5")
    
    try:
        lower_bounds = [float(x.strip()) for x in b_min.split(',')]
        upper_bounds = [float(x.strip()) for x in b_max.split(',')]
        BOUNDS = (lower_bounds, upper_bounds)
    except:
        st.error("Invalid bounds format. Use comma-separated numbers.")
        BOUNDS = ([1, 100, -0.5, -0.5], [20000, 100000, 0.5, 0.5])


# ==========================================
# MATH & ANALYSIS FUNCTIONS
# ==========================================
def linmag(freq, QL, Qe, A, B, omega_r):
    delta = (freq - omega_r) / omega_r
    lorentzian = (QL / Qe) * (1 / (1 + 2j * QL * delta))
    return np.abs(lorentzian + A + 1j * B)

def chi_square(y_obs, y_fit):
    return np.sum((y_obs - y_fit) ** 2) / CHI2_NORM

def interpolate_background(freq_prev, background_prev, freq_curr, edge_points=50):
    interpolator = interp1d(freq_prev, background_prev, kind='linear', bounds_error=False)
    result = interpolator(freq_curr)

    f_min, f_max = freq_prev[0], freq_prev[-1]
    n = min(edge_points, len(freq_prev))

    left_mask = freq_curr < f_min
    if np.any(left_mask):
        slope, intercept = np.polyfit(freq_prev[:n], background_prev[:n], 1)
        result[left_mask] = slope * freq_curr[left_mask] + intercept

    right_mask = freq_curr > f_max
    if np.any(right_mask):
        slope, intercept = np.polyfit(freq_prev[-n:], background_prev[-n:], 1)
        result[right_mask] = slope * freq_curr[right_mask] + intercept

    return result

def get_reference(real_mag, real_mag_s):
    return real_mag_s if USE_SMOOTHED_REFERENCE else real_mag

def fit_resonance(freq, real_mag, real_mag_s):
    reference = get_reference(real_mag, real_mag_s)
    max_index = np.argmax(reference)
    omega_r_initial = freq[max_index]
    omega_r_trials = np.linspace(omega_r_initial - RELAXATION_RANGE, omega_r_initial + RELAXATION_RANGE, N_TRIALS)

    best = {'chi2': 21}  
    p0 = [2000, 2500, float(np.median(reference)), 0.0]  
    
    for omega_r in omega_r_trials:
        model = partial(linmag, omega_r=omega_r)
        try:
            popt, pcov = curve_fit(model, freq, real_mag, p0=p0, bounds=BOUNDS, maxfev=5000, x_scale='jac')
        except RuntimeError:
            continue
        
        p0 = list(popt)  
        fit_values = model(freq, *popt)
        chi2 = chi_square(reference, fit_values)

        if chi2 < best['chi2']:
            best.update(
                chi2=chi2, popt=popt, pcov=pcov, fit_values=fit_values,
                omega_r=omega_r, omega_r_max=freq[np.argmax(fit_values)],
            )

    if 'popt' not in best:
        return None
    best['omega_r_initial'] = omega_r_initial
    return best

def compute_background(freq, corrected, result):
    QL, Qe, A, B = result['popt']
    omega_r = result['omega_r']
    model_values = linmag(freq, QL, Qe, A, B, omega_r)
    residual = corrected - model_values
    return residual + abs(complex(A, B))


# ==========================================
# INTERACTIVE PLOTLY FUNCTIONS
# ==========================================
def create_plotly_fit(T, freq_fit, corrected_fit, fit_values):
    freq_ghz = freq_fit / 1e9
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq_ghz, y=corrected_fit, mode='markers', name='Experiment (Corrected)', 
                             marker=dict(color='#1f77b4', size=5, opacity=0.6)))
    fig.add_trace(go.Scatter(x=freq_ghz, y=fit_values, mode='lines', name='Fitted Curve', 
                             line=dict(color='#d62728', width=3)))
    
    fig.update_layout(title=f"Fitting for T = {T} K", xaxis_title="Frequency (GHz)", yaxis_title="|S21| Magnitude",
                      hovermode="x unified", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_plotly_bg(T, freq, background):
    freq_ghz = freq / 1e9
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq_ghz, y=background, mode='lines', name='Background', 
                             line=dict(color='#9467bd', width=2)))
    
    fig.update_layout(title=f"Background for T = {T} K", xaxis_title="Frequency (GHz)", yaxis_title="Background Magnitude",
                      hovermode="x unified", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig


# ==========================================
# MAIN EXECUTION
# ==========================================
if uploaded_files:
    entries = []
    
    for file in uploaded_files:
        if not file.name.startswith(FILE_PREFIX): continue
        if "_T_" not in file.name: continue
            
        T_str = os.path.splitext(file.name.split("_T_")[1])[0].replace("_", ".")
        try:
            T = float(T_str)
            if T_min <= T <= T_max:
                entries.append((T, file))
        except ValueError:
            continue

    entries.sort(key=lambda e: e[0])
    
    if not entries:
        st.warning(f"No matching files found in T range [{T_min}, {T_max}] with prefix '{FILE_PREFIX}'.")
    else:
        st.success(f"Found {len(entries)} matching sweep dataset(s) ready for analysis.")
        
        # RUN ANALYSIS BUTTON
        if st.button("🚀 Run Analysis", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            freq_prev, background_prev = None, None
            master_results = []
            bg_vs_temp_data = []
            individual_plots_data = {}
            
            for idx, (T, file) in enumerate(entries):
                status_text.text(f"Processing T = {T} K ({idx+1}/{len(entries)})")
                
                # Read data
                file.seek(0)
                raw_data = np.genfromtxt(file, delimiter=',', skip_header=1)
                freq_full = raw_data[:, 0]
                real_mag_raw = raw_data[:, 3]
                
                # Background correction
                if bg_correction and background_prev is not None:
                    background_in = interpolate_background(freq_prev, background_prev, freq_full)
                else:
                    background_in = np.zeros_like(freq_full)
                
                corrected_full = real_mag_raw - background_in
                
                # Row range
                if row_range is not None:
                    start_idx = max(0, row_range[0] - 1)
                    end_idx = min(len(freq_full), row_range[1])
                else:
                    start_idx, end_idx = 0, len(freq_full)
                    
                freq_fit = freq_full[start_idx:end_idx]
                corrected_fit = corrected_full[start_idx:end_idx]
                
                # SG Filter
                n_fit = len(corrected_fit)
                window_length = min(9, n_fit if n_fit % 2 == 1 else n_fit - 1)
                corrected_fit_s = (corrected_fit if window_length < 5 else savgol_filter(corrected_fit, window_length=window_length, polyorder=3))
                
                # Fit
                result = fit_resonance(freq_fit, corrected_fit, corrected_fit_s)
                
                if result is not None:
                    QL, Qex, A, B = result['popt']
                    QL_err, Qex_err, A_err, B_err = np.sqrt(np.diag(result['pcov']))
                    Q_in = 1 / (1 / QL - 1 / Qex)
                    Q_in_err = Q_in**2 * np.sqrt((QL_err**2) / max(QL**4, 1e-10) + (Qex_err**2) / max(Qex**4, 1e-10))
                    f0_ghz = result['omega_r_max'] / 1e9
                    
                    master_results.append({
                        "T (K)": T,
                        "f0 (GHz)": f0_ghz,
                        "Q_in": Q_in, "Q_in_err": Q_in_err,
                        "Q_ex": Qex, "Q_ex_err": Qex_err,
                        "Q_L": QL, "Q_L_err": QL_err,
                        "A": A, "A_err": A_err,
                        "B": B, "B_err": B_err,
                        "chi_square": result['chi2'],
                        "File": file.name
                    })
                    
                    background_local_full = compute_background(freq_full, corrected_full, result)
                    background_out = background_local_full + background_in
                    
                    if bg_correction:
                        freq_prev, background_prev = freq_full, background_out
                        
                    # Save arrays for interactive plotting
                    individual_plots_data[T] = {
                        "freq_fit": freq_fit,
                        "corrected_fit": corrected_fit,
                        "fit_values": result['fit_values'],
                        "freq_full": freq_full[start_idx:end_idx],
                        "bg_out": background_out[start_idx:end_idx]
                    }
                    
                    bg_vs_temp_data.append(np.column_stack([np.full_like(freq_full[start_idx:end_idx], T), freq_full[start_idx:end_idx], background_out[start_idx:end_idx]]))
                
                progress_bar.progress((idx + 1) / len(entries))
            
            # Store results in session state
            st.session_state['master_results'] = pd.DataFrame(master_results)
            st.session_state['individual_plots_data'] = individual_plots_data
            if bg_vs_temp_data:
                st.session_state['bg_combined'] = pd.DataFrame(np.vstack(bg_vs_temp_data), columns=['T', 'freq_Hz', 'background'])
            else:
                st.session_state['bg_combined'] = pd.DataFrame()
            
            status_text.empty()
            progress_bar.empty()
            st.success("✅ Analysis Complete!")

        # ==========================================
        # DASHBOARD VIEWS (RENDER FROM SESSION STATE)
        # ==========================================
        if 'master_results' in st.session_state and not st.session_state['master_results'].empty:
            df_res = st.session_state['master_results']
            individual_plots_data = st.session_state.get('individual_plots_data', {})
            
            tab_trends, tab_individual, tab_data = st.tabs(["📈 Global Trends", "🔍 Individual Scans", "📊 Data Table & Export"])
            
            # --- TAB 1: GLOBAL TRENDS ---
            with tab_trends:
                # 1. All Q-Factors in Log Scale
                st.subheader("Extracted Q-Factors vs Temperature (Log Scale)")
                fig_q_log = go.Figure()
                fig_q_log.add_trace(go.Scatter(x=df_res['T (K)'], y=df_res['Q_in'], mode='lines+markers', name='Internal Q (Q_in)', line=dict(color='#2ca02c', width=2)))
                fig_q_log.add_trace(go.Scatter(x=df_res['T (K)'], y=df_res['Q_L'], mode='lines+markers', name='Loaded Q (Q_L)', line=dict(color='#ff7f0e', width=2)))
                fig_q_log.add_trace(go.Scatter(x=df_res['T (K)'], y=df_res['Q_ex'], mode='lines+markers', name='External Q (Q_ex)', line=dict(color='#1f77b4', width=2)))
                
                fig_q_log.update_layout(
                    xaxis_title="Temperature (K)", 
                    yaxis_title="Q-Factor (Log Scale)", 
                    yaxis_type="log",
                    hovermode="x unified", 
                    template="plotly_white"
                )
                st.plotly_chart(fig_q_log, use_container_width=True)
                
                # 2. Q_in with Error Bars in Linear Scale
                st.subheader("Internal Q-Factor (Q_in) with Error Bars (Linear Scale)")
                fig_qin_err = go.Figure()
                fig_qin_err.add_trace(go.Scatter(
                    x=df_res['T (K)'], 
                    y=df_res['Q_in'], 
                    mode='lines+markers', 
                    name='Q_in', 
                    line=dict(color='#2ca02c', width=2),
                    error_y=dict(
                        type='data', 
                        array=df_res['Q_in_err'], 
                        visible=True,
                        color='#2ca02c',
                        thickness=1.5,
                        width=4
                    )
                ))
                
                fig_qin_err.update_layout(
                    xaxis_title="Temperature (K)", 
                    yaxis_title="Internal Q-Factor (Q_in)", 
                    hovermode="x unified", 
                    template="plotly_white"
                )
                st.plotly_chart(fig_qin_err, use_container_width=True)

                # 3. f0 vs Temperature
                st.subheader("Resonance Frequency (f0) vs Temperature")
                fig_f = px.line(df_res, x='T (K)', y='f0 (GHz)', markers=True, color_discrete_sequence=['#d62728'])
                fig_f.update_layout(
                    xaxis_title="Temperature (K)", 
                    yaxis_title="Resonance Frequency (GHz)", 
                    hovermode="x unified", 
                    template="plotly_white"
                )
                st.plotly_chart(fig_f, use_container_width=True)

                # 4. Fitted Backgrounds across Temperatures
                st.subheader("Fitted Backgrounds vs Frequency (All Temperatures)")
                fig_bg_all = go.Figure()
                
                temps_sorted = sorted(individual_plots_data.keys())
                num_temps = len(temps_sorted)
                
                if num_temps > 0:
                    # Generate smooth color gradient across sorted temperatures
                    norm_indices = np.linspace(0, 1, num_temps) if num_temps > 1 else [0.5]
                    colors = pcolors.sample_colorscale('Viridis', norm_indices)
                    
                    for idx, T_val in enumerate(temps_sorted):
                        pdata = individual_plots_data[T_val]
                        freq_ghz = pdata["freq_full"] / 1e9
                        bg_vals = pdata["bg_out"]
                        
                        fig_bg_all.add_trace(go.Scatter(
                            x=freq_ghz,
                            y=bg_vals,
                            mode='lines',
                            name=f"{T_val:.3f} K",
                            line=dict(color=colors[idx], width=1.5)
                        ))
                    
                    fig_bg_all.update_layout(
                        xaxis_title="Frequency (GHz)",
                        yaxis_title="Background Magnitude",
                        hovermode="x unified",
                        template="plotly_white",
                        legend_title="Temperature (K)"
                    )
                    st.plotly_chart(fig_bg_all, use_container_width=True)

            # --- TAB 2: INDIVIDUAL SCANS ---
            with tab_individual:
                st.markdown("Select a specific temperature from the dropdown below to view its interactive fit and background.")
                selected_T = st.selectbox("Select Temperature (K)", options=df_res['T (K)'].unique())
                
                if selected_T in individual_plots_data:
                    plot_data = individual_plots_data[selected_T]
                    
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        fig_fit = create_plotly_fit(selected_T, plot_data["freq_fit"], plot_data["corrected_fit"], plot_data["fit_values"])
                        st.plotly_chart(fig_fit, use_container_width=True)
                    with col_p2:
                        fig_bg = create_plotly_bg(selected_T, plot_data["freq_full"], plot_data["bg_out"])
                        st.plotly_chart(fig_bg, use_container_width=True)

            # --- TAB 3: DATA & EXPORT ---
            with tab_data:
                st.dataframe(df_res.style.format({"f0 (GHz)": "{:.6f}", "Q_in": "{:.1f}", "Q_L": "{:.1f}"}), use_container_width=True)
                
                col_d1, col_d2 = st.columns(2)
                col_d1.download_button(
                    label="📥 Download Q-Factor Results (CSV)",
                    data=df_res.to_csv(index=False),
                    file_name="q_factor_results.csv",
                    mime="text/csv"
                )
                
                if 'bg_combined' in st.session_state and not st.session_state['bg_combined'].empty:
                    col_d2.download_button(
                        label="📥 Download Background vs Temp Data (CSV)",
                        data=st.session_state['bg_combined'].to_csv(index=False),
                        file_name="background_vs_temperature.csv",
                        mime="text/csv"
                    )
else:
    st.info("👈 Please upload your `S12_..._T_...txt` files in the sidebar to begin.")
