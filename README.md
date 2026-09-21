# 📈 Microwave Resonator Q-Factor & Background Analysis

An interactive web application built with **Streamlit**, **SciPy**, and **Plotly** for automated fitting, background subtraction, and Q-factor extraction ($Q_{in}$, $Q_L$, $Q_{ex}$) of microwave resonator frequency sweeps across varying temperatures.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Plotly](https://img.shields.io/badge/Plotly-Interactive--Plots-brightgreen.svg)
![SciPy](https://img.shields.io/badge/SciPy-Optimization--%26--Fitting-green.svg)

---

## 🌟 Key Features

* **Complex Lorentzian Model Fitting:** Fits raw or background-corrected magnitude $\vert{}S_{21}\vert{}$ data to extract Loaded Quality Factor ($Q_L$), External Quality Factor ($Q_{ex}$), and Internal Quality Factor ($Q_{in}$).
* **Iterative Background Subtraction:** Chains background models across sequential temperature scans, interpolating and extrapolating background curves from lowest to highest temperature.
* **Fully Interactive Visualizations (Plotly):**
  * **Q-Factors vs Temperature (Log Scale):** $Q_{in}$, $Q_L$, and $Q_{ex}$ plotted together on a logarithmic scale with hover tooltips showing exact X/Y values.
  * **Internal Q-Factor ($Q_{in}$) with Error Bars:** Linear-scale representation of $Q_{in}$ featuring calculated error propagation limits ($\pm Q_{in,\text{err}}$).
  * **Resonance Frequency ($f_0$) vs Temperature:** Tracks resonant frequency shift across temperatures.
  * **Individual Scan Inspection:** Select any temperature from a dropdown to view unified hover plots for raw/corrected experiments, Lorentzian fit curves, and local background profiles.
* **Flexible Filtering & Controls:**
  * File prefix filtering (e.g., process only files starting with `S12`).
  * Temperature range bounding ($T_{\min}$ and $T_{\max}$).
  * Row index slicing (start and end row restrictions).
  * Customizable fitting bounds and relaxation ranges.
* **One-Click CSV Data Export:** Download complete aggregated summary results ($Q$-factors, $f_0$, errors, $\chi^2$) and global background vs. temperature matrices.

---

## 🔬 Mathematical Fitting Model

The fitting algorithm models the linear transmission magnitude $\vert{}S_{21}(f)\vert{}$ of a microwave resonator using a **complex Lorentzian resonance model** superimposed on a **complex constant background**.

### 1. ${\lambda/{2}}$ resonator Model Equation
The complex transmission response $S_{21}(f)$ as a function of frequency $f$ is modeled as:

$$S_{21}(f) = \frac{Q_L}{Q_e} \cdot \frac{1}{1 + 2j Q_L \delta(f)} + (A + j B)$$

Fitting is performed directly against the linear magnitude $\vert{}S_{21}(f)\vert{}$:

$$\vert{}S_{21}(f)\vert{} = \left\vert{} \frac{Q_L}{Q_e} \cdot \frac{1}{1 + 2j Q_L \left( \frac{f - f_0}{f_0} \right)} + A + j B \right\vert{}$$

### 2. Parameter Definitions
| Parameter | Symbol | Description |
| :--- | :--- | :--- |
| **Resonant Frequency** | $f_0$ ($\omega_r$) | Center frequency of energy storage peak (Hz). |
| **Fractional Detuning** | $\delta(f)$ | Normalized frequency shift: $\delta(f) = \frac{f - f_0}{f_0}$. |
| **Loaded $Q$-Factor** | $Q_L$ | Total quality factor accounting for all loss mechanisms. |
| **External $Q$-Factor** | $Q_e$ ($Q_{ex}$) | Quality factor representing coupling losses to external feedlines. |
| **Complex Background** | $A + j B$ | Baseline constant accounting for attenuation, phase offset, and crosstalk. |

### 3. Extraction of Internal Quality Factor ($Q_{in}$)
The loaded quality factor combines internal dissipation ($Q_{in}$) and external coupling losses ($Q_e$):

$$\frac{1}{Q_L} = \frac{1}{Q_{in}} + \frac{1}{Q_e}$$

The intrinsic internal quality factor $Q_{in}$ is calculated via:

$$Q_{in} = \frac{Q_L \cdot Q_e}{Q_e - Q_L}$$

### 4. Optimization & $\chi^2$ Minimization Strategy
To avoid local minima in steep resonance landscapes, a two stage grid search strategy is used:
1. **Grid Search over $f_0$:** $f_0$ is evaluated over $N_{\text{trials}}$ fixed frequencies within a window RELAXATION RANGE around the peak.
2. **Sub-Optimization:** For each trial $f_0$, non-linear least squares (`scipy.optimize.curve_fit`) optimizes $[Q_L, Q_e, A, B]$ within strict parameter bounds.
3. **$\chi^2$ Scoring:** The optimal parameter set is chosen by minimizing:

$$\chi^2 = \frac{1}{\text{CHI2 NORM}} \sum_{{i}} \left( y{\text{obs}, i} - y{\text{fit}, i} \right)^2$$

--

## 📁 Input Data Format

### 1. File Naming Convention
To enable automatic temperature parsing, filenames must include `_T_` followed by the temperature in Kelvin:
```text
S12_sample_01_T_0.050.txt  --> Parsed Temperature: 0.050 K
S12_sample_01_T_4.200.txt  --> Parsed Temperature: 4.200 K
