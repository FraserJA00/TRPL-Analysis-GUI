import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import os
from itertools import cycle
import pandas as pd
import re

def rate_equation(t, n, k1, k2):
    return -k1 * n - k2 * n**2

def model(t, k1, k2, n0):
    t_unique = np.unique(t)
    sol = solve_ivp(rate_equation, [t_unique[0], t_unique[-1]], [n0], args=(k1, k2), t_eval=t_unique)
    return np.interp(t, sol.t, sol.y[0])

def compute_r_squared(y_exp, y_fit):
    ss_res = np.sum((y_exp - y_fit) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
    return 1 - (ss_res / ss_tot)

def load_data(file_path, max_rows=None, shift_to_max=False):
    numeric_rows = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            s = line.strip()
            if not s:
                continue
            parts = re.split(r'[,\s]+', s)
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            numeric_rows.append((t, y))

    if not numeric_rows:
        raise ValueError(
            f"No numeric data found in first two columns: {file_path}.\n"
            "Check the delimiter and headers."
        )

    data = np.array(numeric_rows, dtype=float)
    time = data[:, 0]
    intensity = data[:, 1]


    if time.size == 0 or intensity.size == 0:
        raise ValueError("Parsed arrays are empty after filtering.")
    if not np.isfinite(time).all() or not np.isfinite(intensity).all():
        raise ValueError("Found NaN/Inf values in parsed data.")
    if np.all(np.diff(time) <= 0):
        # If time is not strictly increasing, sort by time
        idx = np.argsort(time)
        time = time[idx]
        intensity = intensity[idx]

    return preprocess_data(time, intensity, shift_to_max=shift_to_max)

def preprocess_data(time, intensity, shift_to_max=False):
    mask = np.isfinite(time) & np.isfinite(intensity)
    time = time[mask]
    intensity = intensity[mask]

    if shift_to_max:
        max_index = np.argmax(intensity)
        print(f"Max intensity = {intensity[max_index]:.6f} at time = {time[max_index]:.6f}")
        t_shift = time[max_index]
        time = time - t_shift
    else:
        time -= time[0]

    sorted_indices = np.argsort(time)
    time = time[sorted_indices]
    intensity = intensity[sorted_indices]

    valid_mask = time >= 0
    time = time[valid_mask]
    intensity = intensity[valid_mask]

    max_val = np.max(intensity)
    if max_val != 0:
        intensity = intensity / max_val
        
    return time, intensity

COLOR_CYCLE = cycle(plt.get_cmap('tab10').colors)

class TRPLFittingApp:
    def apply_fit_settings_to_selected(self):
        selected = self.tree.selection()
        if not selected:
            return
        for row_id in selected:
            label = self.tree.item(row_id, "values")[0]
            self.file_fit_options[label] = {
                'fix_k1': self.perfile_fix_k1.get(),
                'fix_k2': self.perfile_fix_k2.get(),
                'k1_val': self.perfile_k1_val.get(),
                'k2_val': self.perfile_k2_val.get()
            }

    def __init__(self, master):
        self.master = master
        master.title("TRPL Fitting GUI")

        self.file_info = []
        self.file_fit_options = {} 

        style = ttk.Style()
        style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'))
        style.configure("TButton", font=('Helvetica', 10))
        style.configure("TLabel", font=('Helvetica', 10))

        container = tk.Frame(master, padx=10, pady=10)
        container.grid(row=0, column=0)

        param_frame = tk.LabelFrame(container, text="Fitting Parameters", padx=10, pady=10)
        param_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        self.k1_guess = tk.DoubleVar(value="1e-3")
        self.k2_guess = tk.DoubleVar(value="1e-19")
        self.G = tk.DoubleVar(value="1e15")
        self.max_rows = tk.IntVar(value=100000)
        self.n_cutoff = tk.DoubleVar(value="1e13")

        tk.Label(param_frame, text="k1 Guess").grid(row=0, column=0, sticky="e")
        tk.Entry(param_frame, textvariable=self.k1_guess, width=12).grid(row=0, column=1, sticky="w")
        tk.Label(param_frame, text="k2 Guess").grid(row=0, column=2, sticky="e")
        tk.Entry(param_frame, textvariable=self.k2_guess, width=12).grid(row=0, column=3, sticky="w")
        tk.Label(param_frame, text="Generation Rate (G)").grid(row=1, column=0, sticky="e")
        tk.Entry(param_frame, textvariable=self.G, width=12).grid(row=1, column=1, sticky="w")
        tk.Label(param_frame, text="Max Rows to Load").grid(row=1, column=2, sticky="e")
        tk.Entry(param_frame, textvariable=self.max_rows, width=12).grid(row=1, column=3, sticky="w")
        tk.Label(param_frame, text="n Cutoff").grid(row=2, column=0, sticky="e")
        tk.Entry(param_frame, textvariable=self.n_cutoff, width=12).grid(row=2, column=1, sticky="w")


        file_frame = tk.LabelFrame(container, text="File Operations", padx=10, pady=10)
        file_frame.grid(row=1, column=0, sticky="ew")

        self.tree = ttk.Treeview(file_frame, columns=("Label", "Color", "k1", "k2", "R1 %", "R2 %", "R²"), show='headings', height=5)
        for col in ("Label", "Color", "k1", "k2", "R1 %", "R2 %", "R²"):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80, anchor='center')
        self.tree.pack(fill="x")
        self.tree.bind("<Double-1>", self.edit_cell)

        setting_frame = tk.LabelFrame(container, text="Fix Parameters for Selected", padx=10, pady=10)
        setting_frame.grid(row=2, column=0, sticky="ew")

        self.perfile_fix_k1 = tk.BooleanVar()
        self.perfile_fix_k2 = tk.BooleanVar()
        self.perfile_k1_val = tk.DoubleVar(value=1e-3)
        self.perfile_k2_val = tk.DoubleVar(value=1e-10)

        tk.Checkbutton(setting_frame, text="Fix k1", variable=self.perfile_fix_k1).grid(row=0, column=0, sticky="w")
        tk.Entry(setting_frame, textvariable=self.perfile_k1_val, width=12).grid(row=0, column=1, sticky="w")
        tk.Checkbutton(setting_frame, text="Fix k2", variable=self.perfile_fix_k2).grid(row=1, column=0, sticky="w")
        tk.Entry(setting_frame, textvariable=self.perfile_k2_val, width=12).grid(row=1, column=1, sticky="w")

        tk.Button(setting_frame, text="Apply to Selected", command=self.apply_fit_settings_to_selected).grid(row=2, column=0, columnspan=2, pady=5)
        tk.Button(setting_frame, text="Run Fitting", command=self.run_fitting).grid(row=3, column=0, columnspan=2, pady=10)

        plot_frame = tk.LabelFrame(container, text="Plot Options", padx=10, pady=10)
        plot_frame.grid(row=2, column=1, sticky="ew")
        
        self.shift_to_max = tk.BooleanVar(value=True)
        tk.Checkbutton(plot_frame, text="Shift t=0 to Max Intensity", variable=self.shift_to_max).grid(row=5, column=0, sticky="w")


        self.plot_carrier_log = tk.BooleanVar(value=False)
        self.plot_carrier_lin = tk.BooleanVar(value=False)
        self.plot_residuals = tk.BooleanVar(value=False)
        self.plot_intensity_log = tk.BooleanVar(value=True)
        self.plot_intensity_lin = tk.BooleanVar(value=True)
        self.plot_raw = tk.BooleanVar(value=False)

        tk.Checkbutton(plot_frame, text="Carrier Log", variable=self.plot_carrier_log).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(plot_frame, text="Carrier Lin", variable=self.plot_carrier_lin).grid(row=1, column=0, sticky="w")
        tk.Checkbutton(plot_frame, text="Residuals", variable=self.plot_residuals).grid(row=2, column=0, sticky="w")
        tk.Checkbutton(plot_frame, text="Intensity Log", variable=self.plot_intensity_log).grid(row=3, column=0, sticky="w")
        tk.Checkbutton(plot_frame, text="Intensity Lin", variable=self.plot_intensity_lin).grid(row=4, column=0, sticky="w")
        tk.Checkbutton(plot_frame, text="Plot Raw Data", variable=self.plot_raw).grid(row=6, column=0, sticky="w")


        tk.Button(container, text="Load Files", command=self.load_files).grid(row=3, column=0, columnspan=2, pady=10)


    def load_files(self):
        global COLOR_CYCLE
        COLOR_CYCLE = cycle(plt.get_cmap('tab10').colors)
        file_paths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt", ".csv")])
        for path in file_paths:
            base = os.path.basename(path)
            # label = base.split('-')[0].split('_')[0].split('.')[0]
            label = base.split('-')[0].split('.')[0]
            color = next(COLOR_CYCLE)
            self.file_info.append({"path": path, "label": label, "color": color})
            self.tree.insert("", "end", values=(label, color, '', '', '', '', ''))

    def edit_cell(self, event):
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if not item or column not in ("#1", "#2"):
            return
        x, y, width, height = self.tree.bbox(item, column)
        value = self.tree.set(item, column)
        entry = tk.Entry(self.tree)
        entry.place(x=x, y=y, width=width, height=height)
        entry.insert(0, value)
        entry.focus()
        def on_enter(event):
            new_value = entry.get()
            col_idx = int(column[1:]) - 1
            current_values = list(self.tree.item(item, 'values'))
            current_values[col_idx] = new_value
            self.tree.item(item, values=current_values)
            self.file_info[self.tree.index(item)]["label"] = current_values[0]
            self.file_info[self.tree.index(item)]["color"] = current_values[1]
            entry.destroy()
        entry.bind('<Return>', on_enter)
        entry.bind('<FocusOut>', lambda e: entry.destroy())
        
    def run_fitting(self):
        if not self.file_info:
            messagebox.showwarning("No Files", "Please load TRPL files first.")
            return

        if self.plot_carrier_log.get():
            fig1, ax1 = plt.subplots(figsize=(7, 6))
        if self.plot_carrier_lin.get():
            fig1_lin, ax1_lin = plt.subplots(figsize=(7, 6))
        if self.plot_residuals.get():
            fig2, ax3 = plt.subplots(figsize=(7, 6))
        if self.plot_intensity_log.get():
            fig3, ax4 = plt.subplots(figsize=(7, 6))
        if self.plot_intensity_lin.get():
            fig3_lin, ax4_lin = plt.subplots(figsize=(7, 6))


        for info in self.file_info:
            path = info["path"]
            label = info["label"]
            color_raw = info["color"]
            if isinstance(color_raw, str):
                try:
                    color = tuple(map(float, color_raw.strip("()").replace(',', ' ').split()))
                except:
                    messagebox.showerror("Color Error", f"Invalid color format for {info['label']}: {color_raw}")
                    continue
            else:
                color = color_raw
                
            time, I_norm = load_data(
                path,
                max_rows=int(self.max_rows.get()),
                shift_to_max=self.shift_to_max.get()
            )
            shifted_output = np.column_stack((time, I_norm))
            shifted_filename = os.path.join(
                os.path.dirname(path),
                os.path.basename(path).replace(".txt", "_Shifted_Data.txt")
            )
            np.savetxt(
                shifted_filename,
                shifted_output,
                header="Time (ns)\tNormalised Intensity (shifted to max)",
                fmt="%.6e",
                delimiter="\t"
            )

            time_raw, I_raw_full = load_data(path, max_rows=int(self.max_rows.get()), shift_to_max=False)

            mask = np.isfinite(time_raw) & np.isfinite(I_raw_full)
            time_raw = time_raw[mask]
            I_raw = I_raw_full[mask]
            
            if self.shift_to_max.get():
                t_shift = time_raw[np.argmax(I_raw)]
                time_raw = time_raw - t_shift
            else:
                time_raw -= time_raw[0]
            
            sorted_indices = np.argsort(time_raw)
            time_raw = time_raw[sorted_indices]
            I_raw = I_raw[sorted_indices]
            
            valid_mask = time_raw >= 0
            time_raw = time_raw[valid_mask]
            I_raw = I_raw[valid_mask]
            
            # Testing if this makes the same plot as the norm plot
            I_raw_norm = (I_raw - np.min(I_raw)) / (np.max(I_raw) - np.min(I_raw))
            I_raw_norm = np.clip(I_raw_norm, 1e-10, 1.0)
            

            time = np.array(time)
            n_exp = np.sqrt(I_norm) * float(self.G.get())
            n_exp = np.clip(n_exp, 1e13, None)
            n0_guess = n_exp[0]
            
            # Option 1: Filter out low n_exp values to avoid tail artifacts
            n_cutoff = self.n_cutoff.get()
            if len(n_exp) == 0:
                messagebox.showerror("Fit Error", f"{label}: no valid data points after applying cutoff of {n_cutoff:.1e}. Try increasing G or lowering the cutoff.")
                continue
            valid_mask = n_exp > n_cutoff
            
            time = time[valid_mask]
            n_exp = n_exp[valid_mask]
            I_norm = I_norm[valid_mask]  
            fit_opts = self.file_fit_options.get(label, {})
            fix_k1 = fit_opts.get('fix_k1', False)
            fix_k2 = fit_opts.get('fix_k2', False)
            k1_val = float(fit_opts.get('k1_val', self.k1_guess.get())) if fix_k1 else None
            k2_val = float(fit_opts.get('k2_val', self.k2_guess.get())) if fix_k2 else None

            try:
                if fix_k1 and fix_k2:
                    k1_opt, k2_opt = k1_val, k2_val
                elif fix_k1:
                    def fit_func(t, k2): return model(t, k1_val, k2, n0_guess)
                    popt, _ = curve_fit(fit_func, time, n_exp, p0=[self.k2_guess.get()])
                    k1_opt, k2_opt = k1_val, popt[0]
                elif fix_k2:
                    def fit_func(t, k1): return model(t, k1, k2_val, n0_guess)
                    popt, _ = curve_fit(fit_func, time, n_exp, p0=[self.k1_guess.get()])
                    k1_opt, k2_opt = popt[0], k2_val
                else:
                    def fit_func(t, k1, k2): return model(t, k1, k2, n0_guess)
                    popt, _ = curve_fit(fit_func, time, n_exp, p0=[self.k1_guess.get(), self.k2_guess.get()])
                    k1_opt, k2_opt = popt
            except Exception as e:
                messagebox.showerror("Fit Error", f"{label}: {str(e)}")
                continue

            n_fit = model(time, k1_opt, k2_opt, n0_guess)
            r_squared_n = compute_r_squared(n_exp, n_fit)
            I_fit_norm = (n_fit / n0_guess) ** 2
            r_squared_I = compute_r_squared(I_norm, I_fit_norm)
            residual = I_norm - I_fit_norm

            if self.plot_carrier_log.get():
                ax1.plot(time, n_exp, 'o', label=f'{label}', alpha=0.5, color=color)
                ax1.plot(time, n_fit, '-.', color='black', linewidth=2.5)
            if self.plot_carrier_lin.get():
                ax1_lin.plot(time, n_exp, 'o', label=f'{label}', alpha=0.5, color=color)
                ax1_lin.plot(time, n_fit, '-.', color='black', linewidth=2.5)
            if self.plot_residuals.get():
                ax3.plot(time, residual, label=label, color=color)
            if self.plot_intensity_log.get():
                ax4.plot(time, I_norm, 'o', label=f'{label}', alpha=0.5, color=color)
                ax4.plot(time, I_fit_norm, '-.', color='black', linewidth=2.5)
            if self.plot_intensity_lin.get():
                ax4_lin.plot(time, I_norm, 'o', label=f'{label}', alpha=0.5, color=color)
                ax4_lin.plot(time, I_fit_norm, '-.', color='black', linewidth=2.5)

            if self.plot_raw.get():
                plt.figure("Raw Intensity")
                ax_raw = plt.gca()
                ax_raw.plot(time_raw, I_raw, 'o-', label=label, color=color)
                ax_raw.set_xlabel("Time (ns)", fontsize=18)
                ax_raw.set_ylabel("Raw PL Intensity (a.u.)", fontsize=18)
                ax_raw.set_yscale("log")
                ax_raw.legend(fontsize=14, frameon=False)
                for spine in ax_raw.spines.values():
                    spine.set_linewidth(1.5)
                ax_raw.tick_params(axis='both', which='both', direction='in', width=1.5)
                plt.tight_layout()
            
                plt.figure("Normalised Raw Intensity")
                ax_norm = plt.gca()
                ax_norm.plot(time_raw, I_raw_norm, 'o-', label=label, color=color)
                ax_norm.set_xlabel("Time (ns)", fontsize=18)
                ax_norm.set_ylabel("Normalised PL Intensity (a.u.)", fontsize=18)
                ax_norm.set_yscale("log")
                ax_norm.legend(fontsize=14, frameon=False)
                for spine in ax_norm.spines.values():
                    spine.set_linewidth(1.5)
                ax_norm.tick_params(axis='both', which='both', direction='in', width=1.5)
                plt.tight_layout()

            output_data = np.column_stack((time, I_fit_norm))
            output_filename = os.path.join(os.path.dirname(path), os.path.basename(path).replace(".txt", "_Fitted_Data.txt"))
            np.savetxt(output_filename, output_data, header="Time (ns)	Fitted Normalised Intensity", fmt="%.6e", delimiter="	")

            R1 = k1_opt * n_fit
            R2 = k2_opt * n_fit ** 2
            R1_avg = np.mean(R1)
            R2_avg = np.mean(R2)
            R_total = R1_avg + R2_avg
            R1_pct = 100 * R1_avg / R_total
            R2_pct = 100 * R2_avg / R_total

            summary_lines = [
                f"{label}\n",
                "  GUI Input Parameters:\n",
                f"    k1 Guess: {self.k1_guess.get():.3e}\n",
                f"    k2 Guess: {self.k2_guess.get():.3e}\n",
                f"    G (Generation Rate): {self.G.get():.3e}\n",
                f"    n_cutoff: {self.n_cutoff.get():.3e}\n",
                f"    Fix k1: {fix_k1} (Value used: {k1_val if fix_k1 else 'Fitted'})\n",
                f"    Fix k2: {fix_k2} (Value used: {k2_val if fix_k2 else 'Fitted'})\n",
                "\n",
                "  Fit Results:\n",
                f"    k1 (ns^-1): {k1_opt:.3e}\n",
                f"    k2 (cm^3/ns): {k2_opt:.3e}\n",
                f"    R² (Carrier Concentration): {r_squared_n:.4f}\n",
                f"    R² (Intensity): {r_squared_I:.4f}\n",
                f"    Recombination Contributions: R1 = {R1_pct:.1f}%, R2 = {R2_pct:.1f}%\n"
            ]

            summary_path = os.path.join(os.path.dirname(path), os.path.basename(path).replace(".txt", "_fit_info.txt"))
            with open(summary_path, 'w') as f:
                f.writelines(summary_lines)

            # Update TreeView row with results
            for row_id in self.tree.get_children():
                row_vals = self.tree.item(row_id)['values']
                if row_vals[0] == label:
                    self.tree.item(row_id, values=(label, color, f"{k1_opt:.2e}", f"{k2_opt:.2e}", f"{R1_pct:.1f}", f"{R2_pct:.1f}", f"{r_squared_I:.4f}"))
                    break

                        
        def format_plot(ax, xlabel, ylabel, title="", log_y=False):
            ax.set_xlabel(xlabel, fontsize=18)
            ax.set_ylabel(ylabel, fontsize=18)
            if title:
                ax.set_title(title)
            if log_y:
                ax.set_yscale('log')
        
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
        
            ax.tick_params(axis='y', which='major', direction='in', length=10, width=1.5,
                           labelsize=14, right=True, left=True, labelright=False, labelleft=True, pad=5)
            ax.tick_params(axis='y', which='minor', direction='in', length=3, width=1.5,
                           labelsize=12, right=True, left=True, labelright=False, labelleft=True)
            ax.tick_params(axis='x', which='both', direction='in', length=5, width=1.5,
                           labelsize=14, top=True, pad=7.5)
            ax.legend(fontsize=18, loc="best", frameon=False)
            # ax.set_xscale('log')
            
        if self.plot_carrier_log.get():
            format_plot(ax1, "Time (ns)", "Carrier Concentration (cm$^{-3}$)", log_y=True)
            fig1.tight_layout()
            fig1.show()
        
        if self.plot_carrier_lin.get():
            format_plot(ax1_lin, "Time (ns)", "Carrier Concentration (cm$^{-3}$)")
            fig1_lin.tight_layout()
            fig1_lin.show()
        
        if self.plot_residuals.get():
            ax3.axhline(0, linestyle='--', color='black', linewidth=1)
            format_plot(ax3, "Time (ns)", "Residual (Exp - Fit)")
            fig3.tight_layout()
            fig3.show()
        
        if self.plot_intensity_log.get():
            format_plot(ax4, "Time (ns)", "Normalised Intensity (a.u.)", log_y=True)
            fig3.tight_layout()
            fig3.show()
        
        if self.plot_intensity_lin.get():
            format_plot(ax4_lin, "Time (ns)", "Normalised Intensity (a.u.)")
            fig3_lin.tight_layout()
            fig3_lin.show()

if __name__ == '__main__':
    root = tk.Tk()
    app = TRPLFittingApp(root)
    root.mainloop()