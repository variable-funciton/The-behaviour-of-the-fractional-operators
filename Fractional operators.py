#!/usr/bin/env python
# coding: utf-8

# In[27]:


import sympy as sym
import matplotlib.pyplot as plt
import numpy as np
from sympy import latex

# --- 1. グローバル設定 (Global Settings) ---
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
FontSize = 20
figsize_std = (11.69, 8.27)

# --- 2. 関数定義 (Function Definitions) ---

def chi(x):
    """Characteristic function on [0, 1]"""
    return np.where((x > 0) & (x < 1), 1.0, 0.0)

def f_orlicz(x):
    """Orlicz Maximal Function (Exp type)"""
    return np.piecewise(x, [(x > 0) & (x < 1), (x <= 0) | (x >= 1)],
        [lambda x: 1 / np.log(2), 
         lambda x: 1 / np.log(1.5 + np.abs(x - 0.5))])

def g_hl(x):
    """Hardy-Littlewood Maximal Function"""
    return np.piecewise(x, [(x > 0) & (x < 1), (x <= 0) | (x >= 1)],
        [lambda x: 1.0, 
         lambda x: 1 / (0.5 + np.abs(x - 0.5))])

def MM_iterated(x):
    """Iterated Maximal Function M^2"""
    return np.piecewise(x, [(x > 0) & (x < 1), (x <= 0) | (x >= 1)],
        [lambda x: 1.0, 
         lambda x: (1 + np.log(np.abs(x - 0.5) + 0.5)) / (0.5 + np.abs(x - 0.5))])

def M_M_alpha(x, A):
    """Composition: M after M_alpha"""
    # Use small epsilon to avoid division by zero at endpoints if necessary
    return np.piecewise(x, [(x > 0) & (x < 1), (x <= 0), (x >= 1)],
        [1.0, 
         lambda x: (1 - 1/A + 1/A * (1-x)**A) / (1-x + 1e-15), 
         lambda x: (1 - 1/A + x**A / A) / (x + 1e-15)])

def M_alpha_M(x, A, B):
    """Composition: M_alpha after M"""
    return np.piecewise(x, [(x > 0) & (x < 1), (x <= 0), (x >= 1)],
        [1.0, 
         lambda x: (1-x)**B * (1 + np.log(1-x + 1e-15)), 
         lambda x: x**B * (1 + np.log(x + 1e-15))])

def fractional_integral(x, A):
    """Fractional Integral I_alpha"""
    return np.piecewise(x, [(x > 0) & (x < 1), (x <= 0), (x >= 1)],
        [lambda x: (x**A + (1-x)**A) / A, 
         lambda x: ((1-x)**A - np.abs(-x)**A) / A, 
         lambda x: (x**A - (x-1)**A) / A])

def fractional_maximal(x, A):
    """Fractional Maximal Function M_alpha"""
    return np.piecewise(x, [(x > 0) & (x < 1), (x <= 0), (x >= 1)],
        [1.0, lambda x: (1-x)**(A-1), lambda x: x**(A-1)])

# --- 3. 描画セクション (Plotting) ---

# FIGURE 1: Orlicz vs Hardy-Littlewood
a, b = -14, 15
x = np.linspace(a, b, 1000)
plt.figure(figsize=figsize_std, dpi=100)
plt.plot(x, f_orlicz(x), "--", color="0.1", label=r'$y = M_{\exp}(\chi_{[0,1]})(x)$')
plt.plot(x, g_hl(x), ":", color="0.1", label=r'$y = M(\chi_{[0,1]})(x)$')
plt.plot(x, chi(x), lw=3, color="0.1", label=r'$y = \chi_{[0,1]}(x)$')
plt.title(rf'The range is ${a} \leq x \leq {b}$', fontsize=FontSize)
plt.xlabel(r'$x$', fontsize=FontSize); plt.ylabel(r'$y$', fontsize=FontSize)
plt.grid(True, linestyle=':', alpha=0.6); plt.legend(fontsize=14)
plt.savefig("Orlicz_maximal_functions.pdf", bbox_inches="tight")

# FIGURE 2: Compositions (M and M_alpha)
A_val = sym.Rational(1, 5)
A_val_f = float(A_val.evalf())
B_val = A_val_f - 1
a, b = -3, 4
x = np.linspace(a, b, 1000)
alpha_sym = sym.symbols(r"\alpha")
x0_expr, y0_expr = sym.exp(alpha_sym / (1 - alpha_sym)), sym.exp(-alpha_sym) / (1 - alpha_sym)
a0_pt, b0_pt = np.exp(-A_val_f / B_val), -np.exp(-A_val_f) / B_val

plt.figure(figsize=figsize_std, dpi=100)
plt.plot(x, MM_iterated(x), "--", color="0.1", label=r'$y=M^{2}(\chi_{[0,1]})(x)$')
plt.plot(x, M_M_alpha(x, A_val_f), ":", color="0.1", label=r'$y=M \circ M_{\alpha}(\chi_{[0,1]})(x)$')
plt.plot(x, M_alpha_M(x, A_val_f, B_val), color="0.4", label=r'$y=M_{\alpha} \circ M(\chi_{[0,1]})(x)$')
plt.plot(a0_pt, b0_pt, 'ko', markersize=8, label=rf'Max: $(x_0, y_0) = ({latex(x0_expr)}, {latex(y0_expr)})$')
plt.plot(1-a0_pt, b0_pt, 'ko', markersize=8)
plt.plot(x, chi(x), lw=3, color="0.1", label=r'$y = \chi_{[0,1]}(x)$')
plt.title(rf'The case of $\alpha={latex(A_val)}$ on ${a} \leq x \leq {b}$', fontsize=FontSize)
plt.xlabel(r'$x$', fontsize=FontSize); plt.ylabel(r'$y$', fontsize=FontSize)
plt.legend(loc='lower right', fontsize=12); plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig("MalpM_and_MMalp.pdf", bbox_inches="tight")

# FIGURE 3: Fractional Integral and Fractional Maximal
alpha_rat = sym.Rational(2, 3)
alpha_f = float(alpha_rat.evalf())
a, b = -2, 3
x = np.linspace(a, b, 1000)

plt.figure(figsize=figsize_std, dpi=100)
plt.plot(x, fractional_integral(x, alpha_f), "--", color="0.1", label=r'$y = I_{\alpha}[\chi_{[0,1]}](x)$')
plt.plot(x, fractional_maximal(x, alpha_f), ":", color="0.1", label=r'$y = M_{\alpha}[\chi_{[0,1]}](x)$')
plt.plot(x, chi(x), lw=3, color="0.1", label=r'$y = \chi_{[0,1]}(x)$')
plt.title(rf'The case of $\alpha = {latex(alpha_rat)}$ on ${a} \leq x \leq {b}$', fontsize=FontSize)
plt.xlabel(r'$x$', fontsize=FontSize); plt.ylabel(r'$y$', fontsize=FontSize)
plt.grid(True, linestyle=':', alpha=0.6); plt.legend(fontsize=14)
plt.savefig("Fractional_integrals.pdf", bbox_inches="tight")


# In[ ]:





# In[ ]:




