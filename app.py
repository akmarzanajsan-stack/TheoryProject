import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import pandas as pd

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="BinomPlay",
    page_icon="🎲",
    layout="wide"
)

# ==================== TITLE ====================
st.title("🎲 BinomPlay")
st.markdown("Binomial Distribution Calculator")
st.markdown("---")

# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ Parameters")

# n - Number of trials
n = st.sidebar.number_input(
    "Number of trials (n):",
    min_value=1,
    max_value=200,
    value=10,
    step=1
)

# p - Success probability
p = st.sidebar.slider(
    "Success probability (p):",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.01
)

# k - Number of successes
k = st.sidebar.number_input(
    "Number of successes (k):",
    min_value=0,
    max_value=n,
    value=min(6, n),
    step=1
)

# Mode
mode = st.sidebar.radio(
    "Calculation type:",
    options=["P(X = k)", "P(X >= k)", "P(X <= k)"],
    index=1
)

# ==================== CALCULATE ====================
dist = binom(n, p)

if mode == "P(X = k)":
    result = dist.pmf(k)
    label = f"P(X = {k})"
elif mode == "P(X >= k)":
    result = dist.sf(k - 1)
    label = f"P(X >= {k})"
else:
    result = dist.cdf(k)
    label = f"P(X <= {k})"

# ==================== RESULT ====================
st.markdown(f"## 📊 {label} = **{result*100:.2f}%**")

# Only E[X]
col1, col2 = st.columns([1, 3])
with col1:
    st.metric("Expected value E[X]", f"{n*p:.2f}")
with col2:
    st.markdown(f"""
    <div style="padding-top: 20px; color: #666; font-size: 14px;">
        💡 <b>Mean</b> — the average number of successes you expect.
        Calculated as <b>n × p = {n} × {p} = {n*p:.2f}</b>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== CHART ====================
st.markdown("### 📈 Distribution Chart")

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(0, n + 1)
y = dist.pmf(x) * 100

# Colors based on mode
colors = []
for i in x:
    highlight = False
    if mode == "P(X = k)" and i == k:
        highlight = True
    elif mode == "P(X >= k)" and i >= k:
        highlight = True
    elif mode == "P(X <= k)" and i <= k:
        highlight = True
    colors.append('#764ba2' if highlight else '#c5cae9')

bars = ax.bar(x, y, color=colors, edgecolor='white', linewidth=1.5)

# Add value labels above bars
for bar, val in zip(bars, y):
    if val > 0.5:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{val:.1f}%',
            ha='center',
            fontsize=9,
            fontweight='bold',
            color='#333'
        )

# Add E[X] line
ax.axvline(x=n*p, color='red', linestyle='--', alpha=0.7, label=f'E[X] = {n*p:.2f}')

ax.set_xlabel('k — Number of successes', fontsize=12, fontweight='bold')
ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
ax.set_title(f'Binomial Distribution (n={n}, p={p})', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend()
ax.set_xticks(range(0, n + 1, max(1, n // 20)))

st.pyplot(fig)

st.markdown("---")

# ==================== TABLE ====================
st.markdown("### 📋 Full Table")

st.markdown("""
**How to read the table:**
- **k** — number of successes (from 0 to n)
- **P(X = k)** — probability of exactly k successes
- **P(X <= k)** — probability of k or fewer successes
- **P(X >= k)** — probability of k or more successes
""")

table_data = []
for i in range(n + 1):
    table_data.append({
        "k": i,
        "P(X = k)": f"{dist.pmf(i)*100:.2f}%",
        "P(X <= k)": f"{dist.cdf(i)*100:.2f}%",
        "P(X >= k)": f"{dist.sf(i-1)*100:.2f}%"
    })

st.dataframe(
    pd.DataFrame(table_data),
    use_container_width=True,
    hide_index=True,
    height=400
)

# ==================== FOOTER ====================
st.markdown("---")
st.caption("🎲 BinomPlay - Binomial Distribution Calculator")
