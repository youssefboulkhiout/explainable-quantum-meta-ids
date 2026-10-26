"""Import-time environment set-up of the original script (headless
Matplotlib backend, seaborn theme, warning filter). Imported first by the
package so the order of side effects matches the single-file script."""
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=0.9)
    HAS_SNS = True
except Exception:
    HAS_SNS = False

warnings.filterwarnings("ignore")
