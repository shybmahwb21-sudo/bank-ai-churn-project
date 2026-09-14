"""Dependency-light report serialization used by Streamlit downloads."""
from io import BytesIO
import pandas as pd

def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")

def pdf_bytes(frame: pd.DataFrame, title: str = "Bank AI report") -> bytes:
    """Create a compact PDF when matplotlib is available (optional UI feature)."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        raise RuntimeError("PDF export requires matplotlib") from exc
    output = BytesIO()
    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title(title)
        view = frame.head(35).astype(str)
        table = ax.table(cellText=view.values, colLabels=view.columns, loc="center")
        table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1, 1.2)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    return output.getvalue()
