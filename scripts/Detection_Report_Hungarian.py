"""
Detection Evaluation Pipeline (IoU + Plots + Statistical Summary + TXT/PDF Report)

- Load & clean prediction, ground truth, and AOI (transect) geometries
- CRS alignment
- Spatial join (assign objects to transects)
- Only evaluate transects that contain BOTH predictions AND ground truth
  (valid_aoi_ids = pred_ids & gt_ids)
- IoU matrix + Hungarian matching per transect
- TP/FP/FN classification
- Micro AND macro metrics
- TXT report with global summary + per-transect table
- PDF report with global summary + per-transect detail pages

Note on Micro vs. Macro:
- Micro: TP/FP/FN are summed globally, every single detection counts equally.
  Large transects with many objects dominate the result.
- Macro: Precision/Recall/F1 are computed PER transect and then averaged.
  Every transect counts equally, regardless of how many objects it contains.

Assumption: transects do not overlap, so
predicate="intersects" in the spatial join is not an issue (no double assignment).
"""

from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from matplotlib.patches import Patch
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image, Table, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4, landscape

from PIL import Image as PILImage

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =====================================================
# CONFIG
# =====================================================

prediction_path = Path("/Volumes/Crucial X9/Masterarbeit/data/python_outputs/predictions_processed/clipped/clipped_predictions_2023.gpkg")
ground_truth_path = Path("/Volumes/Crucial X9/Masterarbeit/data/python_outputs/gt_processed/new_gt_2023_assigned.gpkg")
aoi_path = Path("/Volumes/Crucial X9/Masterarbeit/data/python_outputs/gt_processed/new_transects_2023.gpkg")

prediction_layer = "clipped_predictions_2023"
ground_truth_layer = "assigned"
transects_layer = "new_transects_2023"

IOU_THRESHOLD = 0.5

out_dir = Path("/Volumes/Crucial X9/Masterarbeit/data/python_outputs/detection_metrics/detection_metrics_23/plots")
report_dir = Path("/Volumes/Crucial X9/Masterarbeit/data/python_outputs/detection_metrics/detection_metrics_23/report")

out_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

# =====================================================
# COLOR SYSTEM
# =====================================================

COLORS = {
    "aoi": "#000000",
    "gt": "#0072B2",
    "pred": "#E69F00",
    "tp": "#CC79A7",
    "fp": "#009E73",
    "fn": "#F0E442"
}

# =====================================================
# LOAD DATA
# =====================================================

pred_gdf = gpd.read_file(prediction_path, layer=prediction_layer)
gt_gdf = gpd.read_file(ground_truth_path, layer=ground_truth_layer)
aoi_gdf = gpd.read_file(aoi_path, layer=transects_layer)

# =====================================================
# CLEAN GEOMETRIES
# =====================================================

def clean_geom(gdf):
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf

pred_gdf = clean_geom(pred_gdf)
gt_gdf = clean_geom(gt_gdf)

# =====================================================
# FIX IDS (float -> int, prevents 50.0 vs 50 inconsistencies)
# =====================================================

aoi_gdf["transect_id"] = aoi_gdf["transect_id"].astype(int)

# =====================================================
# CRS ALIGNMENT
# =====================================================

pred_gdf = pred_gdf.to_crs(aoi_gdf.crs)
gt_gdf = gt_gdf.to_crs(aoi_gdf.crs)

# =====================================================
# SPATIAL JOIN
# =====================================================

pred_joined = gpd.sjoin(
    pred_gdf,
    aoi_gdf[["transect_id", "geometry"]],
    how="left",
    predicate="intersects"
)

gt_joined = gpd.sjoin(
    gt_gdf,
    aoi_gdf[["transect_id", "geometry"]],
    how="left",
    predicate="intersects"
)

pred_joined = pred_joined.dropna(subset=["transect_id"])
gt_joined = gt_joined.dropna(subset=["transect_id"])

pred_joined["transect_id"] = pred_joined["transect_id"].astype(int)
gt_joined["transect_id"] = gt_joined["transect_id"].astype(int)

# =====================================================
# VALID AOIS
# Only transects with BOTH predictions AND ground truth
# (intentional choice per user's requirement - no pure FN-/FP-only transects)
# =====================================================

pred_ids = set(map(int, pred_joined["transect_id"].unique()))
gt_ids = set(map(int, gt_joined["transect_id"].unique()))

valid_aoi_ids = pred_ids & gt_ids

# =====================================================
# OPTIONAL AOIS
# Manually specify which transect_id values to evaluate
# =====================================================
#regeneration area
#SELECTED_TRANSECT_IDS = [50, 51] 

#mature stand
#SELECTED_TRANSECT_IDS = [52, 53, 54, 55, 57, 59, 61, 64, 65, 74, 81, 91, 93, 94, 95, 97, 98]

#dominating deciduous 
#SELECTED_TRANSECT_IDS = [54, 61, 64, 65, 91, 94, 97, 98]

#dominating conifers
#SELECTED_TRANSECT_IDS = [50, 51, 52, 53, 55, 57, 59, 95]

#mixed
#SELECTED_TRANSECT_IDS = [74, 81, 93] 


#SELECTED_TRANSECT_IDS = 
#valid_aoi_ids = set(SELECTED_TRANSECT_IDS)

# =====================================================
# IOU FUNCTION
# =====================================================

def iou(a, b):
    if a is None or b is None:
        return 0
    if a.is_empty or b.is_empty:
        return 0
    try:
        inter = a.intersection(b).area
        union = a.union(b).area
        if union == 0:
            return 0
        return inter / union
    except Exception:
        return 0

# =====================================================
# SAFE IMAGE SCALER (for PDF, no cropping)
# =====================================================

def scale_image(path, max_width=260, max_height=320):
    img = PILImage.open(path)
    w, h = img.size
    scale = min(max_width / w, max_height / h)
    return Image(str(path), width=w * scale, height=h * scale)

# =====================================================
# METRICS STORAGE
# =====================================================

TP = FP = FN = 0
iou_list = []
per_aoi_results = []

# =====================================================
# MAIN LOOP: matching + plots + metrics in a single pass
# =====================================================

for aoi in sorted(valid_aoi_ids):

    aoi = int(aoi)

    preds = pred_joined[pred_joined["transect_id"] == aoi].reset_index(drop=True)
    gts = gt_joined[gt_joined["transect_id"] == aoi].reset_index(drop=True)

    n_pred = len(preds)
    n_gt = len(gts)

    # Because valid_aoi_ids = pred_ids & gt_ids, n_pred > 0 and n_gt > 0 are
    # guaranteed here - no dead code needed for n_pred==0 / n_gt==0 cases.

    # -----------------------------------------------
    # IOU MATRIX + HUNGARIAN MATCHING
    # -----------------------------------------------

    iou_matrix = np.zeros((n_gt, n_pred))

    for gi, gt_geom in enumerate(gts.geometry):
        for pi, pred_geom in enumerate(preds.geometry):
            iou_matrix[gi, pi] = iou(pred_geom, gt_geom)

    gt_idx, pred_idx = linear_sum_assignment(-iou_matrix)

    matched_gt = set()
    matched_pred = set()
    local_iou = []

    for gi, pi in zip(gt_idx, pred_idx):
        if iou_matrix[gi, pi] >= IOU_THRESHOLD:
            matched_gt.add(gi)
            matched_pred.add(pi)
            local_iou.append(iou_matrix[gi, pi])

    local_tp = len(matched_pred)
    local_fp = n_pred - len(matched_pred)
    local_fn = n_gt - len(matched_gt)

    TP += local_tp
    FP += local_fp
    FN += local_fn
    iou_list.extend(local_iou)

    precision = local_tp / (local_tp + local_fp) if (local_tp + local_fp) > 0 else 0
    recall = local_tp / (local_tp + local_fn) if (local_tp + local_fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    mean_iou_local = np.mean(local_iou) if local_iou else 0

    per_aoi_results.append({
        "transect_id": aoi,
        "TP": local_tp,
        "FP": local_fp,
        "FN": local_fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MeanIoU": mean_iou_local
    })

    # -----------------------------------------------
    # PLOT 1: GT vs PRED
    # -----------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 8))

    aoi_gdf[aoi_gdf["transect_id"] == aoi].boundary.plot(
        ax=ax, color=COLORS["aoi"], linewidth=1
    )
    gts.plot(ax=ax, facecolor="none", edgecolor=COLORS["gt"], linewidth=2)
    preds.plot(ax=ax, facecolor="none", edgecolor=COLORS["pred"], linewidth=2, linestyle="--")

    ax.set_title(f"Transect {aoi} - GT vs Pred")
    ax.set_axis_off()
    ax.legend(handles=[
        Patch(edgecolor=COLORS["gt"], facecolor="none", label="GT"),
        Patch(edgecolor=COLORS["pred"], facecolor="none", label="Pred")
    ])

    img1_path = out_dir / f"transect_{aoi}_gt_vs_pred.png"
    plt.savefig(img1_path, dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------------------------
    # PLOT 2: TP / FP / FN
    # -----------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 8))

    aoi_gdf[aoi_gdf["transect_id"] == aoi].boundary.plot(
        ax=ax, color=COLORS["aoi"], linewidth=1
    )

    if len(matched_pred) > 0:
        preds.iloc[list(matched_pred)].plot(
            ax=ax, facecolor="none", edgecolor=COLORS["tp"], linewidth=2
        )

    fp_idx = [i for i in range(n_pred) if i not in matched_pred]
    if fp_idx:
        preds.iloc[fp_idx].plot(
            ax=ax, facecolor="none", edgecolor=COLORS["fp"], linewidth=2, linestyle="--"
        )

    fn_idx = [i for i in range(n_gt) if i not in matched_gt]
    if fn_idx:
        gts.iloc[fn_idx].plot(
            ax=ax, facecolor="none", edgecolor=COLORS["fn"], linewidth=2, linestyle=":"
        )

    ax.set_title(f"Transect {aoi} - TP / FP / FN")
    ax.set_axis_off()
    ax.legend(handles=[
        Patch(edgecolor=COLORS["tp"], facecolor="none", label="TP"),
        Patch(edgecolor=COLORS["fp"], facecolor="none", label="FP"),
        Patch(edgecolor=COLORS["fn"], facecolor="none", label="FN")
    ])

    img2_path = out_dir / f"transect_{aoi}_tp_fp_fn.png"
    plt.savefig(img2_path, dpi=300, bbox_inches="tight")
    plt.close()

# =====================================================
# SUMMARY METRICS (MICRO + MACRO)
# =====================================================

df = pd.DataFrame(per_aoi_results)
df_sorted = df.sort_values("F1", ascending=False)

# Micro: summed globally across all detections
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
mean_iou = np.mean(iou_list) if iou_list else 0

# Macro: average of per-transect values (every transect counts equally)
macro_precision = df["Precision"].mean()
macro_recall = df["Recall"].mean()
macro_f1 = df["F1"].mean()
macro_iou = df["MeanIoU"].mean()

print("\n================ DETECTION METRICS ================\n")
print("TP:", TP, "FP:", FP, "FN:", FN)
print("MICRO -> Precision:", round(precision, 4), "Recall:", round(recall, 4),
      "F1:", round(f1, 4), "Mean IoU:", round(mean_iou, 4))
print("MACRO -> Precision:", round(macro_precision, 4), "Recall:", round(macro_recall, 4),
      "F1:", round(macro_f1, 4), "Mean IoU:", round(macro_iou, 4))

# =====================================================
# TXT REPORT (global summary + per-transect table)
# =====================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
txt_path = report_dir / f"detection_report_{timestamp}.txt"

# Table header + column widths
col_widths = {
    "transect_id": 12,
    "TP": 6,
    "FP": 6,
    "FN": 6,
    "Precision": 10,
    "Recall": 10,
    "F1": 10,
    "MeanIoU": 10,
}

def fmt_row(values):
    return "".join(str(v).rjust(w) for v, w in zip(values, col_widths.values()))

header_row = fmt_row(["Transect", "TP", "FP", "FN", "Precision", "Recall", "F1", "MeanIoU"])
separator = "-" * len(header_row)

table_lines = [header_row, separator]
for _, row in df_sorted.iterrows():
    table_lines.append(fmt_row([
        int(row["transect_id"]),
        int(row["TP"]),
        int(row["FP"]),
        int(row["FN"]),
        f"{row['Precision']:.4f}",
        f"{row['Recall']:.4f}",
        f"{row['F1']:.4f}",
        f"{row['MeanIoU']:.4f}",
    ]))

table_text = "\n".join(table_lines)

with open(txt_path, "w") as f:
    f.write(f"""
DETECTION REPORT
============================================================

IoU Threshold: {IOU_THRESHOLD:.2f}
Transects evaluated (GT and Pred both present): {len(valid_aoi_ids)}

GLOBAL (MICRO)
------------------------------------------------------------
TP: {TP}
FP: {FP}
FN: {FN}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-score: {f1:.4f}
Mean IoU: {mean_iou:.4f}

MACRO (average across transects)
------------------------------------------------------------
Precision: {macro_precision:.4f}
Recall: {macro_recall:.4f}
F1-score: {macro_f1:.4f}
Mean IoU: {macro_iou:.4f}

PER-TRANSECT METRICS (sorted by F1, descending)
============================================================
{table_text}
""")

print(f"\nTXT report saved to:\n{txt_path}")

# =====================================================
# PDF REPORT
# =====================================================

pdf_path = report_dir / f"detection_report_{timestamp}.pdf"

doc = SimpleDocTemplate(str(pdf_path), pagesize=landscape(A4))
styles = getSampleStyleSheet()
elements = []

# ---------------- PAGE 1: global summary ----------------
elements.append(Paragraph("Detection Report", styles["Title"]))
elements.append(Spacer(1, 12))

elements.append(Paragraph(f"""
<b>IoU Threshold:</b> {IOU_THRESHOLD:.2f}<br/>
<b>Transects evaluated (GT and Pred both present):</b> {len(valid_aoi_ids)}<br/><br/>

<b>GLOBAL (MICRO)</b><br/>
TP: {TP} | FP: {FP} | FN: {FN}<br/>
Precision: {precision:.4f}<br/>
Recall: {recall:.4f}<br/>
F1: {f1:.4f}<br/>
Mean IoU: {mean_iou:.4f}<br/><br/>

<b>MACRO (average across transects)</b><br/>
Precision: {macro_precision:.4f}<br/>
Recall: {macro_recall:.4f}<br/>
F1: {macro_f1:.4f}<br/>
Mean IoU: {macro_iou:.4f}
""", styles["BodyText"]))

elements.append(PageBreak())

# ---------------- PAGE 2+: per-transect detail pages ----------------
for _, row in df_sorted.iterrows():

    aoi = int(row["transect_id"])

    img1_path = out_dir / f"transect_{aoi}_gt_vs_pred.png"
    img2_path = out_dir / f"transect_{aoi}_tp_fp_fn.png"

    # Since plots are generated in the same pass, these files should always
    # exist. Assert instead of a silent "continue" so a missing image is
    # caught immediately instead of quietly disappearing from the report.
    assert img1_path.exists() and img2_path.exists(), (
        f"Missing plot file for transect {aoi} - report would be incomplete."
    )

    im1 = scale_image(img1_path)
    im2 = scale_image(img2_path)

    table = Table([
        [im1, im2],
        [
            Paragraph(
                f"Precision: {row['Precision']:.4f}<br/>Recall: {row['Recall']:.4f}<br/>"
                f"F1: {row['F1']:.4f}<br/>IoU: {row['MeanIoU']:.4f}",
                styles["BodyText"]
            ),
            Paragraph(
                f"TP: {int(row['TP'])}<br/>FP: {int(row['FP'])}<br/>FN: {int(row['FN'])}",
                styles["BodyText"]
            )
        ]
    ])

    table.setStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 1), (-1, 1), 6),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 6),
    ])

    elements.append(Paragraph(f"<b>Transect {aoi}</b>", styles["Heading2"]))
    elements.append(table)
    elements.append(Spacer(1, 10))
    elements.append(PageBreak())

doc.build(elements)

print(f"\nPDF saved to:\n{pdf_path}")