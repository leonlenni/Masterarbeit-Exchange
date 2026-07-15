"""
Assign Reference Points to Polygons (Predictions AND Ground Truth)

For both the predicted tree crown polygons and the ground truth polygons,
this script assigns each polygon the single field-measured reference tree
(point) that most plausibly corresponds to it.

Matching logic per polygon set (run independently for predictions and GT,
each with its own pool of available points):

1. Spatial join: find all reference points that fall "within" each polygon.
2. Unique matches: polygons with exactly one candidate point get it directly.
3. Multiple candidates: among remaining candidates, keep only those within
   DBH_THRESHOLD of the largest DBH (the model most likely captured the
   dominant/overstory tree). If more than one point remains in that top
   tier, break the tie by picking the one closest to the polygon's
   representative point.
4. Points are removed from the pool once assigned, so no point is reused
   within the same polygon set.
5. calculate the polygon's own area in square meters (planar) as new varibale 
   calculated_area_m2
6. Polygons with no valid area (calculated_area_m2 is NA - e.g. because the
   CRS wasn't UTM - or a degenerate zero-area geometry) are omitted from the
   final output entirely.

"""

import geopandas as gpd
import pandas as pd

# ============================================================
# FILES
# ============================================================

POINT_FILE = "/Volumes/Crucial X9/Masterarbeit/data/refrence_plot/260121_FMS_data.gpkg"
POINT_LAYER = "FMS_data_2023"

PRED_FILE = "/Volumes/Crucial X9/Masterarbeit/data/python_outputs/predictions_processed/clipped/clipped_predictions_2023.gpkg"
PRED_LAYER = "clipped_predictions_2023"
PRED_OUTPUT = "/Volumes/Crucial X9/Masterarbeit/data/python_outputs/predictions_processed/assigned/clipped_predictions_2023_assigned.gpkg"

GT_FILE = "/Volumes/Crucial X9/Masterarbeit/data/python_outputs/gt_processed/new_gt_2023.gpkg"
GT_LAYER = "new_gt_2023"
GT_OUTPUT = "/Volumes/Crucial X9/Masterarbeit/data/python_outputs/gt_processed/new_gt_2023_assigned.gpkg"

DBH_THRESHOLD = 0.1

# Attributes to carry over from the matched reference point onto the polygon
POINT_ATTR_COLUMNS = ["id", "species", "DBH_mm", "height_m", "tree_status"]

# ============================================================
# CRS CHECK
# ============================================================

def is_utm_crs(crs):
    """
    Returns True only if `crs` looks like a projected UTM CRS with meter
    units - the only case where polygon.geometry.area is directly usable
    as square meters. Anything geographic (lat/lon) or non-UTM/non-metric
    is rejected.
    """
    if crs is None:
        return False
    if crs.is_geographic:
        return False
    try:
        units = crs.axis_info[0].unit_name
    except Exception:
        units = None
    if units != "metre":
        return False
    name = (crs.name or "").lower()
    return "utm" in name

# ============================================================
# CORE ASSIGNMENT FUNCTION (shared by predictions and GT)
# ============================================================

def assign_points_to_polygons(polys, points, dbh_threshold=DBH_THRESHOLD):
    """
    Assign each polygon in `polys` the single best-matching point from
    `points`, based on spatial containment and DBH dominance (distance is
    used only internally to break ties, not stored in the output).
    Returns a new GeoDataFrame (copy of polys) with the assignment columns
    added. Does not mutate the input.
    """

    polys = polys.reset_index(drop=True).copy()
    points = points.reset_index(drop=True).copy()

    polys["poly_idx"] = polys.index
    points["point_idx"] = points.index

    polys["rep_point"] = polys.geometry.representative_point()

    # Area of each polygon in square meters - only computed if the CRS is
    # confirmed to be a projected UTM CRS with meter units. Otherwise the
    # column is skipped (left as NaN) and a message is printed, since the
    # raw .area value would not actually be in square meters.
    if is_utm_crs(polys.crs):
        polys["calculated_area_m2"] = polys.geometry.area.round(2)
    else:
        print(f"Skipping calculated_area_m2 - CRS is not UTM (found: {polys.crs}).")
        polys["calculated_area_m2"] = pd.NA

    # --------------------------------------------------
    # SPATIAL JOIN
    # --------------------------------------------------

    join_cols = ["point_idx", "geometry"] + POINT_ATTR_COLUMNS
    join = gpd.sjoin(
        points[join_cols],
        polys[["poly_idx", "geometry"]],
        predicate="within",
        how="inner"
    )

    candidate_dict = (
        join.groupby("poly_idx")["point_idx"]
        .apply(list)
        .to_dict()
    )

    # --------------------------------------------------
    # OUTPUT COLUMNS
    # --------------------------------------------------

    polys["reference_id"] = pd.NA
    polys["assignment_rule"] = None
    for col in POINT_ATTR_COLUMNS:
        if col == "id":
            continue  # already covered by reference_id
        polys[col] = pd.NA

    available_points = set(points["point_idx"])

    def apply_match(poly_idx, pt_row, rule):
        # Note: distance is intentionally not stored anymore - it is only
        # used internally in Step 2 to break ties among similarly-sized
        # candidate trees, not written to the output.
        polys.loc[poly_idx, "reference_id"] = pt_row["id"]
        polys.loc[poly_idx, "assignment_rule"] = rule
        for col in POINT_ATTR_COLUMNS:
            if col == "id":
                continue
            polys.loc[poly_idx, col] = pt_row[col]

    # --------------------------------------------------
    # STEP 1 - UNIQUE MATCHES
    # --------------------------------------------------

    for poly_idx, candidates in candidate_dict.items():

        candidates = [p for p in candidates if p in available_points]

        if len(candidates) == 1:

            pt = candidates[0]
            pt_row = points.loc[pt]

            apply_match(poly_idx, pt_row, "unique")
            available_points.remove(pt)

    # --------------------------------------------------
    # STEP 2 - MULTIPLE MATCHES
    # --------------------------------------------------

    remaining = polys[polys["reference_id"].isna()]

    for poly_idx in remaining["poly_idx"]:

        if poly_idx not in candidate_dict:
            polys.loc[poly_idx, "assignment_rule"] = "no_match"
            continue

        candidates = [
            p for p in candidate_dict[poly_idx]
            if p in available_points
        ]

        if len(candidates) == 0:
            polys.loc[poly_idx, "assignment_rule"] = "no_match"
            continue

        rep = polys.loc[poly_idx, "rep_point"]

        rows = []
        for pt in candidates:
            rows.append({
                "point_idx": pt,
                "dbh": points.loc[pt, "DBH_mm"],
                "distance": rep.distance(points.loc[pt, "geometry"])
            })

        df = pd.DataFrame(rows)

        # FIX: drop candidates with missing DBH before ranking, otherwise
        # an all-NaN group produces an empty "top" slice and crashes below.
        df = df.dropna(subset=["dbh"])

        if df.empty:
            polys.loc[poly_idx, "assignment_rule"] = "no_dbh_data"
            continue

        df = df.sort_values("dbh", ascending=False)
        max_dbh = df.iloc[0]["dbh"]

        top = df[df["dbh"] >= max_dbh * (1 - dbh_threshold)]

        if len(top) == 1:
            best = top.iloc[0]
            rule = "largest_dbh"
        else:
            best = top.sort_values("distance").iloc[0]
            rule = "largest_dbh + distance"

        best_pt = int(best["point_idx"])
        pt_row = points.loc[best_pt]

        apply_match(poly_idx, pt_row, rule)
        available_points.remove(best_pt)

    # --------------------------------------------------
    # CLEANUP
    # --------------------------------------------------

    polys = polys.drop(columns=["poly_idx", "rep_point"])

    # Omit polygons with no valid area (NA - e.g. CRS wasn't UTM - or a
    # degenerate zero-area geometry).
    before = len(polys)
    polys = polys[
        polys["calculated_area_m2"].notna() & (polys["calculated_area_m2"] > 0)
    ].reset_index(drop=True)
    dropped = before - len(polys)
    if dropped > 0:
        print(f"Omitted {dropped} polygon(s) with no valid area.")

    return polys


def print_summary(label, polys):
    assigned = polys["reference_id"].notna().sum()
    unassigned = polys["reference_id"].isna().sum()

    print(f"\n==============================")
    print(f"Finished: {label}")
    print(f"==============================")
    print(f"Assigned   : {assigned}")
    print(f"Unassigned : {unassigned}")
    print(polys["assignment_rule"].value_counts(dropna=False))


# ============================================================
# LOAD SHARED POINT DATA
# ============================================================

print("Loading reference points...")
points = gpd.read_file(POINT_FILE, layer=POINT_LAYER)

# ============================================================
# PREDICTIONS
# ============================================================

print("\nLoading predictions...")
pred_polys = gpd.read_file(PRED_FILE, layer=PRED_LAYER)

if pred_polys.crs != points.crs:
    pred_points = points.to_crs(pred_polys.crs)
else:
    pred_points = points

print("Assigning points to predictions...")
pred_assigned = assign_points_to_polygons(pred_polys, pred_points)

pred_assigned.to_file(PRED_OUTPUT, layer="assigned", driver="GPKG")
print_summary("Predictions", pred_assigned)
print("Output:", PRED_OUTPUT)

# ============================================================
# GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")
gt_polys = gpd.read_file(GT_FILE, layer=GT_LAYER)

if gt_polys.crs != points.crs:
    gt_points = points.to_crs(gt_polys.crs)
else:
    gt_points = points

print("Assigning points to ground truth...")
gt_assigned = assign_points_to_polygons(gt_polys, gt_points)

gt_assigned.to_file(GT_OUTPUT, layer="assigned", driver="GPKG")
print_summary("Ground Truth", gt_assigned)
print("Output:", GT_OUTPUT)