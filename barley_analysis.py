"""
Barley scan data analysis tool
Input: rek file path
Output: average wall thickness and average per-mm pixel intensity
"""

import numpy as np
from scipy.ndimage import binary_fill_holes, label, generate_binary_structure
from skimage.morphology import binary_closing, disk, remove_small_objects
import phenoct


def analyze_barley_scan(rek_file_path, 
                       start_slice=750, 
                       stop_slice=2600, 
                       tube_r=170, 
                       tube_thickness=30, 
                       attenuation_threshold=2000,
                       dz_mm=0.084,
                       px_xy_mm=0.084):
    """
    Analyze barley scan data to calculate average wall thickness and average per-mm pixel intensity
    
    Parameters:
        rek_file_path: path to rek file
        start_slice: starting slice index
        stop_slice: ending slice index
        tube_r: tube radius (160-180)
        tube_thickness: tube wall thickness (25-35)
        attenuation_threshold: attenuation threshold
        dz_mm: Z-axis pixel size (mm/px)
        px_xy_mm: XY plane pixel size (mm/px)
    
    Returns:
        dict: {
            'mean_wall_thickness_mm': average wall thickness (mm),
            'median_wall_thickness_mm': median wall thickness (mm),
            'mean_per_mm_intensity': average per-mm pixel intensity,
            'median_per_mm_intensity': median per-mm pixel intensity
        }
    """
    
    # 1. Load rek file
    tube = phenoct.Tube(rek_file_path)
    
    # 2. Segment sample holder
    tube.segment_sample_holder(
        start_slice=start_slice, 
        stop_slice=stop_slice, 
        tube_r=tube_r, 
        tube_thickness=tube_thickness, 
        debug=False, 
        attenuation_threshold=attenuation_threshold
    )
    
    vol = tube.segmented_data
    
    # 3. Detect peduncle boundaries
    z1, z2 = detect_peduncle_boundaries(vol)
    
    # 4. Extract peduncle region
    ped_final = extract_peduncle_mask(vol, z1, z2)
    
    # 5. Calculate per-mm pixel intensity
    per_mm_mean, per_mm_median = calculate_per_mm_intensity(
        vol, ped_final, dz_mm, px_xy_mm
    )
    
    # 6. Calculate wall thickness
    mean_thickness, median_thickness = calculate_wall_thickness(
        ped_final, px_xy_mm
    )
    
    return {
        'mean_wall_thickness_mm': mean_thickness,
        'median_wall_thickness_mm': median_thickness,
        'mean_per_mm_intensity': per_mm_mean,
        'median_per_mm_intensity': per_mm_median
    }


def detect_peduncle_boundaries(vol, smooth_win=21, slice_mm=0.084, max_len_mm=50.0,
                               head_margin_slices=12, tail_margin_slices=12,
                               drop_threshold=500, rise_threshold=200):
    """
    Detect peduncle boundaries using gradient-based analysis
    
    New approach based on morphological features:
    1. Find largest drop (spike end)
    2. Find stable low region (peduncle middle)
    3. Find largest rise (stem start)
    
    Parameters:
        vol: 3D volume data
        smooth_win: smoothing window size
        slice_mm: mm per slice
        max_len_mm: maximum peduncle length in mm
        head_margin_slices: margin slices at peduncle head
        tail_margin_slices: margin slices at peduncle tail
        drop_threshold: minimum drop magnitude to be considered significant
        rise_threshold: minimum rise magnitude to be considered significant
    """
    
    # --- Basic setup ---
    mask = vol > 0
    slice_area = mask.sum(axis=(1, 2)).astype(float)
    Z = slice_area.size

    nz = np.flatnonzero(slice_area > 0)
    if nz.size == 0:
        raise RuntimeError("No plant detected.")
    z_top, z_bot = int(nz[0]), int(nz[-1])
    span = z_bot - z_top + 1

    # --- Smoothing ---
    def smooth_1d(x, win=21):
        if win <= 1:
            return x.copy()
        k = np.ones(win, float) / win
        return np.convolve(x, k, mode="same")

    area_s = smooth_1d(slice_area, win=smooth_win)

    # --- Compute gradient ---
    grad = np.diff(area_s)

    # ==================== Gradient-based detection ====================

    # Step 1: Find the LARGEST DROP (spike end)
    search_end = z_top + int(0.6 * span)
    drop_region = grad[z_top:search_end]
    idx_drop = int(np.argmin(drop_region) + z_top)

    # Validate drop magnitude
    drop_magnitude = -grad[idx_drop]
    if drop_magnitude < drop_threshold:
        print(f"WARNING: Weak drop detected (magnitude={drop_magnitude:.1f})")

    # Step 2: Find stable LOW region after the drop (peduncle middle)
    low_start = idx_drop + 20
    low_end = z_bot - 100

    # Find the region with minimum variance (most stable)
    window_size = 200
    min_variance = float('inf')
    stable_center = None

    for z in range(low_start, max(low_start + 1, low_end - window_size)):
        win = area_s[z : z + window_size]
        var = np.var(win)
        if var < min_variance:
            min_variance = var
            stable_center = z + window_size // 2

    if stable_center is None:
        stable_center = (low_start + low_end) // 2

    # Step 3: Find the LARGEST RISE after stable region (stem start)
    rise_search_start = stable_center + 100
    rise_region = grad[rise_search_start:z_bot-10]

    if rise_region.size > 0:
        idx_rise = int(np.argmax(rise_region) + rise_search_start)
        rise_magnitude = grad[idx_rise]
        
        # Validate rise magnitude
        if rise_magnitude < rise_threshold:
            # Alternative: find where area exceeds 1.5x stable region mean
            stable_mean = np.mean(area_s[max(0, stable_center-100):min(Z, stable_center+100)])
            threshold = stable_mean * 1.5
            
            candidates = np.where(area_s[rise_search_start:z_bot] > threshold)[0]
            if len(candidates) > 0:
                idx_rise = candidates[0] + rise_search_start
            else:
                idx_rise = z_bot - 200  # Conservative fallback
    else:
        idx_rise = z_bot - 200

    # Step 4: Set boundaries
    spike_base = idx_drop
    flag_node = idx_rise

    # Apply margins
    z1 = max(z_top, spike_base + head_margin_slices)
    z2 = min(z_bot, flag_node - tail_margin_slices)

    # Ensure valid range
    if z2 <= z1:
        # Use stable region as peduncle
        z1 = max(z_top, stable_center - 200)
        z2 = min(z_bot, stable_center + 200)

    # Cap length to max_len_mm
    N5 = int(np.ceil(max_len_mm / slice_mm))
    if z2 - z1 > N5:
        # Keep the middle part
        center = (z1 + z2) // 2
        z1 = center - N5 // 2
        z2 = center + N5 // 2

    return z1, z2


def extract_peduncle_mask(vol, z1, z2):
    """
    Extract peduncle mask using 3D connectivity
    Exact copy from Cell 8 of the notebook
    """
    mask = (vol > 0)
    
    # Assume you already have z1, z2 (peduncle slice range)
    ped_slab = np.zeros_like(mask, dtype=bool)
    ped_slab[z1:z2] = mask[z1:z2]

    # Step 1: Keep only the largest 3D connected component
    ped_clean = remove_small_objects(ped_slab, min_size=3000, connectivity=3)
    conn3d = generate_binary_structure(3, 2)  # 26-neighborhood
    lab, nlab = label(ped_clean, structure=conn3d)
    if nlab > 0:
        counts = np.bincount(lab.ravel())
        keep_label = counts[1:].argmax() + 1  # index of the largest component
        ped_main = (lab == keep_label)
    else:
        ped_main = ped_clean

    # Step 2 (optional): In each slice, keep only the largest 2D connected component
    # ped_refined = np.zeros_like(ped_main, dtype=bool)
    # for z in range(z1, z2):
    #     sl = ped_main[z]
    #     if sl.sum() == 0:
    #         continue
    #     lab2d, n2 = label(sl, structure=np.ones((3,3), bool))
    #     if n2 == 0:
    #         continue
    #     cts = np.bincount(lab2d.ravel())
    #     ped_refined[z] = (lab2d == (cts[1:].argmax()+1))

    # If per-slice filtering is too strict, just use ped_main
    ped_final = ped_main
    
    return ped_final


def calculate_per_mm_intensity(vol, ped_mask, dz_mm, px_xy_mm):
    """
    Calculate per-mm pixel intensity
    Exact copy from Cell 11 of the notebook
    """
    ped_mask = ped_mask.astype(bool)
    Z = vol.shape[0]

    # ---- per-slice stats inside peduncle ----
    area_px = ped_mask.sum(axis=(1, 2))  # per-slice area (pixels)
    int_sum = np.zeros(Z, dtype=np.float64)  # per-slice integrated intensity (sum)
    has_ped = area_px > 0

    for z in np.flatnonzero(has_ped):
        m = ped_mask[z]
        int_sum[z] = vol[z][m].sum()

    # ---- slice-level outlier removal by area (IQR) ----
    area_mm2 = area_px * (px_xy_mm**2)
    a = area_mm2[has_ped]
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    keep = np.abs(area_mm2 - med) < 2 * mad
    valid = has_ped & keep

    # ---- per-mm intensity curve on the kept slices ----
    per_mm_intensity = np.zeros(Z, dtype=np.float64)
    sel = np.flatnonzero(valid)
    per_mm_intensity[sel] = int_sum[sel] / dz_mm

    # ---- summary metric (one number) ----
    per_mm_mean = per_mm_intensity[sel].mean() if sel.size else 0.0
    per_mm_median = np.median(per_mm_intensity[sel]) if sel.size else 0.0

    return per_mm_mean, per_mm_median


def calculate_wall_thickness(ped_mask, px_xy_mm):
    """
    Calculate wall thickness
    Exact copy from Cell 12 of the notebook
    """
    ped_mask = ped_mask.astype(bool)   # ensure boolean
    Z = ped_mask.shape[0]

    kept_z = []
    thick_mm = []
    R_mm_list = []
    r_mm_list = []

    for z in range(Z):
        sl0 = ped_mask[z]
        if not sl0.any():
            continue

        # 1) close tiny gaps (very light: disk radius = 1)
        sl = binary_closing(sl0, disk(1))

        # 2) fill holes -> outer shell
        outer = binary_fill_holes(sl)

        # 3) cavity of this slice (must exist to be kept)
        cavity = outer & (~sl)
        A_lumen_px = int(cavity.sum())
        if A_lumen_px <= 0:
            continue  # skip slices without a detected hole

        # 4) thickness from equivalent radii (outer minus lumen)
        A_outer_px = int(outer.sum())
        R_px = np.sqrt(A_outer_px / np.pi)
        r_px = np.sqrt(A_lumen_px / np.pi)
        t_mm = max((R_px - r_px) * px_xy_mm, 0.0)

        kept_z.append(z)
        thick_mm.append(t_mm)
        R_mm_list.append(R_px * px_xy_mm)
        r_mm_list.append(r_px * px_xy_mm)

    kept_z = np.array(kept_z, dtype=int)
    thick_mm = np.array(thick_mm, dtype=float)

    if kept_z.size:
        mean_thickness = thick_mm.mean()
        median_thickness = np.median(thick_mm)
    else:
        mean_thickness = 0.0
        median_thickness = 0.0

    return mean_thickness, median_thickness


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python barley_analysis.py <rek_file_path>")
        sys.exit(1)
    
    rek_file = sys.argv[1]
    
    print(f"Analyzing: {rek_file}")
    results = analyze_barley_scan(rek_file)
    
    print("\n=== Analysis Results ===")
    print(f"Mean wall thickness: {results['mean_wall_thickness_mm']:.3f} mm")
    print(f"Median wall thickness: {results['median_wall_thickness_mm']:.3f} mm")
    print(f"Mean per-mm pixel intensity: {results['mean_per_mm_intensity']:.2f}")
    print(f"Median per-mm pixel intensity: {results['median_per_mm_intensity']:.2f}")