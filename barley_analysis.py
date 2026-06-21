"""
Barley scan data analysis tool
Input: rek file path
Output: average wall thickness and average per-mm pixel intensity
"""

import numpy as np
from scipy.ndimage import binary_fill_holes, label, generate_binary_structure
from skimage.morphology import binary_closing, disk, remove_small_objects

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import ruptures as rpt
except ImportError:
    rpt = None


class RekTube:
    """Minimal local replacement for phenoct.Tube used by this analysis."""

    def __init__(self, rek_path):
        self.filename = rek_path
        self.data = self._read_rek_file(rek_path)
        self.segmented_data = None

    @staticmethod
    def _read_rek_file(rek_path):
        with open(rek_path, mode="rb") as file:
            hdr_bytes = file.read(2 * 1024)
            hdr = np.frombuffer(hdr_bytes, dtype=np.uint16)
            shape = (hdr[3], hdr[1], hdr[0])

        return np.memmap(
            rek_path,
            offset=2048,
            dtype="uint16",
            shape=shape,
            mode="r",
        )

    def segment_sample_holder(
        self,
        start_slice=0,
        stop_slice=None,
        tube_r=160,
        tube_thickness=30,
        attenuation_threshold=None,
        debug=False,
    ):
        self.segmented_data = segment_sample_holder(
            self.data,
            start_slice=start_slice,
            stop_slice=stop_slice,
            tube_r=tube_r,
            tube_thickness=tube_thickness,
            attenuation_threshold=attenuation_threshold,
            debug=debug,
        )


class TiffTube:

    def __init__(self, tiff_path):
        if tifffile is None:
            raise ImportError(
                "TIFF support requires 'tifffile'. Install it with: pip install tifffile"
            )

        self.filename = tiff_path
        self.segmented_data = tifffile.imread(tiff_path)
        self.data = None


def segment_sample_holder(data,
                          start_slice=0,
                          stop_slice=None,
                          tube_r=160,
                          tube_thickness=30,
                          attenuation_threshold=None,
                          debug=False):
    """
    Segment plant material from a REK volume by masking out the sample holder.

    This is a local extraction of phenoct.Tube.segment_sample_holder with only the
    dependencies needed by this project.
    """
    if cv2 is None:
        raise ImportError(
            "REK segmentation requires OpenCV. Install it with: pip install opencv-python"
        )

    def segment_slice(v_slice):
        if isinstance(attenuation_threshold, int):
            min_v = attenuation_threshold
            _, s_thresh = cv2.threshold(v_slice, min_v, 2 ** 16, cv2.THRESH_BINARY)
        elif isinstance(attenuation_threshold, (list, tuple)):
            min_v = attenuation_threshold[0]
            max_v = attenuation_threshold[1]
            _, s_thresh_min = cv2.threshold(v_slice, min_v, 2 ** 16, cv2.THRESH_BINARY)
            _, s_thresh_max = cv2.threshold(v_slice, max_v, 2 ** 16, cv2.THRESH_BINARY_INV)
            s_thresh = cv2.bitwise_and(s_thresh_min, s_thresh_max)
        elif attenuation_threshold is None:
            min_v = (v_slice.max() + v_slice.min()) // 2
            _, s_thresh = cv2.threshold(v_slice, min_v, 2 ** 16, cv2.THRESH_BINARY)
        else:
            raise ValueError("Please specify attenuation_threshold as an integer or a 2-item list/tuple.")

        s_thresh = s_thresh.astype("uint8")
        h, w = v_slice.shape
        tube_slice_8bit = (v_slice // 256).astype("uint8")

        circles = cv2.HoughCircles(
            tube_slice_8bit,
            cv2.HOUGH_GRADIENT,
            1,
            200,
            param1=50,
            param2=30,
            minRadius=150,
            maxRadius=0,
        )

        if circles is None or len(circles) != 1:
            if debug:
                import matplotlib.pyplot as plt

                plt.imshow(v_slice)
                plt.title("Slice")
                plt.show()
            raise RuntimeError("No sample holder circle found.")

        circles = np.round(circles[0, :]).astype("int")
        x, y, _ = circles[0]

        circ_mask = np.zeros(v_slice.shape, dtype=np.uint8)
        cv2.circle(circ_mask, (x, y), tube_r - tube_thickness, 255, 1)

        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(circ_mask, flood_mask, (h // 2, w // 2), 255)

        comb_mask = cv2.bitwise_and(circ_mask, s_thresh)

        try:
            bool_img = remove_small_objects(comb_mask.astype(bool), 10)
            final_mask = np.copy(bool_img.astype(np.uint8) * 255)
        except RuntimeError:
            final_mask = comb_mask

        return final_mask

    if stop_slice is None:
        stop_slice = data.shape[0]

    segmented_data = np.zeros(data.shape, dtype="uint16")
    iterator = range(start_slice, stop_slice)
    if tqdm is not None:
        iterator = tqdm(iterator, total=stop_slice - start_slice)

    for v_slice in iterator:
        if tqdm is not None:
            iterator.set_description(f"Segmenting slice: {v_slice}")

        img = data[v_slice, :, :]
        mask = segment_slice(img)
        masked = img.copy()
        masked[np.where(mask == 0)] = 0
        segmented_data[v_slice] = masked.reshape(img.shape)

    return segmented_data


def show_tiff_volume(tiff_path, layer_name="tiff_volume", opacity=0.7):
    """
    Open a TIFF volume in napari 3D view for quick inspection.

    Parameters:
        tiff_path: path to a .tif/.tiff volume
        layer_name: napari layer name
        opacity: image layer opacity

    Returns:
        tuple: (viewer, volume)
    """
    if tifffile is None:
        raise ImportError(
            "TIFF support requires 'tifffile'. Install it with: pip install tifffile"
        )

    try:
        import napari
    except ImportError as exc:
        raise ImportError(
            "Visualization requires 'napari'. Install it with: pip install napari"
        ) from exc

    volume = tifffile.imread(str(tiff_path))

    viewer = napari.Viewer()
    viewer.add_image(
        volume,
        name=layer_name,
        opacity=opacity,
        blending="additive",
    )
    viewer.dims.ndisplay = 3

    return viewer, volume


def analyze_barley_scan(rek_file_path, 
                       start_slice=750, 
                       stop_slice=2600, 
                       tube_r=170, 
                       tube_thickness=30, 
                       attenuation_threshold=2000,
                       dz_mm=0.084,
                       px_xy_mm=0.084,
                       boundary_method="legacy"):
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
        boundary_method: peduncle boundary method, one of:
            - "legacy": gradient-based detection
            - "ruptures": change-point detection
    
    Returns:
        dict: {
            'mean_wall_thickness_mm': average wall thickness (mm),
            'median_wall_thickness_mm': median wall thickness (mm),
            'mean_per_mm_intensity': average pixel density (mean intensity),
            'median_per_mm_intensity': median pixel density
        }
    """
    
    # 1. Load file with format compatibility
    file_path = str(rek_file_path)
    lower_path = file_path.lower()

    if lower_path.endswith((".tif", ".tiff")):
        tube = TiffTube(file_path)
    else:
        tube = RekTube(file_path)

        # 2. Segment sample holder (rek workflow)
        tube.segment_sample_holder(
            start_slice=start_slice,
            stop_slice=stop_slice,
            tube_r=tube_r,
            tube_thickness=tube_thickness,
            debug=False,
            attenuation_threshold=attenuation_threshold,
        )
    
    vol = tube.segmented_data
    
    # 3. Detect peduncle boundaries
    if boundary_method == "legacy":
        z1, z2 = detect_peduncle_boundaries(vol)
    elif boundary_method == "ruptures":
        z1, z2 = detect_peduncle_boundaries_ruptures(
            vol,
            slice_mm=dz_mm,
            max_len_mm=50.0,
        )
    else:
        raise ValueError(
            "Invalid boundary_method. Use 'legacy' or 'ruptures'."
        )
    
    print(f"z1:{z1}, z2:{z2}")
    
    # 4. Extract peduncle region
    ped_final = extract_peduncle_mask(vol, z1, z2)
    
    # 5. Calculate pixel density (mean intensity)
    per_mm_mean, per_mm_median = calculate_pixel_density(
        vol, ped_final, px_xy_mm
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
    # if drop_magnitude < drop_threshold:
    #     print(f"WARNING: Weak drop detected (magnitude={drop_magnitude:.1f})")

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


def detect_peduncle_boundaries_ruptures(
    vol,
    smooth_win=100,
    area_nonzero_threshold=15,
    model="l2",
    min_size=300,
    pen=100,
    area_ratio_threshold=0.8,
    boundary_margin_slices=50,
    fallback_head_offset=600,
    fallback_tail_offset=900,
    slice_mm=0.084,
    max_len_mm=50.0,
):
    """
    Detect peduncle boundaries using ruptures change-point detection.

    Workflow:
    1. Build per-slice area signal
    2. Detect change points with PELT
    3. Select low-area candidate interval as peduncle
    4. Apply safety margins and max-length cap
    """
    if rpt is None:
        raise ImportError(
            "Ruptures method requires 'ruptures'. Install it with: pip install ruptures"
        )

    mask = vol > 0
    slice_area = mask.sum(axis=(1, 2)).astype(float)
    nz = np.flatnonzero(slice_area > area_nonzero_threshold)
    if nz.size == 0:
        raise RuntimeError("No plant detected.")

    z_top, z_bot = int(nz[0]), int(nz[-1])

    def light_smooth(x, win=10):
        if win <= 1:
            return x.copy()
        k = np.ones(win, float) / win
        return np.convolve(x, k, mode="same")

    area_s = light_smooth(slice_area, win=smooth_win)

    signal = area_s[z_top:z_bot].reshape(-1, 1)
    algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
    bkps = algo.predict(pen=pen)

    intervals = []
    prev = 0
    for b in bkps:
        start_global = prev + z_top
        end_global = b + z_top
        avg_area = np.mean(area_s[start_global:end_global])
        intervals.append(
            {
                "start": start_global,
                "end": end_global,
                "area": avg_area,
                "len": end_global - start_global,
            }
        )
        prev = b

    global_mean_area = np.mean(area_s[z_top:z_bot])
    candidates = [
        interval
        for interval in intervals
        if interval["area"] < global_mean_area * area_ratio_threshold
    ]

    if not candidates:
        z1 = z_top + fallback_head_offset
        z2 = z_top + fallback_tail_offset
    else:
        candidates.sort(key=lambda x: x['start'])
        best_ped = candidates[0]
        
        if len(candidates) > 1 and best_ped['len'] < 150:
            best_ped = candidates[1]

        z1_raw, z2_raw = best_ped["start"], best_ped["end"]
        z1 = z1_raw + boundary_margin_slices
        z2 = z2_raw - boundary_margin_slices

    max_slices = int(np.ceil(max_len_mm / slice_mm))
    if z2 - z1 > max_slices:
        z2 = z1 + max_slices

    z1 = int(max(z_top, z1))
    z2 = int(min(z_bot, z2))

    if z2 <= z1:
        center = (z_top + z_bot) // 2
        half = max_slices // 2
        z1 = int(max(z_top, center - half))
        z2 = int(min(z_bot, center + half))

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


def calculate_pixel_density(vol, ped_mask, px_xy_mm):
    """
    Calculate mean pixel density (average pixel intensity)
    
    Updated algorithm:
    - Computes mean pixel intensity per slice (density = pixel_sum / pixel_count)
    - Filters out slices with abnormal cross-sectional area
    - Returns average density across valid slices
    """
    ped_mask = ped_mask.astype(bool)
    Z = vol.shape[0]

    # ---- per-slice stats inside peduncle ----
    area_px = ped_mask.sum(axis=(1, 2))  # per-slice area (pixels)
    mean_density = np.zeros(Z, dtype=np.float64)  # per-slice mean density
    has_ped = area_px > 0

    for z in np.flatnonzero(has_ped):
        m = ped_mask[z]
        # Calculate mean density = sum(pixel_values) / count(pixels)
        mean_density[z] = vol[z][m].mean()

    # ---- slice-level outlier removal by area (MAD) ----
    area_mm2 = area_px * (px_xy_mm**2)
    a = area_mm2[has_ped]
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    keep = np.abs(area_mm2 - med) < 2 * mad
    valid = has_ped & keep

    # ---- summary metrics on the kept slices ----
    sel = np.flatnonzero(valid)
    density_mean = mean_density[sel].mean() if sel.size else 0.0
    density_median = np.median(mean_density[sel]) if sel.size else 0.0

    return density_mean, density_median


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
        print("Usage: python barley_analysis.py <rek_file_path> [legacy|ruptures]")
        sys.exit(1)
    
    rek_file = sys.argv[1]
    boundary_method = sys.argv[2] if len(sys.argv) >= 3 else "legacy"
    
    print(f"Analyzing: {rek_file}")
    print(f"Boundary method: {boundary_method}")
    results = analyze_barley_scan(rek_file, boundary_method=boundary_method)
    
    print("\n=== Analysis Results ===")
    print(f"Mean wall thickness: {results['mean_wall_thickness_mm']:.3f} mm")
    print(f"Median wall thickness: {results['median_wall_thickness_mm']:.3f} mm")
    print(f"Mean pixel density: {results['mean_per_mm_intensity']:.2f}")
    print(f"Median pixel density: {results['median_per_mm_intensity']:.2f}")
