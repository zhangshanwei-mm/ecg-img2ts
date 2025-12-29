import os
from scipy.signal import find_peaks
import cv2
import numpy as np
from scipy.signal import medfilt, savgol_filter  # You can skip using this; kept here as a backup
from matplotlib.patches import Rectangle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal
from typing import Sequence, List
from collections import Counter


px_per_mm = 2200 / 210  # Or manually adjust until waveform height fits the 10mm/mV standard
mv_per_pixel = 1 / (10 * px_per_mm)
time_per_pixel = 1 / (25 * px_per_mm)

def scan_image_files(folder_path, image_extensions={'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}):
    image_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                full_path = os.path.join(root, file)

                file_name_no_ext = os.path.splitext(file)[0]  # Remove extension
                # file is the full filename
                image_files.append((full_path, file_name_no_ext))  # (Full path, Filename)

    return image_files

def remove_background(path):
    """
    path : Input image path
    """

    manual_thresh = 180  # Lower threshold, keep darker areas (black signals)

    lead_name_regions = [
    (105, 490,130, 520),
    (1086, 490,1130, 520),

    (105, 678,130, 710),
    (1086, 678,1130, 710),

    (105, 866,130, 898),
    (1086, 866,1130, 898),

    (105, 1063,150, 1097),
    (1086, 1063,1130, 1097),

    (105, 1253,150, 1283),
    (1086, 1253,1130, 1283),

    (105, 1443,150, 1473),
    (1086, 1443,1130, 1473)

    ]

    # Read original image (keep 3 channels)
    image = cv2.imread(path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur (optional)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Keep black parts: values below threshold are foreground (signal), others are background
    _, mask = cv2.threshold(blurred, manual_thresh, 255, cv2.THRESH_BINARY_INV)

    # White background image
    background = np.full_like(image, 255)

    # Use mask to keep black signal areas, others white
    result = np.where(mask[:, :, None] == 255, image, background)

    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Binarize again to ensure values are 0 or 255
    _, final_binary = cv2.threshold(result_gray, 127, 255, cv2.THRESH_BINARY)

    # 6. Mask lead name regions
    for (x1, y1, x2, y2) in lead_name_regions:
        cv2.rectangle(final_binary, (x1, y1), (x2, y2), 255, -1)  # -1 means fill

    # 7.2 Remove single pixel noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_binary, connectivity=8)
    # stats[i, cv2.CC_STAT_AREA] is the pixel area of the i-th connected component
    for i in range(1, num_labels):  # 0 is background
        if stats[i, cv2.CC_STAT_AREA] == 1:
            final_binary[labels == i] = 0  # Or 255, depending on your foreground/background definition

    # print(f"Type {type(final_binary)}")
    # print(final_binary.shape)
    # Save processed image
    # cv2.imwrite(save_path, final_binary)
    # print(f"Saved to {save_path}")
    return final_binary

def find_peak_index(data):

    # Example data: 1D signal
    data = np.array(data)
    # Detect peaks
    peaks, _ = find_peaks(data, height=50, distance=50, prominence=1)
    # print("Peak position indices:", peaks)
    # print("Peak values:", data[peaks])

    return peaks

def detect_peaks_via_projection(
    roi_ad: np.ndarray,
    height: float | None = None,
    distance: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect peaks corresponding to "low points" in roi_ad using inverted pixel projection.

    Args:
      roi_ad   – Grayscale image sub-region, shape=(h_ad, w_ad)
      height   – Minimum height threshold for peak detection; if None, defaults to inv_proj.mean()
      distance – Minimum horizontal distance (pixels) between peaks to avoid duplicates

    Returns:
      peaks      – Array of peak indices in column direction (relative to roi_ad)
      inv_proj   – 1D array of inverted projection, useful for visualization or tuning
    """
    # 1. Sum pixels column-wise -> project
    project = np.sum(roi_ad, axis=0)         # shape (w_ad,)
    proj_norm  = project / (project.max() + 1e-7)
    # 2. Invert: max(project) - project
    inv_proj = np.max(proj_norm) - proj_norm     # Darker (blacker) columns have higher inv_proj values

    # 3. Adaptive height threshold
    if height is None:
        height = inv_proj.mean()

    # 4. Peak detection
    peaks, properties = find_peaks(
        inv_proj,
        height=height,
        distance=distance
    )

    # # Plot pixel values and peak positions to visualize if this single lead has adhesion
    # plt.figure(figsize=(10, 3)) # Plot for each border
    # plt.plot(inv_proj, label='Vertical Projection')
    # plt.plot(peaks, inv_proj[peaks], 'ro', label='Detected Valleys as Peaks')
    # plt.title("Detected Peaks (Low Points) in Inverted Signal")
    # plt.xlabel("X (Column Index)")
    # plt.ylabel("Sum of Pixel Values")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # # plt.savefig(r"D:/code/tools/dig/6x2test_overlop/"+str(x1)+"_"+str(y1)+"_"+str(x2)+"_"+str(y2)+".png")
    # plt.show()



    return peaks, inv_proj

def refine_peaks(
    ecg_mv: np.ndarray,
    peaks: list[int],
    search_radius: int = 5
) -> list[int]:
    """
    Refine peak positions by searching for the maximum signal value within ±search_radius
    around each initial peak peaks[i].

    Args:
      ecg_mv         – Digitized ECG signal (1D numpy array)
      peaks          – List of initially detected peak indices
      search_radius  – Search radius (number of sample points)

    Returns:
      new_peaks      – List of refined peak indices (deduplicated and sorted)
    """
    n = len(ecg_mv)
    refined = []
    for p in peaks:
        # Define search window
        start = max(0, p - search_radius)
        end   = min(n - 1, p + search_radius)
        # Find relative index of the max value in the window
        local_seg = ecg_mv[start:end+1]
        # np.argmax returns position of the first maximum
        offset = int(np.argmax(local_seg))
        new_p = start + offset
        refined.append(new_p)

    # Deduplicate and sort
    refined = np.array(sorted(set(refined)), dtype=int)
    return refined

def Pixel_to_signal(roi, mv_per_pixel=1 / (10 * px_per_mm)):

    h, w = roi.shape
    signal = []
    for col in range(w):
        col_data = roi[:, col]
        black_indices = np.where(col_data < 100)[0]  # Black line threshold adjustable
        if len(black_indices) > 0:
            y = np.mean(black_indices)
        else:
            y = np.nan
        signal.append(y)

    # Interpolate to fill missing values (NaN)
    signal = np.array(signal)
    nan_mask = np.isnan(signal)
    if np.any(~nan_mask):
        signal[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), signal[~nan_mask])
    else:
        signal[:] = h / 2

    # Vertical center baseline
    center = h / 2
    ecg_mv = -(signal - center) * mv_per_pixel
    return ecg_mv

def detect_S_points(
    ecg: np.ndarray,
    R_peaks: np.ndarray,
    fs: float,
    window_ms: int = 200
) -> np.ndarray:
    """
    Find the first local minimum within window_ms after each R peak as the S wave point.
    """
    S_points = []
    win = int(window_ms * fs / 1000)
    for r in R_peaks:
        start = r
        end = min(len(ecg)-1, r + win)
        segment = ecg[start:end+1]
        if len(segment) > 0:
            offset = np.argmin(segment)
            S_points.append(start + offset)
    return np.array(S_points)

def mask_baselines_v4(
    roi_ad: np.ndarray,
    ecg_mv: np.ndarray,
    R_idxs,
    S_idxs,
    masked: np.ndarray | None = None,
    search_frac: float = 0.1
) -> tuple[np.ndarray, list[dict]]:
    """
    For each pair (R_peak, S_peak), locate and mask two baseline regions:
      1) Find baseline1 within ±search_frac*h_ad around 1/4 ROI height
      2) Find baseline2 within ±search_frac*h_ad around 3/4 ROI height
      3) Between [baseline1, baseline2], select the "max row pixel sum" as the true ECG baseline
      4) Mask: Area above baseline x1..x2 & Area below baseline x3..x4

    Args:
      roi_ad      – Grayscale sub-image, shape=(h_ad, w_ad)
      ecg_mv      – Digitized ECG amplitude, len=w_ad
      R_idxs      – List or array of R peak indices
      S_idxs      – List or array of S wave indices (length must match R_idxs)
      masked      – Optional initial mask image (grayscale); if None, defaults to white background
      search_frac – Search window half-height ratio (e.g., 0.1 means ±10%)

    Returns:
      masked      – Image after masking
      coords_list – List of dictionaries for each (R, S) pair, containing x1..x4, y1..y4,
                    baseline1, baseline2, baseline (three baselines)
    """
    R_arr = np.atleast_1d(R_idxs).astype(int).flatten()
    S_arr = np.atleast_1d(S_idxs).astype(int).flatten()
    assert len(R_arr) == len(S_arr), "R_idxs and S_idxs must have the same length"

    h_ad, w_ad = roi_ad.shape
    if masked is None:
        masked = np.full_like(roi_ad, 255)

    # Row index of the "darkest" pixel in each column (signal position)
    row_idxs = np.argmin(roi_ad, axis=0).astype(int)
    # ECG first derivative (approximate slope)
    d = np.diff(ecg_mv).astype(float)
    # Sum of pixels per row, used to pick the whitest row
    row_sums = np.sum(roi_ad, axis=1).astype(float)

    # Search window bounds: ± delta around 1/4 and 3/4
    delta = int(h_ad * search_frac)
    q1 = int(h_ad * 0.25)
    q3 = int(h_ad * 0.75)
    low1, high1 = max(0, q1 - delta), min(h_ad-1, q1 + delta)
    low2, high2 = max(0, q3 - delta), min(h_ad-1, q3 + delta)

    coords_list = []
    for R_idx, S_idx in zip(R_arr, S_arr):
        # 1) Find x1, x2, x3, x4
        x1 = next((x for x in range(R_idx-1, -1, -1) if d[x] > 0), 0)
        x2 = next((x for x in range(R_idx, len(d))    if d[x] < 0), w_ad-1)
        x3 = next((x for x in range(S_idx-1, -1, -1) if d[x] < 0), 0)
        x4 = next((x for x in range(S_idx, len(d))    if d[x] > 0), w_ad-1)

        # 2) Corresponding row coordinates
        y1, y2 = row_idxs[x1], row_idxs[x2]
        y3, y4 = row_idxs[x3], row_idxs[x4]

        # 3) Select baseline1 closest to avg1 in window [low1, high1]
        avg1 = (y1 + y2) / 2
        cands1 = np.arange(low1, high1+1)
        baseline1 = int(cands1[np.abs(cands1 - avg1).argmin()])

        # 4) Select baseline2 closest to avg2 in window [low2, high2]
        avg2 = (y3 + y4) / 2
        cands2 = np.arange(low2, high2+1)
        baseline2 = int(cands2[np.abs(cands2 - avg2).argmin()])

        # 5) Between [baseline1, baseline2], select the row with max pixel sum as true baseline
        win = row_sums[baseline1:baseline2+1]
        offset = int(np.argmax(win))
        baseline = baseline1 + offset

        # # 6) Masking: Above baseline x1..x2
        # masked[0:baseline+1, x1:x2+1] = 255
        # #    Masking: Below baseline x3..x4
        # masked[baseline:h_ad, x3:x4+1] = 255
        masked[0 : h_ad, x1 : x4+1] = 255

        coords_list.append({
            "x1": x1, "y1": int(y1),
            "x2": x2, "y2": int(y2),
            "x3": x3, "y3": int(y3),
            "x4": x4, "y4": int(y4),
            "baseline1": baseline1,
            "baseline2": baseline2,
            "baseline": baseline
        })

    return masked, coords_list

def restore_rs_waves_v2(
    ecg_mv: np.ndarray,
    coords_list: list[dict]
) -> np.ndarray:
    """
    For multiple [x1, x4] intervals specified by coords_list in ecg_mv,
    delete the original signal and reconstruct using three segment slopes:
      1. R wave segment [x1, x2]: recover with positive slope (x1,y1)->(x2,y2)
      2. ST segment [x2, x3]: force recover with constant negative slope
      3. S wave segment [x3, x4]: recover with slope (x3,y3)->(x4,y4)

    Args:
      ecg_mv      – Original digitized ECG (1D array)
      coords_list – List of coordinate dictionaries, each containing x1,x2,x3,x4

    Returns:
      ecg_restored – Restored ECG (float, 1D array)
    """
    ecg_restored = ecg_mv.copy().astype(float)

    for coords in coords_list:
        x1, x2 = int(coords["x1"]), int(coords["x2"])
        x3, x4 = int(coords["x3"]), int(coords["x4"])

        # Endpoint voltages
        y1 = ecg_mv[x1]
        y2 = ecg_mv[x2]
        y3 = ecg_mv[x3]
        y4 = ecg_mv[x4]

        # First delete original signal in [x1, x4] interval
        ecg_restored[x1 : x4 + 1] = np.nan

        # 1) R wave slope
        m1 = (y2 - y1) / (x2 - x1)

        # 2) ST segment slope (force negative slope)
        raw_m2 = (y3 - y2) / (x3 - x2)
        m2 = -abs(raw_m2)

        # 3) S wave slope
        m3 = (y4 - y3) / (x4 - x3)

        # Piecewise linear interpolation
        # [x1, x2]
        xs = np.arange(x1, x2 + 1)
        ecg_restored[x1 : x2 + 1] = y1 + m1 * (xs - x1)

        # [x2, x3]
        xs = np.arange(x2, x3 + 1)
        ecg_restored[x2 : x3 + 1] = y2 + m2 * (xs - x2)

        # [x3, x4]
        xs = np.arange(x3, x4 + 1)
        ecg_restored[x3 : x4 + 1] = y3 + m3 * (xs - x3)

    return ecg_restored

def pad_arrays_to_length(arrays: Sequence[np.ndarray],
                         target_len: int,
                         axis: int = -1,
                         pad_value: float | int = 0,
                         truncate: bool = True) -> List[np.ndarray]:
    """
    Pad a batch of ndarrays with 0 (or other value) along the specified axis to target_len.

    Parameters
    ----------
    arrays : Sequence[np.ndarray]
        List of arrays to be padded. E.g., [array([..]), array([..]), ...] or
        [array(12,5000), array(12,4800), ...] etc.
    target_len : int
        Target length (along the axis direction).
    axis : int, optional
        Axis to perform padding; default -1 is fine for 1-D data.
    pad_value : float | int, optional
        Constant used for padding. Default 0.
    truncate : bool, optional
        If a sequence length > target_len:
        - True  -> Truncate to target_len
        - False -> Keep original length (return value lengths will be inconsistent)

    Returns
    -------
    List[np.ndarray]
        List of padded (or truncated) arrays. Dimensions are same as original, length unified to target_len.
    """
    padded = []
    for a in arrays:
        # Get actual length of the padding axis
        cur_len = a.shape[axis]

        # ------ Case 1: Need truncation ------
        if truncate and cur_len > target_len:
            slicer = [slice(None)] * a.ndim
            slicer[axis] = slice(0, target_len)
            a = a[tuple(slicer)]
            cur_len = target_len

        # ------ Case 2: Need padding with 0 ------
        if cur_len < target_len:
            pad_width = [(0, 0)] * a.ndim
            pad_width[axis] = (0, target_len - cur_len)
            a = np.pad(a, pad_width,
                       mode="constant",
                       constant_values=pad_value)

        padded.append(a)

    return padded

def concat_12lead(group1: np.ndarray,
                  group2: np.ndarray,
                  pad_val: float | int | Literal["nan"] = 0) -> np.ndarray:
    """
    Parameters
    ----------
    group1 : ndarray, shape (6, len_1)
    group2 : ndarray, shape (6, len_2)
        Two groups of 6-lead ECG, column direction is time.
    pad_val : float | int | "nan", optional
        Value used for blank padding; passing "nan" will automatically convert to np.nan.

    Returns
    -------
    combined : ndarray, shape (12, len_1 + len_2)
        First 6 rows are group1, last 6 rows are group2 (starting from len_1).
    """
    if group1.shape[0] != 6 or group2.shape[0] != 6:
        raise ValueError("The first dimension of both data groups must be 6 (number of leads)")

    len_1 = group1.shape[1]
    len_2 = group2.shape[1]
    T     = len_1 + len_2                       # Total time length

    fill = np.nan if pad_val == "nan" else pad_val
    combined = np.full((12, T), fill, dtype=float)

    # Write first 6 leads
    combined[:6, :len_1] = group1

    # Write last 6 leads, starting from len_1
    combined[6:, len_1:len_1 + len_2] = group2

    return combined

def save_path_6x2(digitized_leads, save_path):
    columus_1 = list()
    columus_2 = list()
    max_len_1 = 0
    max_len_2 = 0

    for i in range(6):
        temp_len_1 = digitized_leads[i].tolist() # ndarray converted to list
        if len(temp_len_1) > max_len_1:
            max_len_1 = len(temp_len_1)
    # Pad data based on max_len_1

    sig_1 = digitized_leads[0]
    sig_2 = digitized_leads[1]
    sig_3 = digitized_leads[2]
    sig_4 = digitized_leads[3]
    sig_5 = digitized_leads[4]
    sig_6 = digitized_leads[5]
    sig_7 = digitized_leads[6]
    sig_8 = digitized_leads[7]
    sig_9 = digitized_leads[8]
    sig_10 = digitized_leads[9]
    sig_11 = digitized_leads[10]
    sig_12 = digitized_leads[11]


    padded_ecg_1 = pad_arrays_to_length([sig_1, sig_2, sig_3,sig_4,sig_5,sig_6],
                                    target_len=max_len_1,
                                    pad_value=np.nan)   # Or fill np.nan

    for i in range(6):
        temp_len_2 = digitized_leads[i+6].tolist()
        if len(temp_len_2) > max_len_2:
            max_len_2 = len(temp_len_2)
    # Pad data based on max_len_2

    padded_ecg_2 = pad_arrays_to_length([sig_7, sig_8, sig_9,sig_10,sig_11,sig_12],
                                    target_len=max_len_2,
                                    pad_value=np.nan)   # Or fill np.nan

    a = np.array(padded_ecg_1)
    b = np.array(padded_ecg_2)



    ecg12 = concat_12lead(np.array(padded_ecg_1), np.array(padded_ecg_2), pad_val="nan")
    # Save as CSV file

    # Transpose so each column is Lead1..Lead12, for better readability
    df = pd.DataFrame(ecg12.T,
                    columns=[f"Lead_{i+1}" for i in range(12)])
    df.to_csv(save_path, index=False, float_format="%.6f")   # 6 decimal places are sufficient for ECG precision
    # print(f"✔ CSV saved: {save_path}")

def dig_ecg_picture(final_binary, save_path):
    # --- Set column coordinates & vertical boundaries ---
    columns = [(108, 1084), (1092, 2074)] # Only 2 columns, adjusted slightly
    y_min, y_max = 430, 1575

    projection_list = list() # Save pixel sum of each column
    projection_norm_list = list() # Save pixel sum of each column, norm
    proj_width = 300


    # --- Main processing flow ---
    valley_index_list = list()
    for i, (x1, x2) in enumerate(columns):
        # Crop original column region image
        try:
            col_img = final_binary[y_min:y_max, x1:x2]
            col_gray = final_binary[y_min:y_max, x1:x2]

            # Calculate pixel sum (invert black/white)
            signal = 255 - col_gray
            projection = np.sum(signal, axis=1) # Pixel sum (1145,)
            projection_norm = (projection / np.max(projection) * proj_width).astype(np.uint8)

            projection_list.append(projection)
            projection_norm_list.append(projection_norm)

            peak_return = find_peak_index(projection_norm) # find peak

            # Find minimum value between peaks (valleys)
            # 2. Detect minimum value (valley) between adjacent peaks
            valleys = []

            for i in range(len(peak_return) - 1):
                start = peak_return[i]
                end   = peak_return[i + 1]
                dist  = end - start

                # 1. Midpoint
                mid = start + dist // 2

                # 2. Sub-interval half-width: 25% of the entire peak distance
                half_win = int(dist * 0.25)

                # Limit interval to avoid out of bounds
                win_start = max(start, mid - half_win)
                win_end   = min(end,   mid + half_win)

                # 3. Find minimum in the sub-interval
                segment = projection_norm[win_start : win_end + 1]
                if len(segment) > 0:
                    offset = np.argmin(segment)
                    valley_idx = win_start + offset
                else:
                    # If sub-interval length is 0, fallback to full interval search
                    valley_idx = start + np.argmin(projection_norm[start : end + 1])

                valley_val = projection_norm[valley_idx]
                # print(f"valley index {valley_idx}, min value {valley_val}")

                valleys.append((valley_idx, valley_val))
            valley_index_list.append(valleys)
            # print("Valley positions and values:", valleys)

        except Exception as e:
            print(f"Error processing lead adhesion in region: {e}")
            continue  # Skip current region, continue to next
        
    flattened_list = [num for sublist in valley_index_list for point in sublist for num in point]
    a = np.array(flattened_list)
    print(a.shape)    

    col_1 = [int(a[0]), int(a[2]), int(a[4]), int(a[6]), int(a[8])]
    col_2 = [int(a[10]), int(a[12]), int(a[14]), int(a[16]), int(a[18])]

    regions = [
    (105 , 430,1084,430+col_1[0]),
    (105 , 430+col_1[0],1084,430+col_1[1]),
    (105 , 430+col_1[1],1084,430+col_1[2]),
    (105 , 430+col_1[2],1084,430+col_1[3]),
    (105 , 430+col_1[3],1084,430+col_1[4]),
    (105 , 430+col_1[4],1084,1575),

    (1092 , 430,2074,430+col_2[0]),
    (1092 , 430+col_2[0],2074,430+col_2[1]),
    (1092 , 430+col_2[1],2074,430+col_2[2]),
    (1092 , 430+col_2[2],2074,430+col_2[3]),
    (1092 , 430+col_2[3],2074,430+col_2[4]),
    (1092 , 430+col_2[4],2074,1575)]



    # Record adhesion borders
    adhesion_regions = list() # Record borders where adhesion occurs
    for group_idx, group in enumerate(valley_index_list):
        print(f"Group {group_idx + 1} valleys:")
        for valley_idx, (idx, value) in enumerate(group):
            print(f"  → Group {group_idx}, Valley {valley_idx}: Pos = {idx}, Value = {value}")
            # group_idx is the group with adhesion, valley is the adhesion border
            if value>0:
                adhesion_regions.append(regions[6*group_idx+valley_idx])
                adhesion_regions.append(regions[6*group_idx+valley_idx+1])

    adhesion_regions = list(dict.fromkeys(adhesion_regions)) # Remove duplicates
    print(adhesion_regions)

    # =================================== Determine if adhesion occurred ==============================================
    digitized_leads = []
    for (x1, y1, x2, y2) in regions:
        roi = final_binary[y1:y2, x1:x2]
        h, w = roi.shape

        # Digitize directly first; if adhesion occurs, re-digitize based on adhesion parts
        ecg_mv = Pixel_to_signal(roi, mv_per_pixel=1 / (10 * px_per_mm))

        # 1. Determine if adhesion occurred
        if (x1, y1, x2, y2) in adhesion_regions:
        # 2. If it is a lead adhesion segment, perform QRS detection
            roi_ad = final_binary[y1:y2, x1:x2]
            h_ad, w_ad = roi_ad.shape

            # Locate R peaks directly using pixel values
            pre_peaks, inv_proj = detect_peaks_via_projection(
            roi_ad,
            height=None,                # Automatic threshold: mean(inv_proj)
            distance=int(0.1 * w_ad)    # Set to 20% of ROI width pixels (Note: Logic says 0.1 which is 10%, comment said 20%. Kept 0.1 logic)
            )

            peaks = refine_peaks(ecg_mv, pre_peaks, search_radius=30)
            S_points = detect_S_points(ecg_mv, peaks, fs=w_ad/5) 
            t = np.arange(len(ecg_mv)) / (w_ad/5)
            arr_range = 3 # Range around baseline
            center = h / 2 # Middle baseline
            background_val = 255 # If background is black, change to 0
            masked = roi_ad.copy()

            masked_out, coords = mask_baselines_v4(roi_ad, ecg_mv, peaks, S_points, masked=roi_ad.copy())
            ecg_mv_fixed = Pixel_to_signal(masked_out, mv_per_pixel=1 / (10 * px_per_mm))
            ecg_restored = restore_rs_waves_v2(ecg_mv, coords)
            digitized_leads.append(ecg_restored)
        else:
            digitized_leads.append(ecg_mv)


    # Save to CSV file
    save_path_6x2(digitized_leads, save_path)

if __name__ == '__main__':
    # Input part is image path
    # 1. Traverse all image data
    lead_name_regions = [
    (105, 490,130, 520),
    (1086, 490,1130, 520),

    (105, 678,130, 710),
    (1086, 678,1130, 710),

    (105, 866,130, 898),
    (1086, 866,1130, 898),

    (105, 1063,150, 1097),
    (1086, 1063,1130, 1097),

    (105, 1253,150, 1283),
    (1086, 1253,1130, 1283),

    (105, 1443,150, 1473),
    (1086, 1443,1130, 1473)]
    folder_path = r"" # Path to store png data
    images = scan_image_files(folder_path)

    
    processing_info = [] # Create info table to record processing status

    count = 0
    for path, name in images:

        # path: image path
        # name: image filename
        # print(f"Filename: {name} \nFull path: {path}\n")
        #
        unique_id = count + 1 # Generate unique ID (can use sequence or timestamp)
        is_completed = 0 # Initialize status

        # 1. Remove background, convert image to binary grayscale
        final_binary = remove_background(path) # ndarray
        save_path = "6x2/output_csv/"+name+".csv"

        # 2. Digitization
        try : 
            dig_ecg_picture(final_binary, save_path)
            
            is_completed = 1
        except Exception as e:
            print(f"Error processing {name}! ")
            is_completed = 0
            # continue  # Skip current image, continue to next
            # Record file info to table
        file_info = {
            'unique_id': unique_id,
            'filename': name,
            'file_path': path,
            'save_path': save_path,
            'is_completed': is_completed
        }
        processing_info.append(file_info)

        count = count + 1

    
    df_info = pd.DataFrame(processing_info) # After processing, create DataFrame and save

    # Save info table as CSV file
    info_table_path = "spilt_layout/6x2/processing_info_table.csv"

    
    df_info.to_csv(info_table_path, index=False, encoding='utf-8-sig')

    # Display processing results statistics
    total_files = len(processing_info)
    successful_files = df_info['is_completed'].sum()
    failed_files = total_files - successful_files

    print(f"\n{'='*50}")
    print(f"Processing completion statistics:")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {successful_files}")
    print(f"Processing failed: {failed_files}")
    print(f"Success rate: {successful_files/total_files*100:.1f}%")
    print(f"Info table saved to: {info_table_path}")  
    print(f"{count} samples processed!")