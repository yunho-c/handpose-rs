def normalize_radians(
    angle: float
) -> float:
    """__normalize_radians

    Parameters
    ----------
    angle: float

    Returns
    -------
    normalized_angle: float
    """
    return angle - 2 * math.pi * math.floor((angle + pi) / (2 * pi))

    
    
def calculate_rects(palms, img):
    w, h = img.shape[1], img.shape[0] # of output image
    wh_ratio = 1 # NOTE perhaps shouldn't remain 1...?
    rects = [] # List to store rectangle information for palms.

    # Check if any palms are detected in the input.
    if len(palms) > 0:
        # Loop through each detected palm.
        for palm in palms:
            # Extract details of the palm.
            sqn_rr_size = palm[0]
            rotation = palm[1]
            sqn_rr_center_x = palm[2]
            sqn_rr_center_y = palm[3]

            # Convert relative coordinates to actual pixel values.
            cx = int(sqn_rr_center_x * w)
            cy = int(sqn_rr_center_y * h)
            xmin = int((sqn_rr_center_x - (sqn_rr_size / 2)) * w)
            xmax = int((sqn_rr_center_x + (sqn_rr_size / 2)) * w)
            ymin = int((sqn_rr_center_y - (sqn_rr_size * wh_ratio / 2)) * h)
            ymax = int((sqn_rr_center_y + (sqn_rr_size * wh_ratio / 2)) * h)

            # Ensure coordinates do not exceed image boundaries.
            xmin = max(0, xmin)
            xmax = min(w, xmax)
            ymin = max(0, ymin)
            ymax = min(w, ymax)

            # Calculate rotation degree.
            degree = math.degrees(rotation)
            rects.append([cx, cy, (xmax-xmin), (ymax-ymin), degree])

        # Convert the list of rectangles to a numpy array.
        rects = np.asarray(rects, dtype=np.float32)

    return rects