def calculate_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0

Point = tuple[int, int]  # (x, y)

def side_of_line(point: Point, line_start: Point, line_end: Point) -> float:
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

def classify_side(value: float, epsilon: float = 1.0) ->int:
    if value > epsilon:
        return 1  # Point is on one side of the line
    elif value < -epsilon:
        return -1  # Point is on the other side of the line
    return 0  # Point is on the line (within epsilon)

def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    side_c = side_of_line(c, a, b)
    side_d = side_of_line(d, a, b)
    side_a = side_of_line(a, c, d)
    side_b = side_of_line(b, c, d)

    return side_c * side_d <= 0 and side_a * side_b <= 0

def has_crossed_line(previous_point: Point, current_point: Point, line_start: Point, line_end: Point, epsilon: float = 1.0) -> bool:
    previous_side = classify_side(side_of_line(previous_point, line_start, line_end), epsilon)
    current_side = classify_side(side_of_line(current_point, line_start, line_end), epsilon)

    if previous_side == 0 or current_side == 0:
        return False

    if previous_side == current_side:
        return False

    return segments_intersect(previous_point, current_point, line_start, line_end)

