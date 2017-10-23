import WindowUtil
import time
from PIL import ImageGrab


# Returns a pillow image
def scan_window(process):
    rect = WindowUtil.get_window_rectangle(process)
    return scan_window_partition(rect)


# Returns a pillow image
def scan(rect):
    time.sleep(0.1)
    img = ImageGrab.grab(rect)
    time.sleep(0.1)
    return img


# Returns a pillow image
# Partition is a min_max rect used instead of the window rect
def scan_window_partition(process, partition):
    active = WindowUtil.make_active_window(process)
    img = scan(partition)
    WindowUtil.make_active_window(active)
    return img


# Partition is a width_height rect offset from window rect's x/y
def scan_window_relative(process, partition):
    rect = WindowUtil.get_window_rectangle(process)
    partition[0] += rect[0]
    partition[1] += rect[1]
    partition = WindowUtil.convert_to_min_max(partition)

    return scan_window_partition(process, partition)


# Returns a pillow image
# Partition is a min_max rect of floats used to find positions on the window rect
def scan_window_partition_mapped(process, partition):
    rect = WindowUtil.get_window_rectangle(process)
    WindowUtil.map_partition_to_rect(partition, rect)
    return scan_window_partition(process, partition)


# Returns a pillow image
# Partition is a width_height rect of floats used to find positions on the window rect
def scan_window_relative_mapped(process, partition):
    rect = WindowUtil.get_window_rectangle(process)
    WindowUtil.map_partition_to_rect(partition, rect)
    return scan_window_relative(process, partition)
