# HEAVILY MODIFIED FROM
# https://stackoverflow.com/questions/10266281/obtain-active-window-using-python

from win32gui import GetWindowText, GetForegroundWindow, FindWindow, GetWindowRect, SetForegroundWindow


def get_window_name(window):
    return GetWindowText(window)


# Returns ProcessIndex, Name
def get_active_window_and_name():
    window = GetForegroundWindow()
    window_name = get_window_name(window)
    return window, window_name


# Returns Name
def get_active_window_name():
    return get_active_window_and_name()[1]


# Returns ProcessIndex
def get_active_window():
    return get_active_window_and_name()[0]


# Returns ProcessIndex
def get_window_named(name):
    result = FindWindow(None, name)
    # Assuming its returning 0 for null pointer,
    # in any case, it returned 0 when something was not found, so... MAKE IT NONE!
    if result is 0:
        return None
    else:
        return result


# Converts a min_max rect to a width_height rect
def convert_to_width_height(bounds):
    temp = []
    temp.extend(bounds)
    temp[2] -= temp[0]
    temp[3] -= temp[1]
    return temp


# Converts a width_height rect to a min_max rect
def convert_to_min_max(bounds):
    temp = []
    temp.extend(bounds)
    temp[2] += temp[0]
    temp[3] += temp[1]
    return temp


def get_window_rectangle(process, use_min_max=True):
    # rect is a min_max rect
    rect = GetWindowRect(process)
    # Quickly converts tuple to list
    bounds = []
    bounds.extend(rect)
    # convert to width_height as needed
    if not use_min_max:
        bounds = convert_to_width_height(bounds)
    return bounds


# returns the previous active process for easy restoration
def make_active_window(process):
    active = get_active_window()
    SetForegroundWindow(process)
    return active


def map_partition_to_rect(partition, rect, rect_is_min_max=True):
    if rect_is_min_max:
        rect = convert_to_width_height(rect)
    # Partition should be x,y, or x,y,w,h or l,u,r,d
    # rect should always be l,u,r,d
    for i in range(len(partition)):
        # 2 + i % 2
        # 2 ensure we are looking at W,H
        # i % 2 ensures when we look at X and W, when i % 2 == 0,we inspect rect W,
        # whereas when we look at Y and H, when i % 2 == 1, we inspect rect H
        partition[i] *= rect[2 + i % 2]
        partition[i] += rect[i % 2]
        partition[i] = int(partition[i])
    return partition
