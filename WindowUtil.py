# Lifted from
# https://stackoverflow.com/questions/10266281/obtain-active-window-using-python


def get_active_window_name():
    """
    Get the currently active window.

    Returns
    -------
    string :
        Name of the currently active window.
    """
    import sys
    active_window_name = None
    if sys.platform in ['linux', 'linux2']:
        # Alternatives: http://unix.stackexchange.com/q/38867/4784
        try:
            import wnck
        except ImportError:
            # logging.info("wnck not installed")
            wnck = None
        if wnck is not None:
            screen = wnck.screen_get_default()
            screen.force_update()
            window = screen.get_active_window()
            if window is not None:
                pid = window.get_pid()
                with open("/proc/{pid}/cmdline".format(pid=pid)) as f:
                    active_window_name = f.read()
        else:
            try:
                from gi.repository import Gtk, Wnck
                gi = "Installed"
            except ImportError:
                # logging.info("gi.repository not installed")
                gi = None
            if gi is not None:
                Gtk.init([])  # necessary if not using a Gtk.main() loop
                screen = Wnck.Screen.get_default()
                screen.force_update()  # recommended per Wnck documentation
                active_window = screen.get_active_window()
                pid = active_window.get_pid()
                with open("/proc/{pid}/cmdline".format(pid=pid)) as f:
                    active_window_name = f.read()
    elif sys.platform in ['Windows', 'win32', 'cygwin']:
        # http://stackoverflow.com/a/608814/562769
        import win32gui
        window = win32gui.GetForegroundWindow()
        active_window_name = win32gui.GetWindowText(window)
    elif sys.platform in ['Mac', 'darwin', 'os2', 'os2emx']:
        # http://stackoverflow.com/a/373310/562769
        from AppKit import NSWorkspace
        active_window_name = (NSWorkspace.sharedWorkspace()
                              .activeApplication()['NSApplicationName'])
    else:
        print("sys.platform={platform} is unknown. Please report."
              .format(platform=sys.platform))
        print(sys.version)
    return active_window_name

# Mofified from Lifted code


def get_active_window():
    def internal_get_active_window():
        """
            Get the currently active window.

            Returns
            -------
            string :
                Name of the currently active window.
            """
        import sys
        active_window_name = None
        if sys.platform in ['linux', 'linux2']:
            # Alternatives: http://unix.stackexchange.com/q/38867/4784
            try:
                import wnck
            except ImportError:
                # logging.info("wnck not installed")
                wnck = None
            if wnck is not None:
                screen = wnck.screen_get_default()
                screen.force_update()
                return screen.get_active_window()
            else:
                try:
                    from gi.repository import Gtk, Wnck
                    gi = "Installed"
                except ImportError:
                    # logging.info("gi.repository not installed")
                    gi = None
                if gi is not None:
                    Gtk.init([])  # necessary if not using a Gtk.main() loop
                    screen = Wnck.Screen.get_default()
                    screen.force_update()  # recommended per Wnck documentation
                    return screen.get_active_window()
        elif sys.platform in ['Windows', 'win32', 'cygwin']:
            # http://stackoverflow.com/a/608814/562769
            import win32gui
            return win32gui.GetForegroundWindow()
        elif sys.platform in ['Mac', 'darwin', 'os2', 'os2emx']:
            # http://stackoverflow.com/a/373310/562769
            from AppKit import NSWorkspace
            return (NSWorkspace.sharedWorkspace()
                    .activeApplication())
        else:
            print("sys.platform={platform} is unknown. Please report."
                  .format(platform=sys.platform))
            print(sys.version)
        return None

    val = internal_get_active_window()
    if val is None or val == 0:
        return None
    return val

# Own code


def get_window_named(name):
    import win32gui
    try:
        result = win32gui.FindWindow(None, name)
        # Assuming its returning 0 for null pointer,
        # in any case, it returned 0 when something was not found, so... MAKE IT NONE!
        if result is 0:
            return None
        else:
            return result
    except win32gui.error as e:
        print(str(e))
        return None


def scan_window(hwnd):
    import win32gui
    bbox = win32gui.GetWindowRect(hwnd)
    return scan_window_absolute_partition(hwnd, bbox)

def scan_window_absolute_partition(hwnd, partition):
    import time
    import win32gui
    from PIL import ImageGrab
    # Allows us to return to previously active window
    active = get_active_window()
    win32gui.SetForegroundWindow(hwnd)
    # Pause for the camera!
    time.sleep(0.1)
    img = ImageGrab.grab(partition)
    time.sleep(0.1)
    # Return to previous window
    win32gui.SetForegroundWindow(active)
    return img

def scan_window_relative_partition(hwnd, partition):
    import time
    import win32gui
    from PIL import ImageGrab
    bbox = win32gui.GetWindowRect(hwnd)

    partition[0] += bbox[0]
    partition[1] += bbox[1]
    partition[2] += bbox[0]
    partition[3] += bbox[1]

    return scan_window_absolute_partition(hwnd,partition)

def scan_window_scaled_partition(hwnd, partition):
    import time
    import win32gui
    from PIL import ImageGrab
    bbox = win32gui.GetWindowRect(hwnd)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    partition[0] *= w
    partition[1] *= h
    partition[2] *= w
    partition[3] *= h

    return scan_window_absolute_partition(hwnd, partition)




    # print("Active window: %s" % str(get_active_window()))
