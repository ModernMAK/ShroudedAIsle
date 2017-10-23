from WindowUtil import get_window_named, scan_window
from time import sleep
# print("Please Open \"The Shrouded Isle\"")
# game = None
# while game is None:
#     print("Waiting...")
#     sleep(5)
#     game = get_window_named("The Shrouded Isle")
# print("\"The Shrouded Isle\" was opened")
#
# scan = scan_window(game)
# scan.save("Scan.png", "PNG")
import ScanManip

# ScanManip.idea()


import win32api, win32gui
from win32con import MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP


class GameInput():
    def __init__(self, game_hwnd):
        self.game_window = game_hwnd

    def get_window_and_focus(self):
        win32gui.SetForegroundWindow(self.game_window)
        bbox = win32gui.GetWindowRect(self.game_window)
        rect = [0,0,0,0]
        rect[0] = bbox[0]
        rect[1] = bbox[1]
        rect[2] = bbox[2]
        rect[3] = bbox[3]
        return rect

    def click(self, pos):
        pos[0] = int(pos[0])
        pos[1] = int(pos[1])
        win32api.SetCursorPos(pos)
        win32api.mouse_event(MOUSEEVENTF_LEFTDOWN, pos[0], pos[1], 0, 0)
        win32api.mouse_event(MOUSEEVENTF_LEFTUP, pos[0], pos[1], 0, 0)

    def map(self, pos):
        screen_rect = self.get_window_and_focus()
        screen_rect[2] = screen_rect[2] - screen_rect[0]  # Right to Width
        screen_rect[3] = screen_rect[3] - screen_rect[1]  # Bottom to Height

        for i in range(2):
            pos[i] = pos[i] * screen_rect[2 + i] + screen_rect[i]
        return pos

    def click_unmapped(self, pos):
        pos = self.map(pos)
        self.click(pos)

    # 0 = Start
    # When has Continue
    # 1 = Continue
    # 2 = Settings
    # 3 = Quit
    # Otherwise
    # 1 = Settings
    # 2 = Quit
    def click_main(self, button, has_continue=False):
        def get_position():
            button_positions = [[0, 0], [0, 0], [0, 0]]
            if has_continue:
                button_positions = [[178, 624], [428, 624], [678, 624], [928, 624]]
            button_size = [176, 54]
            reference_window_size = [1280, 720]

            click_position = [0, 0]
            for i in range(2):
                click_position[i] = (button_positions[button][i] + button_size[i] / 2) / reference_window_size[i]

            return click_position[0], click_position[1]

        self.click_unmapped(get_position())

    # 0 = Start
    # When has Continue
    # 1 = Continue
    # 2 = Settings
    # 3 = Quit
    # Otherwise
    # 1 = Settings
    # 2 = Quit
    def click_main(self, button, has_continue=False):
        def get_position():
            button_positions = [[0, 0], [0, 0], [0, 0]]
            if has_continue:
                button_positions = [[178, 624], [428, 624], [678, 624], [928, 624]]
            button_size = [176, 54]
            reference_window_size = [1280, 720]
            click_position = [0, 0]
            for i in range(2):
                click_position[i] = (button_positions[button][i] + button_size[i] / 2) / reference_window_size[i]
            return click_position

        self.click_unmapped(get_position())

    def click_main_confirmation(self, confirmation=False):
        def get_position():
            button = int(confirmation)
            button_positions = [[460, 412], [688, 412]]
            button_size = [134, 56]
            reference_window_size = [1280, 720]
            click_position = [0, 0]
            for i in range(2):
                click_position[i] = (button_positions[button][i] + button_size[i] / 2) / reference_window_size[i]
            return click_position

        self.click_unmapped(get_position())

    def click_intro_skip(self):
        def get_position():
            button_positions = [1164, 678]
            button_size = [108, 56]
            reference_window_size = [1280, 720]
            click_position = [0, 0]
            for i in range(2):
                click_position[i] = (button_positions[i] + button_size[i] / 2) / reference_window_size[i]
            return click_position

        self.click_unmapped(get_position())

    def click_game_house(self, house):
        def get_position():
            button_positions = [[374, 108], [270, 240], [548, 308], [694, 402], [762, 150], [880, 316]]
            button_size = [128, 128]
            reference_window_size = [1280, 720]
            click_position = [0, 0]
            for i in range(2):
                click_position[i] = (button_positions[house][i] + button_size[i] / 2) / reference_window_size[i]
            return click_position

        self.click_unmapped(get_position())

    def click_game_house_exit(self):
        def get_position():
            button_positions = [1170, 26]
            button_size = [64, 64]
            reference_window_size = [1280, 720]
            click_position = [0, 0]
            for i in range(2):
                click_position[i] = (button_positions[i] + button_size[i] / 2) / reference_window_size[i]
            return click_position

        self.click_unmapped(get_position())


import WindowUtil


def collect_gender_data_from_game():
    from os import walk
    from os import path
    print("Please Open \"The Shrouded Isle\"")
    game = None
    while game is None:
        print("Waiting...")
        sleep(5)
        game = get_window_named("The Shrouded Isle")
    print("\"The Shrouded Isle\" was opened")
    print("Waiting for Loading")
    # sleep(30)

    def cull_names_from_portraits(outdir):
        for directory, dirnames, filenames in walk(outdir):
            for file in filenames:
                ext = path.splitext(file)[1]
                allowed_ext = [".png", ".PNG", ".Png"]
                if ext in allowed_ext:
                    from PIL import Image
                    full_path = path.join(directory, file)
                    img = Image.open(full_path)
                    img = img.crop((0, 0, 128, 128))
                    img.save(full_path)

    def collect_portraits_and_names(outdir):
        def get_position(index):
            portrait_positions = [
                [240, 137], [436, 137],
                [134, 374], [269, 374], [406, 374], [538, 374]
            ]
            portrait_size = [128, 176]
            position = [0, 0, 0, 0]
            position[0] = portrait_positions[index][0]
            position[1] = portrait_positions[index][1]
            position[2] = portrait_positions[index][0] + portrait_size[0]
            position[3] = portrait_positions[index][1] + portrait_size[1]

        names = ["Father", "Mother", "Child1", "Child2", "Child3", "Child4"]
        for member in range(6):
            img = WindowUtil.scan_window_relative_partition(game, get_position(member))
            from os import path
            fp_base = path.join(outdir, names)
            fp = fp_base
            counter = 1
            while path.exists(fp):
                fp = path.join(fp, str.format(" ({0})", counter))
                counter += 1

            fp = path.join(fp, ".png")
            img.save(fp)

    g_in = GameInput(game)
    # On Main, click within [X,Y,W,H]={178,624,176,54}
    # On Intro, double click within [X,Y,W,H]={1164,678,108,56}
    print("Navigating Main")
    g_in.click_main(0, True)
    sleep(.1)
    print("Navigating Main Confirmation")
    g_in.click_main_confirmation(True)
    sleep(.1)
    print("Skipping Intro")
    g_in.click_intro_skip()
    sleep(.1)
    g_in.click_intro_skip()
    sleep(.1)

    print("Fetching Images")
    for i in range(6):
        if i == 0:
            continue
        g_in.click_game_house(i)
        sleep(.1)
        name = ["Kegnni", "Iosefka", "Cadwell", "Efferson", "Blackborn"]
        print("Fetching " + name[i-1])
        collect_portraits_and_names(path.join("Training/Unknown", name[i - 1]))
        g_in.click_game_house_exit()
        sleep(.1)
    input("Waiting for Renaming To finish")
    cull_names_from_portraits("Training/Unknown")
