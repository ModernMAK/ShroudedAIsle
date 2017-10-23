from time import sleep
import win32api
from win32con import MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, KEYEVENTF_KEYUP
import WindowUtil


#
# def calculate_pos(button, size, reference):
#     pos = [0, 0]
#     for i in range(2):
#         pos[i] = (button[i] + (size[i] / 2)) / reference[i]
#     return pos

def scale_to_reference(rect, reference):
    temp = []
    temp.extend(rect)
    for i in range(len(temp)):
        temp[i] /= reference[i % 2]
    return temp


def calculate_pos(button, size, reference=None):
    rect = calculate_rect(button, size)
    x = int((rect[0] + rect[2]) / 2)
    y = int((rect[1] + rect[3]) / 2)
    pos = [x, y]
    if reference is not None:
        pos = scale_to_reference(pos, reference)
    return pos


# Returns MinMax
def calculate_rect(button, size, reference=None):
    pos = [0, 0, 0, 0]
    for i in range(4):
        pos[i] = button[i % 2]
    for i in range(2):
        pos[2 + i] += size[i]
    if reference is not None:
        pos = scale_to_reference(pos, reference)
    return pos


# We teach the bot patterns, I need to actually write the code to interact with the game
# This is also kinda/hardcoded
# It expects that given the same resolution, it will click on the right spot
# It then expects that given any resolution,
# that right spot will be at the same offset percentage wise, relative to the window
class InputController:
    def __init__(self, game_process, debounce=0.0001):
        self.game_window = game_process
        self.debounce = debounce

    # RAW, lowlevel stuff, is extremely stupid, but extremely simple

    def get_window_rectangle_and_focus(self):
        WindowUtil.make_active_window(self.game_window)
        rect = WindowUtil.get_window_rectangle(self.game_window)
        return rect

    def move_off(self):
        rect = self.get_window_rectangle_and_focus()
        rect = [rect[0], rect[1]]
        rect[0] -= 1
        rect[1] -= 1
        self.move(rect)

    # Moving the mouse doesn't cause debounce, it still brings focus however
    def move(self, pos, should_map_pos=False):
        if should_map_pos:
            rect = self.get_window_rectangle_and_focus()
            pos = WindowUtil.map_partition_to_rect(pos, rect)
        win32api.SetCursorPos(pos)

    def press(self, key):
        code_dict = {
            "esc": 0x1B,
        }
        key_code = code_dict.get(key, None)
        if key_code is None:
            raise IndexError()

        win32api.keybd_event(key_code, 0, 0, 0)
        win32api.keybd_event(key_code, 0, KEYEVENTF_KEYUP, 0)
        sleep(self.debounce)

    def click(self, pos, should_map_pos=False):
        self.move(pos, should_map_pos)
        win32api.mouse_event(MOUSEEVENTF_LEFTDOWN, pos[0], pos[1], 0, 0)
        win32api.mouse_event(MOUSEEVENTF_LEFTUP, pos[0], pos[1], 0, 0)
        sleep(self.debounce)

    # 0 = Start
    # 1 = Continue
    # 2 = Settings
    # 3 = Quit
    def click_main(self, button, has_continue=False):
        button_positions = [[0, 0], [0, 0], [0, 0]]
        if has_continue:
            button_positions = [[178, 624], [428, 624], [678, 624], [928, 624]]
        elif button > 0:  # 2 and 3 get mapped to 1 and 2 if we don't have continue
            button -= 1
        button_size = [176, 54]
        reference_window_size = [1282, 747]

        click_position = calculate_pos(button_positions[button], button_size, reference_window_size)

        self.click(click_position, True)

    def click_main_start_confirmation(self, confirmation=False):
        button = int(confirmation)
        button_positions = [[688, 412], [460, 412]]
        button_size = [134, 56]
        reference_window_size = [1282, 747]
        click_position = calculate_pos(button_positions[button], button_size, reference_window_size)

        self.click(click_position, True)

    def click_intro_skip(self):
        button_positions = [1164, 678]
        button_size = [108, 56]
        reference_window_size = [1282, 747]
        click_position = calculate_pos(button_positions, button_size, reference_window_size)

        self.click(click_position, True)

    # -1 = Cathedral
    # 0 = Keggni
    # 1 = Iosefka
    # 2 = Cadwell? (From left to right... I forgot the rest after Iosefka)
    def click_game_house(self, house=-1):
        house += 1
        button_positions = [[374, 108], [270, 240], [548, 308], [694, 402], [762, 150], [880, 316]]
        button_size = [128, 128]
        reference_window_size = [1282, 747]
        click_position = calculate_pos(button_positions[house], button_size, reference_window_size)

        self.click(click_position, True)

    def click_game_house_exit(self):
        button_positions = [1170, 26]
        button_size = [64, 64]
        reference_window_size = [1282, 747]
        click_position = calculate_pos(button_positions, button_size, reference_window_size)

        self.click(click_position, True)

    def press_game_menu_open(self):
        self.press("esc")  # I know, ONE LINE? Well I never have to remember to press escape

    # 0 = Resume, 1 = Settings, 2 = Title, 3 = Quit
    def click_game_menu(self, button):
        button_positions = [[540, 276], [540, 332], [540, 390], [540, 446]]
        button_size = [200, 50]
        reference_window_size = [1282, 747]
        click_position = calculate_pos(button_positions[button], button_size, reference_window_size)

        self.click(click_position, True)

    def click_game_menu_rtt_confirmation(self, confirmation=False):
        button = int(confirmation)
        button_positions = [[688, 412], [460, 412]]
        button_size = [134, 56]
        reference_window_size = [1282, 747]
        click_position = calculate_pos(button_positions[button], button_size, reference_window_size)

        self.click(click_position, True)

    # 0 = Credits, 1 = Return
    def click_settings_button(self, button):
        button_positions = [[536, 494], [536, 560]]
        button_size = [210, 54]
        reference_window_size = [1282, 747]
        click_position = calculate_pos(button_positions[button], button_size, reference_window_size)

        self.click(click_position, True)

    # Toggles opening the settings panel
    def click_settings_color(self):
        #When -1, does the same thing
        self.click_settings_color_select()

    # Color is an int, -1 toggles the select panel
    # This assumes the color sellector is not scrollable, like resolution
    # Expected max color is 4
    def click_settings_color_select(self, color=-1):
        button_positions = [670, 448]
        button_size = [188, 34]
        reference_window_size = [1282, 747]
        color += 1

        button_positions[1] += button_size[1] * color

        click_position = calculate_pos(button_positions, button_size, reference_window_size)

        self.click(click_position, True)

    # Refined, Highlevel stuff that performs basic actions
    def start_new_game(self, has_continue=False):
        self.click_main(0, has_continue)
        if has_continue:
            self.click_main_start_confirmation(True)

    # Refined, Highlevel stuff that performs basic actions
    def skip_intro_cutscene(self):
        self.click_intro_skip()
        self.click_intro_skip()

    def return_to_title_from_game(self, menu_open=False):
        if not menu_open:
            self.press_game_menu_open()
        self.click_game_menu(2)
        self.click_game_menu_rtt_confirmation(True)

    def open_settings(self, on_main_menu=True, has_continue=False):
        if on_main_menu:
            self.click_main(2, has_continue)
        else:
            self.press_game_menu_open()
            self.click_game_menu(1)

    def close_settings(self):
        self.click_settings_button(1)
