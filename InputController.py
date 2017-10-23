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
class InputController:
    def __init__(self, game_process, debounce=0.0001):
        self.game_window = game_process
        self.debounce = debounce

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
    # When has Continue
    # 1 = Continue
    # 2 = Settings
    # 3 = Quit
    # Otherwise
    # 1 = Settings
    # 2 = Quit
    def click_main(self, button, has_continue=False):
        button_positions = [[0, 0], [0, 0], [0, 0]]
        if has_continue:
            button_positions = [[178, 624], [428, 624], [678, 624], [928, 624]]
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

    def click_game_house(self, house):
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

    def click_game_menu_rtt_confirmation(self,confirmation=False):
        button = int(confirmation)
        button_positions = [[688, 412], [460, 412]]
        button_size = [134, 56]
        reference_window_size = [1282, 747]
        click_position = calculate_pos(button_positions[button], button_size, reference_window_size)

        self.click(click_position, True)
