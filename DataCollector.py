import Scanner
from os import path
from os import walk, makedirs
from PIL import Image
from InputController import InputController, calculate_rect
from time import sleep
import random

from WindowUtil import get_window_named


def get_game():
    print("Please Open \"The Shrouded Isle\"")
    game = get_window_named("The Shrouded Isle")
    unopened_on_start = False
    while game is None:
        unopened_on_start = True
        print("Waiting...")
        sleep(1)
        game = get_window_named("The Shrouded Isle")
    print("\"The Shrouded Isle\" was opened")
    if unopened_on_start:
        print("Waiting for Loading...")
        sleep(30)
    return game


class DataCollector:
    def __init__(self, outdir, game=None, debounce=0.75):
        if game is None:
            game = get_game()
        print("!",game)
        self.game = game
        self.output_dir = outdir
        self.input = InputController(self.game, debounce=debounce)

    def get_portrait_dir(self, is_raw=True):
        root = path.join(self.output_dir, "Gender")
        if is_raw:
            return path.join(root, "Raw")
        else:
            return path.join(root, "Processed")

    def get_screen_dir(self, is_raw=True):
        root = path.join(self.output_dir, "Screen")
        if is_raw:
            return path.join(root, "Raw")
        else:
            return path.join(root, "Processed")

    def select_random_color(self):
        print("Selecting Color Scheme")
        self.input.open_settings(True, True)
        self.input.click_settings_color()
        color = random.randrange(5)
        self.input.click_settings_color_select(color)
        self.input.close_settings()

    def save(self, directory, file, img):
        fp = path.join(directory, file)
        fp += ".png"
        enforce_directory(fp)
        img.save(fp)

    def collect_portraits(self, count=1, shuffle_colors=False):
        for iteration in range(count):
            all_house_names = ["Kegnni", "Iosefka", "Cadwell", "Efferson", "Blackborn"]

            def collect_portraits_from_house(house):
                def get_portrait_rect(member):
                    portrait_positions = [
                        [240, 137], [436, 137],
                        [134, 374], [269, 374], [406, 374], [538, 374]
                    ]
                    portrait_size = [128, 176]
                    reference_window_size = [1282, 747]
                    position = calculate_rect(portrait_positions[member], portrait_size, reference_window_size)
                    return position

                all_portrait_names = ["Father", "Mother", "Child1", "Child2", "Child3", "Child4"]
                for member in range(6):
                    bbox = get_portrait_rect(member)
                    img = Scanner.scan_window_partition_mapped(self.game, bbox)
                    fname = all_house_names[house] + " " + all_portrait_names[member]
                    self.save(self.get_portrait_dir(), fname, img)

            if shuffle_colors:
                self.select_random_color()
            print("Starting New Game")
            self.input.start_new_game(True)
            sleep(3)  # Wait for cut-scene
            print("Skipping Intro")
            self.input.skip_cutscene()
            sleep(4)  # Wait for game to start
            print("Fetching Portraits")
            for i in range(5):
                self.input.click_game_house(i)
                house_name = all_house_names[i]
                print("Fetching House %s" % house_name)
                self.input.move_off()
                collect_portraits_from_house(i)
                self.input.click_game_house_exit()
            print("Returning To Title Screen")
            self.input.return_to_title_from_game()
            sleep(5)  # Wait to return to main menu

    def fix_portraits(self, shape):
        indir = self.get_portrait_dir(True)
        outdir = self.get_portrait_dir(False)
        for directory, _, filenames in walk(indir):
            for file in filenames:
                allowed_ext = [".png", ".PNG", ".Png"]
                if path.splitext(file)[1] in allowed_ext:
                    in_path = path.join(directory, file)
                    out_path = str.replace(in_path, indir, outdir)
                    img = Image.open(in_path)
                    img = img.crop(shape)
                    enforce_directory(out_path)
                    img.save(out_path)

    def collect_screens(self, count=1, shuffle_colors=False):
        def quick_scan(file_name):
            self.input.move_off()
            img = Scanner.scan_window(self.game)
            self.save(self.get_screen_dir(), file_name, img)

        for iterations in range(count):
            if shuffle_colors:
                self.select_random_color()

            print("Starting New Game")
            self.input.start_new_game(True)
            sleep(3)
            print("Skipping Intro")
            self.input.skip_cutscene()
            sleep(4)
            print("Fetching House Screens")
            for i in range(6):
                self.input.click_game_house(i - 1)
                if i == 0:
                    print("Cathedral Decision")
                    # Cathedral seems like the most unpredictable aspect of this bot
                    self.input.click_game_cathedral_selection(0)
                    self.input.click_game_cathedral_selection_confirmation()
                else:
                    print("Appoint House Advisor")
                    member = random.randrange(6)
                    self.input.click_game_house_option(1)  # Appoint Mode
                    self.input.click_game_house_member(member)  # Click
                    self.input.click_game_house_exit()  # Leave
            print("Starting Season")
            self.input.click_game_start_season()
            for i in range(3):
                advisor = random.randrange(5)
                print("Selecting Advisor")
                self.input.click_court_advisor(advisor)
                print("Begin Month")
                self.input.click_court_begin_month()
                print("Account for Potential Popup")
                self.input.click_game_cathedral_selection(0)
                self.input.click_game_cathedral_selection_confirmation()
                print("Skip Vice/Virtue Results")
                self.input.click_court_next_result()
                self.input.click_court_next_result()
                print("Account for Potential Popup")
                self.input.click_game_cathedral_selection(0)
                self.input.click_game_cathedral_selection_confirmation()
                print("Skip Relation Results")
                self.input.click_court_relations_result()
                self.input.click_court_relations_result()

            print("Selecting Sacrifice")
            sacrifice = random.randrange(5)
            self.input.click_sacrifice_advisor(sacrifice)
            print("Confirming Sacrifice")
            self.input.click_sacrifice_confirmation(0)

            sleep(5)
            print("Skipping Cutscene")
            self.input.skip_cutscene()

            sleep(5)
            print("Checking Sacrifice Results")
            self.input.click_sacrifice_results()

            print("Returning To Title")
            self.input.return_to_title_from_game()
            sleep(5)


def enforce_directory(fp):
    directory = path.dirname(fp)
    if not path.exists(directory):
        makedirs(directory)


def fetch_non_conflict_path(fp):
    fp, ext = path.splitext(fp)
    ncfp = fp + ext
    counter = 1
    while path.exists(ncfp):
        ncfp = fp + str.format(" ({0})", counter) + ext
        counter += 1
    return ncfp


# print("Please Open \"The Shrouded Isle\"")
# game = None
# while game is None:
#     print("Waiting...")
#     sleep(1)
#     game = get_window_named("The Shrouded Isle")
# print("\"The Shrouded Isle\" was opened")
# print("Waiting for Loading...")
# # sleep(30)


def parse_command(command):
    i = None
    w = None
    try:
        i = int(command)
    except Exception:
        w = str(command)
    return i, w


def parse_bool(command):
    i, w = parse_command(command)
    if i == 0 or w in ["F", "f", "N", "n", "No", "no", "NO", "False", "false"]:
        return False
    elif i == 1 or w in ["T", "t", "Y", "y", "Yes", "yes", "YES", "True", "true"]:
        return True
    return None


def print_menu_proto(title, prompts=None):
    print(" <-< Shrouded AIsle >-> ")
    print(" <-< %s >-> " % title)
    try:
        for msg in prompts:
            print(msg)
    except TypeError:
        pass
    print("------------------------")


def get_debounce():
    try:
        bounce = float(input("How fast are inputs? (Recommended 0.75)\n"))
    except ValueError:
        bounce = 0.75
    return bounce


def get_count():
    try:
        count = int(input("How many sets of portraits to collect?\n"))
    except ValueError:
        count = 1
    return count


def get_shuffle():
    shuffle = parse_bool(input("Shuffle Color Scheme? (Y/N)\n"))
    if shuffle is None:
        shuffle = False
    return shuffle


def collection_menu(dc):
    def print_menu():
        print_menu_proto(
            "Data Collection",
            ["1) Collect Portraits",
             "2) Cull Portraits",
             "3) Collect Screens",
             "4) Fix Screens",
             "R) Return"])

    def collect_portraits_prompt():
        print_menu_proto("Portrait Collection")
        count = get_count()
        shuffle = get_shuffle()
        dc.collect_portraits(count, shuffle)

    def collect_screens_prompt():
        print_menu_proto("Screen Collection")
        count = get_count()
        shuffle = get_shuffle()
        dc.collect_screens(count, shuffle)

    def parse_menu(command):
        i, w = parse_command(command)

        if i == 1 or w in ["cp"]:
            collect_portraits_prompt()
        elif i == 2 or w in ["qp"]:
            dc.fix_portraits([128, 128, 0, 0])
        elif i == 3 or w in ["cs"]:
            collect_screens_prompt()
        elif i == -1 or w in ["R", "r", "Return", "return", "ret"]:
            return True
        return False

    menu_done = False
    while not menu_done:
        print_menu()
        menu_done = parse_menu(input(""))
