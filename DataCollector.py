from threading import Thread

from WindowUtil import get_window_named, get_window_name, map_partition_to_rect, convert_to_width_height
import Scanner
from os import path
from os import walk, makedirs
from PIL import Image
from InputController import InputController, calculate_rect
from time import sleep
import random
import FFMpegWrapper
import atexit


class DataCollector:
    def __init__(self, game, outdir, debounce=0.75):
        self.game = game
        self.output_dir = outdir
        self.input = InputController(self.game, debounce=)

    def collect_portraits(self, iterations=1, shuffle_colors=False):
        all_house_names = ["Kegnni", "Iosefka", "Cadwell", "Efferson", "Blackborn"]
        def collect_portraits_from_house(house):


        if shuffle_colors:
            select_random_color(input_control)
        print("Starting New Game")
        self.input.start_new_game(True)
        sleep(3) # Wait for cut-scene
        print("Skipping Intro")
        self.input.skip_cutscene()
        sleep(4) # Wait for game to start
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
        sleep(5) #Wait to return to main menu

def enforce_directory(fp):
    dir = path.dirname(fp)
    if not path.exists(dir):
        makedirs(dir)


def fetch_non_conflict_path(fp):
    fp, ext = path.splitext(fp)
    ncfp = fp + ext
    counter = 1
    print(ncfp, fp, ext, "START")
    while path.exists(ncfp):
        print(ncfp, fp, ext, "EXISTS")
        ncfp = fp + str.format(" ({0})", counter) + ext
        counter += 1
    print(ncfp, fp, ext, "DONE")
    return ncfp


def collect_portraits(game, debounce, outdir, shuffle_colors):
    input_control = InputController(game, debounce=debounce)
    # On Main, click within [X,Y,W,H]={178,624,176,54}
    # On Intro, double click within [X,Y,W,H]={1164,678,108,56}
    if shuffle_colors:
        select_random_color(input_control)

    print("Starting New Game")
    input_control.start_new_game(True)
    sleep(3)
    print("Skipping Intro")
    input_control.skip_cutscene()
    sleep(4)
    print("Fetching Images")
    for i in range(5):
        input_control.click_game_house(i)
        name = ["Kegnni", "Iosefka", "Cadwell", "Efferson", "Blackborn"]
        print("Fetching " + name[i])
        input_control.move_off()
        collect_portraits_and_names(game, outdir, name[i])
        input_control.click_game_house_exit()
        sleep(0.5)
    print("Returning To Title")
    input_control.return_to_title_from_game()
    sleep(5)
    # input("Waiting for Renaming To finish")
    # cull_names_from_portraits(game, outdir)


def collect_portraits_and_names(game, outdir, house_specifier):
    def get_portrait_rect(index):
        portrait_positions = [
            [240, 137], [436, 137],
            [134, 374], [269, 374], [406, 374], [538, 374]
        ]
        portrait_size = [128, 176]
        reference_window_size = [1282, 747]
        position = calculate_rect(portrait_positions[index], portrait_size, reference_window_size)
        return position

    names = ["Father", "Mother", "Child1", "Child2", "Child3", "Child4"]
    for member in range(6):
        bbox = get_portrait_rect(member)
        img = Scanner.scan_window_partition_mapped(game, bbox)
        fp = path.join(outdir, house_specifier + " " + names[member])
        fp += ".png"
        fp = fetch_non_conflict_path(fp)
        enforce_directory(fp)

        img.save(fp)


def cull_names_from_portraits(indir, outdir, shape):
    for directory, dirnames, filenames in walk(indir):
        for file in filenames:
            allowed_ext = [".png", ".PNG", ".Png"]
            if path.splitext(file)[1] in allowed_ext:
                in_path = path.join(directory, file)
                out_path = str.replace(in_path, indir, outdir)
                img = Image.open(in_path)
                img = img.crop(shape)
                enforce_directory(out_path)
                img.save(out_path)


def collect_gender_data_from_game(count=1, debounce=1.0, shuffle_colors=False):
    print("Please Open \"The Shrouded Isle\"")
    game = None
    while game is None:
        print("Waiting...")
        sleep(1)
        game = get_window_named("The Shrouded Isle")
    print("\"The Shrouded Isle\" was opened")
    print("Waiting for Loading...")
    # sleep(30)
    for i in range(count):
        collect_portraits(game, debounce, "Data/Gender/Raw", shuffle_colors)


def collect_screens_from_game(count=1, debounce=1.0, shuffle_colors=False):
    print("Please Open \"The Shrouded Isle\"")
    game = None
    while game is None:
        print("Waiting...")
        sleep(1)
        game = get_window_named("The Shrouded Isle")
    print("\"The Shrouded Isle\" was opened")
    print("Waiting for Loading...")
    # sleep(30)
    for i in range(count):
        collect_screens(game, debounce, "Data/Screen/Raw", shuffle_colors)


def select_random_color(input_control):
    print("Selecting Color Scheme")
    input_control.open_settings(True, True)
    input_control.click_settings_color()
    color = random.randrange(5)
    input_control.click_settings_color_select(color)
    input_control.close_settings()


def scan_with_ffmpeg(framerate, name, codec, directory):
    ffmpeg = FFMpegWrapper.get_ffmpeg(framerate, name, codec, directory)
    import ffmpy
    def execute():
        try:
            FFMpegWrapper.run_ffmpeg(ffmpeg)
        except ffmpy.FFRuntimeError as ex:
            if ex.exit_code and ex.exit_code != 255:
                raise

    process = Thread(target=execute)
    process.start()
    return ffmpeg, process


1


def collect_screens(game, debounce, outdir, shuffle_colors):
    input_control = InputController(game, debounce=debounce)
    # On Main, click within [X,Y,W,H]={178,624,176,54}
    # On Intro, double click within [X,Y,W,H]={1164,678,108,56}

    fp = fetch_non_conflict_path(path.join(outdir, "Screen.avi"))
    enforce_directory(fp)
    name = get_window_name(game)
    ffmpeg, process = scan_with_ffmpeg(1, name, "mpeg4", fp)
    print(game, "->", name)
    print("Hit")
    if shuffle_colors:
        select_random_color(input_control)

    atexit.register(FFMpegWrapper.close_ffmpeg, ffmpeg)

    print("Starting New Game")
    input_control.start_new_game(True)
    sleep(3)
    print("Skipping Intro")
    input_control.skip_cutscene()
    sleep(4)
    print("Fetching House Screens")
    for i in range(6):
        input_control.click_game_house(i - 1)
        # sleep(0.5)
        if i == 0:
            # Cathedral seems like the most unpredictable aspect of this bot
            input_control.click_game_cathedral_selection(0)
            # sleep(0.5)
            input_control.click_game_cathedral_selection_confirmation()
        else:
            member = random.randrange(6)
            input_control.click_game_house_option(1)  # Appoint Mode
            # sleep(0.5)
            input_control.click_game_house_member(member)  # Click
            # sleep(0.5)
            input_control.click_game_house_exit()  # Leave
        sleep(0.5)

    # input_control.click_game_start_season()
    # sleep(3)
    # for i in range(3):
    #     advisor = random.randrange(5)
    #     print("Advisor Click")
    #     input_control.click_court_advisor(advisor)
    #     sleep(0.5)
    #     print("Begin Month")
    #     input_control.click_court_begin_month()
    #     sleep(5)
    #     print("Next Result")
    #     input_control.click_court_next_result()
    #     sleep(5)
    #     print("Finish Month")
    #     input_control.click_court_relations_result()
    #     sleep(3)
    #
    # sleep(5)
    # advisor = random.randrange(5)
    # input_control.click_sacrifice_advisor(advisor)
    # sleep(0.5)
    # input_control.click_sacrifice_confirmation(0)
    # sleep(0.5)
    # input_control.skip_cutscene()
    # sleep(0.5)
    # input_control.click_sacrifice_results()
    # sleep(0.5)

    print("Returning To Title")
    input_control.return_to_title_from_game()
    sleep(5)

    # input("Waiting for Renaming To finish")
    # cull_names_from_portraits(game, outdir)
    FFMpegWrapper.close_ffmpeg(ffmpeg)
    sleep(1)


def cull_gender_data():
    cull_names_from_portraits("Data/Raw", "Data/Processed", [0, 0, 128, 128])


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


def collection_menu():
    def print_menu():
        print_menu_proto(
            "Data Collection",
            ["1) Collect Portraits",
             "2) Cull Portraits",
             "3) Collect Screens",
             "4) Cull Screens",
             "R) Return"])

    def collection_input():
        try:
            count = int(input("How many sets of portraits to collect?\n"))
        except ValueError:
            count = 1

        try:
            bounce = float(input("How fast are inputs? (Recommended 0.75)\n"))
        except ValueError:
            bounce = 0.75

        shuffle = parse_bool(input("Shuffle Color Scheme? (Y/N)\n"))
        return count, bounce, shuffle

    def collect_portraits_prompt():
        print_menu_proto("Portrait Collection")
        count, bounce, shuffle = collection_input()
        if shuffle is None:
            collect_gender_data_from_game(count, bounce)
        else:
            collect_gender_data_from_game(count, bounce, shuffle)

    def collect_screens_prompt():
        print_menu_proto("Screen Collection")
        count, bounce, shuffle = collection_input()
        if shuffle is None:
            collect_screens_from_game(count, bounce)
        else:
            collect_screens_from_game(count, bounce, shuffle)

    def parse_menu(command):
        i, w = parse_command(command)

        if i == 1 or w in ["cp"]:
            collect_portraits_prompt()
        elif i == 2 or w in ["qp"]:
            cull_gender_data()
        elif i == 3 or w in ["cs"]:
            collect_screens_prompt()
        elif i == -1 or w in ["R", "r", "Return", "return", "ret"]:
            return True
        return False

    menu_done = False
    while not menu_done:
        print_menu()
        menu_done = parse_menu(input(""))
