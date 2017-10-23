from WindowUtil import get_window_named, get_window_rectangle, map_partition_to_rect, convert_to_width_height
import Scanner
from os import path
from os import walk, makedirs
from PIL import Image
from InputController import InputController, calculate_rect
from time import sleep
import random


def enforce_directory(fp):
    dir = path.dirname(fp)
    if not path.exists(dir):
        makedirs(dir)


def fetch_non_conflict_path(fp):
    fp, ext = path.splitext(fp)
    ncfp = fp + ext
    counter = 1
    while path.exists(ncfp):
        ncfp = fp + str.format(" ({0})", counter) + ext
        counter += 1
    return ncfp


def collect_portraits(game, debounce, outdir, shuffle_colors):
    input_control = InputController(game, debounce=debounce)
    # On Main, click within [X,Y,W,H]={178,624,176,54}
    # On Intro, double click within [X,Y,W,H]={1164,678,108,56}
    if shuffle_colors:
        print("Selecting Pallete")
        input_control.open_settings(True, True)
        input_control.click_settings_color()
        color = random.randrange(5)
        input_control.click_settings_color_select(color)
        input_control.close_settings()

    print("Starting New Game")
    input_control.start_new_game(True)
    sleep(3)
    print("Skipping Intro")
    input_control.skip_intro_cutscene()
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
        print(fp, " => ", bbox)
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
        collect_portraits(game, debounce, "Data/Raw", shuffle_colors)


def cull_gender_data():
    cull_names_from_portraits("Data/Raw", "Data/Processed", [0, 0, 128, 128])
