from WindowUtil import get_window_named, get_window_rectangle, map_partition_to_rect, convert_to_width_height
import Scanner
from os import path
from os import walk, makedirs
from PIL import Image
from InputController import InputController, calculate_rect
from time import sleep


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


def collect_portraits(game, outdir):
    input_control = InputController(game, debounce=1)
    # On Main, click within [X,Y,W,H]={178,624,176,54}
    # On Intro, double click within [X,Y,W,H]={1164,678,108,56}
    print("Navigating Main")
    input_control.click_main(0, True)
    print("Navigating Main Confirmation")
    input_control.click_main_start_confirmation(True)
    sleep(3)
    print("Skipping Intro")
    input_control.click_intro_skip()
    input_control.click_intro_skip()

    sleep(4)
    print("Fetching Images")
    for i in range(6):
        if i == 0:
            continue
        input_control.click_game_house(i)
        name = ["Kegnni", "Iosefka", "Cadwell", "Efferson", "Blackborn"]
        print("Fetching " + name[i - 1])
        input_control.move_off()
        collect_portraits_and_names(game, path.join(outdir, name[i - 1]))
        input_control.click_game_house_exit()
        sleep(0.1)
    input_control.press_game_menu_open()
    input_control.click_game_menu(2)
    input_control.click_game_menu_rtt_confirmation(True)
    sleep(5)
    # input("Waiting for Renaming To finish")
    # cull_names_from_portraits(game, outdir)


def collect_portraits_and_names(game, outdir):
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
        fp = path.join(outdir, names[member])
        fp += ".png"
        fp = fetch_non_conflict_path(fp)
        print(fp, " => ", bbox)
        enforce_directory(fp)

        img.save(fp)


def cull_names_from_portraits(outdir):
    for directory, dirnames, filenames in walk(outdir):
        for file in filenames:
            allowed_ext = [".png", ".PNG", ".Png"]
            if path.splitext(file)[1] in allowed_ext:
                full_path = path.join(directory, file)
                img = Image.open(full_path)
                img = img.crop((0, 0, 128, 128))
                img.save(full_path)


def collect_gender_data_from_game(count=1):
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
        collect_portraits(game, "Training/Unknown")
