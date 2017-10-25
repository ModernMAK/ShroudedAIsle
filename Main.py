# # This is a fun little idea I cooked up while playing the Shrouded Isle
# # A game which is rather simple to play (Select Court, Use members of the Court, Find a Sinner, Execute, Repeat)
# # but is rather complicated to complete (Manage Virtues, Find Vices, Maintain Relations, Complete Dreams)
# # Because its a point in click game, I feel it is "pretty simple" to hack up some rudimentary AI to play it.
# # That being said, I have no idea how to use Tensorflow, nor do I have access to an API to interact with the game.

from ImageDataset import DisposableImageDataSet, ImageGenderDataSet, ImageDataSet
from os.path import join, basename
from os import getcwd, rename
from GenderBrain import GenderBrain
from time import sleep
from DataCollector import print_menu_proto, parse_command, collection_menu, DataCollector, get_game


def load_dataset_from_directory(directory, is_prediction=False):
    if is_prediction:
        dataset = ImageDataSet()
    else:
        dataset = ImageGenderDataSet()
    dataset.load_from_images(directory, [128, 128], 3)
    return dataset


def load_dataset_from_file(file, mode):
    dataset = ImageGenderDataSet()
    dataset.load_from_file(file, mode=mode)
    return dataset


def run_training(epochs=100, batch_size=32):
    raw_dataset = load_dataset_from_directory("Data/Training")
    training_dataset = DisposableImageDataSet(raw_dataset).repeat(epochs, shuffle_on_repeat=True)

    params = {
        "learning_rate": 0.001
    }
    brain = GenderBrain("graph/gender", params=params)
    # brain.build_graph()
    brain.train(training_dataset, epochs, batch_size, print_every_nth_step=10, save_every_nth_step=10)


def run_prediction(batch_size=32):
    raw_dataset = load_dataset_from_directory("Data\Predict\\Unsorted", True)
    predict_dataset = DisposableImageDataSet(raw_dataset)

    def get_category(pair):
        if pair[0] < pair[1]:
            return 1
        elif pair[0] > pair[1]:
            return 0
        else:
            raise Exception("Equal!?")

    brain = GenderBrain("graph/gender")
    # brain.build_graph()
    predictions = brain.predict(predict_dataset, batch_size)
    for i in range(predict_dataset.size):
        out_dir = "Data\Predict\\Unsorted"
        pred = get_category(predictions[i])
        if pred == 0:
            out_dir = "Data\Predict\Male"
        elif pred == 1:
            out_dir = "Data\Predict\Female"
        else:
            print("Uh... Houstin, there's a problem...")

        file_path = predict_dataset[1][i]
        file_name = basename(file_path)
        rename(file_path, join(join(getcwd(), out_dir), file_name))
        sleep(0.01)


def main():
    game = get_game()
    dc = DataCollector("Data",game=game)

    def print_menu():
        print_menu_proto(
            "Main Menu",
            [
                "1) Collect Data",
                "2) Train Gender Recognition",
                "3) Predict Gender",
                "Q) Quit"
            ])

    def parse_cmd(cmd):
        int_cmd, str_cmd = parse_command(cmd)

        if int_cmd == 1 or str_cmd in ["c", "C", "collect", "Collect", "COLLECT"]:
            collection_menu(dc)
        elif int_cmd == 2 or str_cmd in ["T", "t", "train", "Train", "TRAIN"]:
            run_training()
        elif int_cmd == 3 or str_cmd in ["P", "p", "predict", "Predict", "PREDICT"]:
            run_prediction()
        elif str_cmd in ["q", "x", "X", "Q", "quit", "QUIT", "Quit"]:
            exit(0)
        else:
            print("I don't understand '%s'" % str(cmd))

    while True:
        print_menu()
        code = input()
        parse_cmd(code)


if __name__ == "__main__":
    print("RUNNING")
    main()
    # run()
