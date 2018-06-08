import traceback

from db_manager import Connection
from utils import PCKL_POSNEG_FILENAME_START
from trainers.trainer_general import train_classifier


if __name__ == "__main__":
    conn = Connection()
    collection = conn.get_english_collection()

    try:
        train_classifier(collection=collection, filename_start=PCKL_POSNEG_FILENAME_START,
            opposite_cat_name="NEG", opposite_cat_filenames=["..\\training_data\\negative.txt"],
            main_cat_name="POS", main_cat_filenames=["..\\training_data\\positive.txt"],
            )
    except:
        traceback.print_exc()
    finally:
        conn.close_connection()
