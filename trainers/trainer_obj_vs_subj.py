import traceback

from db_manager import Connection
from utils import PCKL_OBJSUBJ_FILENAME_START
from trainers.trainer_general import train_classifier


if __name__ == "__main__":
    conn = Connection()
    collection = conn.get_english_collection()

    try:
        train_classifier(collection=collection, filename_start=PCKL_OBJSUBJ_FILENAME_START,
            main_cat_name="OBJ", main_cat_filenames=[
                "..\\training_data\\neutral.txt", "..\\training_data\\news.txt",
            ],
            opposite_cat_name="SUBJ", opposite_cat_filenames=[
                "..\\training_data\\positive.txt", "..\\training_data\\negative.txt",
            ])
    except:
        traceback.print_exc()
    finally:
        conn.close_connection()
