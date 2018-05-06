import pickle


PCKL_DATADIR = "pickled_objects\\{}.pickle"
PCKL_POSNEG_FILENAME_START = "pos_vs_neg_"


def serialize(object, filename):
    """
    Serialize object (can be a classifier, or featureset, whatever)

    :param object
    :param filename: filename of file where classifier should be saved
    """
    with open(filename, "wb") as dumpfile:
        pickle.dump(object, dumpfile)

    print(f'Object is saved to {filename}')


def deserialize(filename):
    """
    Loads serialized object from given filename

    :param filename: name of the *.pickle file where classifier is stored
    :return: a ready-to-use object
    """
    with open(filename, "rb") as sourcefile:
        return pickle.load(sourcefile)

