WALL_ORIENTATION_FOR_TURN = {"wall": "all", "agarose": "tb", "boundary": "tb"}


class DataContainer:
    """
    A base class for handling data encapsulation from a given object.

    Attributes:
        keys (tuple): A tuple of strings representing the attribute names to be extracted from
                      the input object.

    Methods:
        __init__(obj): Initializes a new instance of the DataContainer, extracting attributes
                       from `obj` based on `keys`.
    """

    keys = ()

    def __init__(self, obj):
        """
        Initializes a new instance of the DataContainer, extracting attributes from `obj`
        based on `keys`.

        Parameters:
            obj (object): The source object from which attributes are extracted.
        """
        for k in self.keys:
            if hasattr(obj, k):
                setattr(self, k, getattr(obj, k))


class TrjDataContainer(DataContainer):
    """
    A specialized DataContainer for handling trajectory-related data.

    Inherits from DataContainer and pre-defines `keys` for trajectory-specific attributes.
    """

    keys = (
        "pxPerMmFloor",
        "f",
        "flt",
        "h",
        "w",
        "nan",
        "theta",
        "sp",
        "x",
        "y",
        "velAngles",
    )


class TrainingProxy:
    """
    A proxy class for encapsulating training start information.

    Attributes:
        start (int): The start index or timestamp of the training data.
    """

    def __init__(self, start):
        """
        Initializes a new instance of TrainingProxy with a specified start.

        Parameters:
            start (int): The start index or timestamp of the training data.
        """
        self.start = start


class VaDataContainer(DataContainer):
    """
    A specialized DataContainer for handling video analysis data.

    Inherits from DataContainer and pre-defines `keys` for VideoAnalysis-specific
    attributes. Additionally, initializes training proxies for each training session found in
    the input object.

    Methods:
        __init__(obj): Extends DataContainer's initialization to include setup of training
                       proxies based on `obj.trns`.
    """

    keys = ("ct", "xf", "f", "fn", "ef", "noyc", "nef", "fps")

    def __init__(self, obj):
        """
        Initializes a new instance of VaDataContainer, extracting attributes from `obj` based
        on `keys` and setting up training proxies for each training session found in
        `obj.trns`.

        Parameters:
            obj (object): The source object from which attributes and training sessions are
                          extracted.
        """
        super().__init__(obj)
        self.trns = [TrainingProxy(trn.start) for trn in obj.trns]
