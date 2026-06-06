"""Project constants that must remain stable across experiments."""

CLASS_NAMES = (
    "dislike",
    "fist",
    "like",
    "palm",
    "peace",
    "peace_inverted",
    "rock",
    "stop",
    "two_up",
    "two_up_inverted",
)

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SUPPORTED_MODELS = ("resnet18", "mobilenetv3_small")
TRAINING_MODES = ("classifier_only", "full_finetuning")
