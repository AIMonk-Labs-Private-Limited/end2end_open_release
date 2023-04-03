import argparse
import math
from easydict import EasyDict as edict

try:
    from . import vocabularies

except (ImportError, SystemError) as e:
    import vocabularies

def size_after_convolution(width):
    return int(math.floor(width/4)) -1 
def expand_for_ctc(width):
    return (width//4)*4
ARGS = edict()
class Config:
    IMAGE_HEIGHT = 64
    NUM_HIDDEN = 128
    VAL_BATCH_SIZE = 1
    MODE = 'infer'
    VOCAB_CLASS = "English_Numbers_Symbols_Specials_extended"
    LOAD_MODEL = True
    def __init__(self,args):
        parser = argparse.ArgumentParser()
        parser.add_argument('--image_height',
                        type=int, default=self.IMAGE_HEIGHT)
        parser.add_argument('--num_hidden',
                        type=int, default=self.NUM_HIDDEN)
        
        parser.add_argument('--val_batch_size',
                        type=int, default=self.VAL_BATCH_SIZE)
        parser.add_argument('--mode',
                        type=str, default=self.MODE)
        parser.add_argument('--load_model',type=str, default=self.LOAD_MODEL)
        parser.add_argument('--vocab_class',
                        type=str, default=self.VOCAB_CLASS)
        global ARGS

        temp = parser.parse_args(args).__dict__
        for k,v in temp.items():
            ARGS[k] = v
        vocab_class = ARGS["vocab_class"]
        vocabulary_class = getattr(vocabularies,vocab_class)()
        ARGS.vocabulary_class = vocabulary_class
        ARGS.num_classes = vocabulary_class.num_classes

