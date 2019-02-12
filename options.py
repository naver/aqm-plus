import os
import argparse
from six import iteritems
from itertools import product
from time import gmtime, strftime


def readCommandLine(argv=None):
    parser = argparse.ArgumentParser(description='Train and Tet the Visual Dialog model')

    #-------------------------------------------------------------------------
    # Data input settings
    parser.add_argument('-dataRoot', default='data',
                        help='Data root folder')
    parser.add_argument('-dataset', default='',
                        help='Dataset name')
    parser.add_argument('-inputImg', default='',
                            help='HDF5 file with image features')
    parser.add_argument('-inputQues', default='',
                            help='HDF5 file with preprocessed questions')
    parser.add_argument('-inputJson', default='',
                            help='JSON file with info and vocab')
    parser.add_argument('-cocoDir', default='',
                            help='Directory for coco images, optional')
    parser.add_argument('-cocoInfo', default='',
                            help='JSON file with coco split information')
    parser.add_argument('-poolsInfo', default='',
                            help='JSON file with pool information')

    #-------------------------------------------------------------------------
    # Logging settings
    parser.add_argument('-verbose', type=int, default=1,
                            help='Level of verbosity (default 1 prints some info)',
                            choices=[1, 2])
    parser.add_argument('-savePath', default='checkpoints/',
                            help='Path to save checkpoints')
    parser.add_argument('-saveName', default='',
                            help='Name of save directory within savePath')
    parser.add_argument('-startFrom', type=str, default='',
                            help='Copy weights from model at this path')
    parser.add_argument('-qstartFrom', type=str, default='',
                            help='Copy weights from qbot model at this path')
    parser.add_argument('-aqmstartFrom', type=str, default='',
                            help='Copy weights from AQM model at this path')
    parser.add_argument('-aqmQStartFrom', type=str, default='',
                            help='AQM questioner weights from model')
    parser.add_argument('-aqmAStartFrom', type=str, default='',
                            help='AQM answerer weights from model')
    parser.add_argument('-continue', action='store_true',
                            help='Continue training from last epoch')
    parser.add_argument('-enableVisdom', type=int, default=0,
                            help='Flag for enabling visdom logging')
    parser.add_argument('-visdomEnv', type=str, default='',
                            help='Name of visdom environment for plotting')
    parser.add_argument('-visdomServer', type=str, default='127.0.0.1',
                            help='Address of visdom server instance')
    parser.add_argument('-visdomServerPort', type=int, default=8893,
                            help='Port of visdom server instance')

    #-------------------------------------------------------------------------
    # Model params for both a-bot and q-bot
    parser.add_argument('-randomSeed', default=1234, type=int,
                            help='Seed for random number generators')
    parser.add_argument('-imgEmbedSize', default=300, type=int,
                            help='Size of the multimodal embedding')
    parser.add_argument('-imgFeatureSize', default=4096, type=int,
                            help='Size of the image feature')
    parser.add_argument('-embedSize', default=300, type=int,
                            help='Size of input word embeddings')
    parser.add_argument('-rnnHiddenSize', default=512, type=int,
                            help='Size of the LSTM state')
    parser.add_argument('-numLayers', default=2, type=int,
                            help='Number of layers in LSTM')
    parser.add_argument('-imgNorm', default=1, type=int,
                            help='Normalize the image feature. 1=yes, 0=no')

    # A-Bot encoder + decoder
    parser.add_argument('-encoder', default='hre-ques-lateim-hist',
                            help='Name of the encoder to use',
                            choices=['hre-ques-lateim-hist'])
    parser.add_argument('-decoder', default='gen',
                            help='Name of the decoder to use (gen)',
                            choices=['gen'])
    # Q-bot encoder + decoder
    parser.add_argument('-qencoder', default='hre-ques-lateim-hist',
                            help='Name of the encoder to use',
                            choices=['hre-ques-lateim-hist'])
    parser.add_argument('-qdecoder', default='gen',
                            help='Name of the decoder to use (only gen supported now)',
                            choices=['gen'])

    #-------------------------------------------------------------------------
    # Optimization / training params
    parser.add_argument('-trainMode', default='sl-abot',
                            help='What should train.py do?',
                            choices=['sl-abot', 'sl-qbot', 'rl-full-QAf', 'aqmbot-ind', 'aqmbot-dep'])
    parser.add_argument('-trainSplit', default=None,
                            help='Which part of dataset should train on',
                            choices=['first', 'last'])
    parser.add_argument('-numRounds', default=10, type=int,
                            help='Number of rounds of dialog (max 10)')
    parser.add_argument('-batchSize', default=20, type=int,
                            help='Batch size (number of threads) '
                                    '(Adjust base on GPU memory)')
    parser.add_argument('-learningRate', default=1e-3, type=float,
                            help='Learning rate')
    parser.add_argument('-minLRate', default=5e-5, type=float,
                            help='Minimum learning rate')
    parser.add_argument('-dropout', default=0.0, type=float, help='Dropout')
    parser.add_argument('-numEpochs', default=65, type=int, help='Epochs')
    parser.add_argument('-lrDecayRate', default=0.9997592083, type=float,
                            help='Decay for learning rate')
    parser.add_argument('-CELossCoeff', default=200, type=float,
                            help='Coefficient for cross entropy loss')
    parser.add_argument('-featLossCoeff', default=1000, type=float,
                            help='Coefficient for feature regression loss')
    parser.add_argument('-useCurriculum', default=1, type=int,
                            help='Use curriculum or for RL training (1) or not (0)')
    parser.add_argument('-freezeQFeatNet', default=0, type=int,
                            help='Freeze weights of Q-bot feature network')
    parser.add_argument('-rlAbotReward', default=1, type=int,
                            help='Choose whether RL reward goes to A-Bot')

    # Other training environmnet settings
    parser.add_argument('-useGPU', action='store_true', help='Use GPU or CPU')
    parser.add_argument('-numWorkers', default=2, type=int,
                            help='Number of worker threads in dataloader')

    #-------------------------------------------------------------------------
    # Evaluation params
    parser.add_argument('-beamSize', default=1, type=int,
                            help='Beam width for beam-search sampling')
    parser.add_argument('-qbeamSize', default=None, type=int,
                            help='Define the beam-search width of AQM')
    parser.add_argument('-evalModeList', default=[], nargs='+',
                            help='What task should the evaluator perform?',
                            choices=['ABotRank', 'QBotRank', 'QABotsRank', 'dialog',
                                     'AQMBotRank', 'AQMdialog'])
    parser.add_argument('-evalSplit', default='val',
                            choices=['train', 'val', 'test'])
    parser.add_argument('-evalTitle', default='eval',
                            help='If generating a plot, include this in the title')
    parser.add_argument('-aqmRealQA', default=0, type=int,
                            help='Whether use real QA in dataset to evaluate AQM')
    parser.add_argument('-saveLogs', default=0, type=int,
                            help='Save logs to file')
    parser.add_argument('-showQA', default=0, type=int,
                            help='Print QA to console while evaluating')
    parser.add_argument('-expLowerLimit', default=None, type=int,
                            help='Evaluate on image [expLowerLimit, expUpperLimit)')
    parser.add_argument('-expUpperLimit', default=None, type=int,
                            help='Evaluate on image [expLowerLimit, expUpperLimit)')
    parser.add_argument('-selectedBatchIdxs', default=None, nargs='+', type=int,
                            help='Evaluate on images of the provided batch idxs')
    parser.add_argument('-runRounds', default=None, type=int,
                            help='Number of rounds of dialog (max 10)')
    parser.add_argument('-lambda', default=6, type=int,
                            help='Lambda used in Prior')
    parser.add_argument('-alpha', default=0, type=float,
                            help='Alpha used in infogain')
    parser.add_argument('-gamma', default=0, type=float,
                            help='Gamma used in beam search')
    parser.add_argument('-delta', default=0, type=float,
                            help='Delta used in beam search')
    parser.add_argument('-onlyGuesser', default=0, type=int,
                            help='0 for guesser, otherwise infogain')

    parser.add_argument('-numImg', default=20, type=int,
                        help='Set # candidate images for infogain')
    parser.add_argument('-numQ', default=20, type=int,
                            help='Set # candidate questions for infogain')
    parser.add_argument('-numA', default=20, type=int,
                        help='Set # beam search answers for infogainImpGen')

    parser.add_argument('-randQ', default=0, type=int,
                        help='randomly choose question candidates')
    parser.add_argument('-randA', default=0, type=int,
                       help='randomly choose answer candidates')
    parser.add_argument('-resampleEveryDialog', default=0, type=int,
                        help='sample new fix for every dialog')

    parser.add_argument('-gen1Q', default=0, type=int,
                        help='generate questions via SL and then fix the set')
    parser.add_argument('-gtQ', default=0, type=int,
                        help='use questions in dataset')

    parser.add_argument('-noHistory', default=0, type=int,
                        help='do not consider history when calculating p_a')

    parser.add_argument('-slGuesser', default=0, type=int,
                        help='use SL guesser')

    parser.add_argument('-zeroCaption', default=0, type=int,
                        help='Use zero paddings instead of captions')
    parser.add_argument('-randomCaption', default=0, type=int,
                        help='Use random captions instead of true captions')
    #-------------------------------------------------------------------------

    try:
        parsed = vars(parser.parse_args(args=argv))
    except IOError as msg:
        parser.error(str(msg))

    # get path
    try:
        import nsml
        from nsml import HAS_DATASET, IS_ON_NSML, DATASET_PATH

        if not HAS_DATASET and not IS_ON_NSML:  # local
            prePath = os.path.join(parsed['dataRoot'], parsed['dataset'])
        else:
            prePath = os.path.join(DATASET_PATH, 'train')
    except:  # local
        prePath = os.path.join(parsed['dataRoot'], parsed['dataset'])

    # default path
    if prePath == os.path.join(parsed['dataRoot'], ''):
        if not os.path.exists(prePath):
            prePath = 'resources/'

    # cocoInfo
    if not parsed['cocoInfo'] and 'train' not in parser.prog:
        parsed['cocoDir'] = '.'
        parsed['cocoInfo'] = 'val.json'

    # set default path
    parsed['inputImg'] = parsed['inputImg'] or os.path.join(prePath, 'data/visdial/data_img.h5')
    parsed['inputJson'] = parsed['inputJson'] or os.path.join(prePath, 'data/visdial/chat_processed_params.json')

    if not parsed['inputQues']:
        if 'train' not in parser.prog:  # use genCap for eval
            parsed['inputQues'] = os.path.join(prePath, 'data/visdial/chat_processed_data_gencaps.h5')
        else:  # use gtCap for train
            parsed['inputQues'] = os.path.join(prePath, 'data/visdial/chat_processed_data.h5')

    # make as full path
    for optionName in ['startFrom', 'qstartFrom', 'aqmstartFrom', 'aqmQStartFrom', 'aqmAStartFrom']:
        if parsed[optionName]:
            if not os.path.exists(parsed[optionName]):
                parsed[optionName] = os.path.join(prePath, 'checkpoints', parsed[optionName])

    if parsed['aqmQStartFrom'] or parsed['aqmAStartFrom']:
        assert parsed['aqmQStartFrom'] and parsed['aqmAStartFrom'], "Please speicify Q and A model for AQM!"
    
    if parsed['trainMode'] in ['aqmbot-dep-hlf']:
        assert parsed['trainSplit']

    if parsed['saveName']:
        # Custom save file path
        parsed['savePath'] = os.path.join(parsed['savePath'],
                                          parsed['saveName'])
    else:
        # Standard save path with time stamp
        import random
        timeStamp = strftime('%d-%b-%y-%H-%M', gmtime())
        parsed['savePath'] = os.path.join(parsed['savePath'], timeStamp)
        parsed['savePath'] += '_{:0>6d}'.format(random.randint(0, 10e6))
        print(parsed['savePath'])

    # check if history is needed
    parsed['useHistory'] = True if 'hist' in parsed['encoder'] else False

    # check if image is needed
    if 'lateim' in parsed['encoder']:
        parsed['useIm'] = 'late'
    elif 'im' in parsed['encoder']:
        parsed['useIm'] = True
    else:
        parsed['useIm'] = False
    
    return parsed
