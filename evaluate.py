import os
import gc
import random
import pprint
from six.moves import range
from markdown2 import markdown
from time import gmtime, strftime
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import options
from dataloader import VisDialDataset
from torch.utils.data import DataLoader
from eval_utils.dialog_generate import dialogDump
from eval_utils.rank_answerer import rankABot
from eval_utils.rank_questioner import rankQBot, rankQABots
from utils import utilities as utils
from utils.visualize import VisdomVisualize
import visdom

import numpy as np

from eval_utils.aqm_runner import AQMRunner

try:
    from nsml import Visdom
    print('able to use nsml Visdom')
except ImportError:
    pass

def getAQMSetting(params):
    if params["aqmstartFrom"]:
        strategy = "depA"
        qBot = os.path.splitext(os.path.basename(params["qstartFrom"]))[0]
        aBot = os.path.splitext(os.path.basename(params["startFrom"]))[0]
        aprxABot = os.path.splitext(os.path.basename(params["aqmstartFrom"]))[0]
    else:
        aBot = os.path.splitext(os.path.basename(params["startFrom"]))[0]
        aprxABot = os.path.splitext(os.path.basename(params["aqmAStartFrom"]))[0]
        strategy = "trueA" if aBot == aprxABot else "indA"
        qBot = os.path.splitext(os.path.basename(params["aqmQStartFrom"]))[0]

    if "delta" in qBot:
        assert "delta" in aBot and "delta" in aprxABot
    else:
        assert "delta" not in aBot and "delta" not in aprxABot

    if "delta" in qBot:
        hpSetting = "delta"
    else:
        hpSetting = "nondelta"

    aqmSetting = {
        "hpSetting": hpSetting,
        "strategy": strategy,
        "qBot": qBot,
        "aBot": aBot,
        "aprxABot": aprxABot,
    }
    return aqmSetting

def main(params):
    aqmSetting = None
    if ("AQMBotRank" in params["evalModeList"]
            or "AQMdialog" in params["evalModeList"]
            or "AQMdemo" in params["evalModeList"]):
        aqmSetting = getAQMSetting(params)

    # setup dataloader
    dlparams = params.copy()
    dlparams['useIm'] = True
    dlparams['useHistory'] = True
    dlparams['numRounds'] = 10
    splits = ['val', 'test']

    dataset = VisDialDataset(dlparams, splits)

    # Transferring dataset parameters
    transfer = ['vocabSize', 'numOptions', 'numRounds']
    for key in transfer:
        if hasattr(dataset, key):
            params[key] = getattr(dataset, key)

    if 'numRounds' not in params:
        params['numRounds'] = 10

    # Always load checkpoint parameters with continue flag
    params['continue'] = True

    excludeParams = ['batchSize', 'visdomEnv', 'startFrom', 'qstartFrom', 'trainMode', \
        'evalModeList', 'inputImg', 'inputQues', 'inputJson', 'evalTitle', 'beamSize', \
        'enableVisdom', 'visdomServer', 'visdomServerPort', 'randomCaption', 'zeroCaption',
                     'numImg', 'numQ', 'numA', 'alpha',
                     'qbeamSize', 'gamma', 'delta', 'lambda',
                     'onlyGuesser', 'randQ', 'gen1Q', 'gtQ', 'randA', 'noHistory',
                     'slGuesser', 'resampleEveryDialog']

    aBot = None
    qBot = None
    aqmBot = None

    # load aBot
    print('load aBot')
    if params['startFrom']:
        aBot, loadedParams, _ = utils.loadModel(params, 'abot', overwrite=True)
        assert aBot.encoder.vocabSize == dataset.vocabSize, "Vocab size mismatch!"
        for key in loadedParams:
            params[key] = loadedParams[key]
        aBot.eval()

    # Retaining certain dataloder parameters
    for key in excludeParams:
        params[key] = dlparams[key]

    print('load qBot')
    # load qBot
    if params['qstartFrom'] and not params['aqmstartFrom']:
        qBot, loadedParams, _ = utils.loadModel(params, 'qbot', overwrite=True)
        assert qBot.encoder.vocabSize == params[
            'vocabSize'], "Vocab size mismatch!"
        for key in loadedParams:
            params[key] = loadedParams[key]
        qBot.eval()

    # Retaining certain dataloder parameters
    for key in excludeParams:
        params[key] = dlparams[key]

    print('load AQM-Bot')
    # load aqmBot
    if params['aqmstartFrom']:  # abot of AQM
        assert params['qstartFrom']  # qbot of AQM

        aqmBot, loadedParams, _ = utils.loadModel(params, 'AQM-qbot', overwrite=True)
        assert aqmBot.questioner.encoder.vocabSize == params[
            'vocabSize'], "Vocab size mismatch!"
        for key in loadedParams:
            params[key] = loadedParams[key]
        aqmBot.eval()

        # load qBot
        for key in excludeParams:
            params[key] = dlparams[key]
        aqmQ, loadedParams, _ = utils.loadModel(params, 'qbot', overwrite=True)
        assert aqmQ.encoder.vocabSize == params[
            'vocabSize'], "Vocab size mismatch!"
        for key in loadedParams:
            params[key] = loadedParams[key]
        aqmQ.eval()
        for key in excludeParams:
            params[key] = dlparams[key]
        aqmBot.setQuestioner(aqmQ)

    elif params['aqmQStartFrom']:
        from visdial.models.aqm_questioner import AQMQuestioner
        aqmBot = AQMQuestioner()
        aqmBot.eval()

        params['qstartFrom'] = params['aqmQStartFrom']
        aqmQ, loadedParams, _ = utils.loadModel(params, 'qbot', overwrite=True)
        assert aqmQ.encoder.vocabSize == params[
            'vocabSize'], "Vocab size mismatch!"
        for key in loadedParams:
            params[key] = loadedParams[key]
        aqmQ.eval()
        for key in excludeParams:
            params[key] = dlparams[key]
        aqmBot.setQuestioner(aqmQ)

        params['startFrom'] = params['aqmAStartFrom']
        aqmA, loadedParams, _ = utils.loadModel(params, 'abot', overwrite=True)
        assert aqmA.encoder.vocabSize == dataset.vocabSize, "Vocab size mismatch!"
        for key in loadedParams:
            params[key] = loadedParams[key]
        aqmA.eval()
        aqmBot.setAppAnswerer(aqmA)

    for key in excludeParams:
        params[key] = dlparams[key]

    pprint.pprint(params)
    #viz.addText(pprint.pformat(params, indent=4))
    print("Running evaluation!")

    numRounds = params['numRounds']
    if 'ckpt_iterid' in params:
        iterId = params['ckpt_iterid'] + 1
    else:
        iterId = -1

    if 'test' in splits:
        split = 'test'
        splitName = 'test - {}'.format(params['evalTitle'])
    else:
        split = 'val'
        splitName = 'full Val - {}'.format(params['evalTitle'])

    print("Using split %s" % split)
    dataset.split = split

    if 'ABotRank' in params['evalModeList']:
        if params['aqmstartFrom']:
            aBot = aqmBot.appAnswerer
            print('evaluating appBot of AQM')
        print("Performing ABotRank evaluation")
        rankMetrics = rankABot(
            aBot, dataset, split, scoringFunction=utils.maskedNll,
            expLowerLimit=params['expLowerLimit'],
            expUpperLimit=params['expUpperLimit'])
        print(rankMetrics)
        for metric, value in rankMetrics.items():
            plotName = splitName + ' - ABot Rank'
            #viz.linePlot(iterId, value, plotName, metric, xlabel='Iterations')

    if 'QBotRank' in params['evalModeList']:
        print("Performing QBotRank evaluation")
        rankMetrics, roundRanks = rankQBot(qBot, dataset, split,
            expLowerLimit=params['expLowerLimit'],
            expUpperLimit=params['expUpperLimit'],
            verbose=1)
        for metric, value in rankMetrics.items():
            plotName = splitName + ' - QBot Rank'
            #viz.linePlot(iterId, value, plotName, metric, xlabel='Iterations')

        for r in range(numRounds + 1):
            for metric, value in roundRanks[r].items():
                plotName = '[Iter %d] %s - QABots Rank Roundwise' % \
                            (iterId, splitName)
                #viz.linePlot(r, value, plotName, metric, xlabel='Round')

    if 'QABotsRank' in params['evalModeList']:
        print("Performing QABotsRank evaluation")
        outputPredFile = "data/visdial/visdial/output_predictions_rollout.h5"
        rankMetrics, roundRanks = rankQABots(
            qBot, aBot, dataset, split, beamSize=params['beamSize'],
            expLowerLimit=params['expLowerLimit'],
            expUpperLimit=params['expUpperLimit'],
            zeroCaption=params['zeroCaption'],
            randomCaption=params['randomCaption'],
            numRounds=params['runRounds'])
        for metric, value in rankMetrics.items():
            plotName = splitName + ' - QABots Rank'
            #viz.linePlot(iterId, value, plotName, metric, xlabel='Iterations')

        for r in range(numRounds + 1):
            for metric, value in roundRanks[r].items():
                plotName = '[Iter %d] %s - QBot All Metrics vs Round'%\
                            (iterId, splitName)
                #viz.linePlot(r, value, plotName, metric, xlabel='Round')

    if 'AQMBotRank' in params['evalModeList']:
        print("Performing AQMBotRank evaluation")
        outputPredFile = "data/visdial/visdial/output_predictions_rollout.h5"
        rankMetrics, roundRanks = AQMRunner(
            aqmBot, aBot, dataset, split, beamSize=params['beamSize'], realQA=params['aqmRealQA'],
            saveLogs=params['saveLogs'], showQA=params['showQA'],
            expLowerLimit=params['expLowerLimit'],
            expUpperLimit=params['expUpperLimit'],
            selectedBatchIdxs=params['selectedBatchIdxs'],
            numRounds=params['runRounds'],
            lda=params['lambda'],
            onlyGuesser=params['onlyGuesser'],
            numQ=params['numQ'],
            qbeamSize=params['qbeamSize'],
            numImg=params['numImg'],
            alpha=params['alpha'],
            numA=params['numA'],
            randQ=params['randQ'],
            randA=params['randA'],
            zeroCaption=params['zeroCaption'],
            randomCaption=params['randomCaption'],
            gamma=params['gamma'],
            delta=params['delta'],
            gen1Q=params['gen1Q'],
            gtQ=params['gtQ'],
            noHistory=params['noHistory'],
            slGuesser=params['slGuesser'],
            resampleEveryDialog=params['resampleEveryDialog'],
            aqmSetting=aqmSetting,
        ).rankQuestioner()
        for metric, value in rankMetrics.items():
            plotName = splitName + ' - QABots Rank'
            #viz.linePlot(iterId, value, plotName, metric, xlabel='Iterations')

        for r in range(numRounds + 1):
            for metric, value in roundRanks[r].items():
                plotName = '[Iter %d] %s - QBot All Metrics vs Round'%\
                            (iterId, splitName)
                #viz.linePlot(r, value, plotName, metric, xlabel='Round')

    if 'dialog' in params['evalModeList']:
        print("Performing dialog generation...")
        split = 'test'
        outputFolder = "dialog_output/results"
        os.makedirs(outputFolder, exist_ok=True)
        outputPath = os.path.join(outputFolder, "results.json")
        dialogDump(
            params,
            dataset,
            split,
            aBot=aBot,
            qBot=qBot,
            expLowerLimit=params['expLowerLimit'],
            expUpperLimit=params['expUpperLimit'],
            beamSize=params['beamSize'],
            savePath=outputPath)

    if 'AQMdialog' in params['evalModeList']:
        print("Performing AQM dialog generation...")

        split = 'test'
        AQMRunner(
            aqmBot, aBot, dataset, split, beamSize=params['beamSize'], realQA=params['aqmRealQA'],
            saveLogs=params['saveLogs'], showQA=params['showQA'],
            expLowerLimit=params['expLowerLimit'],
            expUpperLimit=params['expUpperLimit'],
            selectedBatchIdxs=params['selectedBatchIdxs'],
            numRounds=params['runRounds'],
            lda=params['lambda'],
            onlyGuesser=params['onlyGuesser'],
            numQ=params['numQ'],
            qbeamSize=params['qbeamSize'],
            numImg=params['numImg'],
            alpha=params['alpha'],
            numA=params['numA'],
            randQ=params['randQ'],
            randA=params['randA'],
            zeroCaption=params['zeroCaption'],
            randomCaption=params['randomCaption'],
            gamma=params['gamma'],
            delta=params['delta'],
            gen1Q=params['gen1Q'],
            gtQ=params['gtQ'],
            noHistory=params['noHistory'],
            slGuesser=params['slGuesser'],
            resampleEveryDialog=params['resampleEveryDialog'],
            aqmSetting=aqmSetting,
        ).dialogDump(params)


    #viz.addText("Evaluation run complete!")

if __name__ == '__main__':
    # read the command line options
    params = options.readCommandLine()

    # seed rng for reproducibility
    manualSeed = 1234
    # manualSeed = params['randomSeed']
    if "AQMdemo" not in params["evalModeList"]:
        random.seed(manualSeed)
        np.random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        if params['useGPU']:
            torch.cuda.manual_seed_all(manualSeed)
    main(params)
