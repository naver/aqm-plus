"""
Copyright 2019 NAVER Corp.
Licensed under BSD 3-clause
"""

import sys
import gc
import json
import os
import copy
import csv
import string
from timeit import default_timer as timer

from uuid import uuid1
import datetime

import torch
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import pairwise_distances
from nltk.tokenize import word_tokenize

import visdial.metrics as metrics


class AQMRunner:
    def __init__(self, qBot, aBot, dataset, split, exampleLimit=400, beamSize=1, realQA=False,
                 saveLogs=False, showQA=False, expLowerLimit=None, expUpperLimit=None,
                 selectedBatchIdxs=None, numRounds=None,
                 lda=6, onlyGuesser=False, numQ=20, numImg=20, numA=20, randQ=0, randA=0,
                 alpha=0.5, zeroCaption=0, randomCaption=0, qbeamSize=None, gamma=0, delta=0,
                 gen1Q=0, gtQ=0, noHistory=0, slGuesser=0, resampleEveryDialog=0,
                 aqmSetting=None):

        lda = 0. if (randomCaption or zeroCaption) else lda

        qbeamSize = qbeamSize if (qbeamSize is not None) else numQ
        assert qbeamSize >= numQ

        if realQA:
            print('Evaluating in RealQA Mode')

        assert (expLowerLimit is not None and expUpperLimit is not None
                or expLowerLimit is None and expLowerLimit is None)

        self.dataset = dataset
        self.original_split = dataset.split
        if dataset.split != split:
            print('Original split:', dataset.split)
            print('Using split:', split)

        dataset.split = split
        self.split = split

        numRounds = numRounds if (numRounds is not None) else dataset.numRounds

        self.batchSize = 1

        numExamples = exampleLimit if (exampleLimit is not None) else dataset.numDataPoints[split]
        self.numBatches = (numExamples - 1) // self.batchSize + 1

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batchSize,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset.collate_fn)

        # Set dataset for answerer in AQM-qbot
        qBot.setData(dataset, split)

        self.gtImgFeatures = []

        # For idx to word translation
        self.ind2word = dataset.ind2word
        # [8:] to strip off <START>
        self.to_str_gt = lambda w: str(" ".join(
            [self.ind2word[x] for x in filter(lambda x: x>0,w.data.cpu().numpy())]
        ))[8:]
        self.to_str_pred = lambda w, l: str(" ".join(
            [self.ind2word[x] for x in list(filter(
                lambda x:x>0,w.data.cpu().numpy()
            ))][:l.data.cpu()[0]]
        ))[8:]

        if '%s_img_fnames' % split not in dataset.data.keys():
            raise RuntimeError("Need coco directory and info as input " \
                  "to -cocoDir and -cocoInfo arguments for locating " \
                  "coco image files.")
        self.getImgFileName = lambda x: self.dataset.data['%s_img_fnames' % split][x]
        self.getImgId = lambda x: int(self.getImgFileName(x)[:-4][-12:])

        for batch in self.dataloader:
            batch = prepareBatch(dataset, batch)
            self.gtImgFeatures.append(Variable(batch['img_feat'], volatile=True))
        self.gtImgFeatures = torch.cat(self.gtImgFeatures, 0)   # [numAllImgs, featDim]

        # set bots
        self.qBot = qBot
        self.aBot = aBot

        # set arguments
        self.realQA = realQA
        self.saveLogs = saveLogs
        self.expLowerLimit, self.expUpperLimit = expLowerLimit, expUpperLimit
        self.selectedBatchIdxs = selectedBatchIdxs
        self.numRounds = numRounds

        self.beamSize = beamSize  # aBot beamSize
        self.qbeamSize = qbeamSize  # qBot beamSize

        self.realQA = realQA
        self.showQA = showQA

        # important arguments
        self.numImg = numImg
        self.numQ = numQ
        self.numA = numA
        self.lda = lda

        # ablation study modes
        self.onlyGuesser = onlyGuesser
        self.slGuesser = slGuesser

        self.gen1Q = gen1Q
        self.gtQ = gtQ
        self.randQ = randQ
        self.randA = randA
        self.resampleEveryDialog = resampleEveryDialog

        self.zeroCaption = zeroCaption
        self.randomCaption = randomCaption

        self.noHistory = noHistory

        self.alpha = alpha

        self.gamma = gamma
        self.delta = delta

        # mutable!
        self.sampledRandQuesForDialog = False
        self.sampledRandAnsForDialog = False

        self.roundwiseFeaturePreds = [[] for _ in range(numRounds + 1)]
        self.ranks = None

        # for randQ or gen1Q
        self._questions, self._quesLens = None, None

        # for randA
        self._randAnswers, self._randAnsLens = None, None

        self.aqmSetting = aqmSetting

        print(self.__repr__())

        if self.saveLogs:
            assert self.aqmSetting is not None

            self.logsDir = self.getLogsDir()
            if not os.path.exists(self.logsDir):
                os.makedirs(self.logsDir, exist_ok=True)
                print(self.logsDir)

            json.dump(self.property, open(os.path.join(self.logsDir, "meta.json"), "w"), indent=4)

    def getImgURL(self, batchIdx):
        def pad(idx, size=12):
            idx = str(idx)
            idx = '0' * (size - len(idx)) + idx
            return idx

        image_root = "https://vision.ece.vt.edu/mscoco/images/val2014/"
        imgId = self.getImgId(batchIdx)
        url = '%sCOCO_val2014_%s.jpg' % (image_root, pad(imgId))
        return url

    def getLogsDir(self):
        p = copy.deepcopy({key: value for key, value in self.property.items() if value != 0})

        dirname = "%s_%s__%s__%s__%s" % (
            p["hpSetting"],
            p["strategy"],
            p["qBot"],
            p["aBot"],
            p["aprxABot"],
        )

        del p["hpSetting"]
        del p["strategy"]
        del p["qBot"]
        del p["aBot"]
        del p["aprxABot"]
        del p["split"]

        additionals = "__".join("%s_%s" % (key, value) for key, value in p.items())
        dirname = "__".join([dirname, additionals])
        return dirname

    @property
    def property(self):
        p = self.aqmSetting if (self.aqmSetting is not None) else {}
        p.update({
            'realQA': self.realQA,
            'split': self.split,
            'numImg': self.numImg,
            'numQ': self.numQ,
            'numA': self.numA,
            'lambda': self.lda,
            'onlyGuesser': self.onlyGuesser,
            'slGuesser': self.slGuesser,
            'gen1Q': self.gen1Q,
            'randQ': self.randQ,
            'randA': self.randA,
            'resampleEveryDialog': self.resampleEveryDialog,
            'zeroCaption': self.zeroCaption,
            'randomCaption': self.randomCaption,
            'noHistory': self.noHistory,
        })

        return p

    def __repr__(self):
        p = {key: value for key, value in self.property.items() if value != 0}
        return json.dumps(p, indent=4)

    def dialogDump(self, params):
        outputFolder = os.path.join("dialog_output", "results")
        os.makedirs(outputFolder, exist_ok=True)

        outputPath = os.path.join(outputFolder, "results.json")

        text = {"data": []}

        for batch_idx, batch in enumerate(self.dataloader):

            if self.selectedBatchIdxs is not None:
                if batch_idx not in self.selectedBatchIdxs:
                    continue
            else:
                if self.expLowerLimit is not None:
                    if batch_idx < self.expLowerLimit:
                        continue
                    if batch_idx >= self.expUpperLimit:
                        break
                else:
                    if batch_idx >= self.numBatches:
                        break

            dialog = self.runDialog(batch_idx, batch, printSummary=True, dialogDump=True)
            text["data"].extend(dialog)

        text['opts'] = {
            'qbot': self.aqmSetting["qBot"] + " - " + self.aqmSetting["aprxABot"],
            'abot': self.aqmSetting["aBot"],
            'backend': 'cudnn',
            'beamLen': self.qbeamSize,
            'gpuid': 0,
            'params': sys.argv,
            'imgNorm': params['imgNorm'],
            'inputImg': params['inputImg'],
            'inputJson': params['inputJson'],
            'inputQues': params['inputQues'],
            'loadPath': 'checkpoints/',
            'maxThreads': 1,
            'resultPath': outputFolder,
            'sampleWords': 0,
            'temperature': 1,
            'useHistory': True,
            'useIm': True,
            'AQM': 1,
        }
        with open(outputPath, "w") as fp:
            print("Writing dialog text data to file: {}".format(outputPath))
            json.dump(text, fp)
        print("Done!")

        self.dataset.split = self.original_split

    def rankQuestioner(self):
        if self.selectedBatchIdxs is not None:
            raise RuntimeError("Cannot use selectedBatchIdxs when evaluating PMR")

        for batch_idx, batch in enumerate(self.dataloader):
            if self.expLowerLimit is not None:
                if batch_idx < self.expLowerLimit:
                    continue
                if batch_idx >= self.expUpperLimit:
                    break
            else:
                if batch_idx >= self.numBatches:
                    break

            self.runDialog(batch_idx, batch, printSummary=True)

        rankMetricsRounds = []
        print("Percentile mean rank (round, mean, low, high)")

        if self.saveLogs:
            csv_file = open(os.path.join(self.logsDir, "PMR__%s_%s.csv"), mode="w")
            writer = csv.DictWriter(csv_file, ["round", "meanPercRank", "percRankLow", "percRankHigh"])
            writer.writeheader()

        if not self.slGuesser:
            for round in range(self.numRounds + 1):
                '''
                1. sort and get index within each example
                2. get ranks
                '''
                # num_examples x num_examples
                rank = self.ranks[round]
                rankMetrics = metrics.computeMetrics(Variable(torch.from_numpy(rank)))
                poolSize = len(self.dataset)
                # assert len(ranks) == len(dataset)
                meanRank = rank.mean()
                se = rank.std() / np.sqrt(poolSize)
                meanPercRank = 100 * (1 - (meanRank / poolSize))
                percRankLow = 100 * (1 - ((meanRank + se) / poolSize))
                percRankHigh = 100 * (1 - ((meanRank - se) / poolSize))
                print('%d\t%f\t%f\t%f' % (round, meanPercRank, percRankLow, percRankHigh))
                rankMetrics['percentile'] = meanPercRank
                rankMetricsRounds.append(rankMetrics)

                if self.saveLogs:
                    writer.writerow({"round": round, "meanPercRank": meanPercRank,
                                     "percRankLow": percRankLow, "percRankHigh": percRankHigh})
        else:
            gtFeatures = self.gtImgFeatures.data.cpu().numpy()
            for round in range(self.numRounds + 1):
                predFeatures = torch.cat(self.roundwiseFeaturePreds[round],
                                         0).data.cpu().numpy()  # (
                dists = pairwise_distances(predFeatures, gtFeatures)
                # num_examples x num_examples
                ranks = []
                for i in range(dists.shape[0]):
                    # Computing rank of i-th prediction vs all images in split
                    if self.expLowerLimit is None:
                        rank = int(np.where(dists[i, :].argsort() == i)[0]) + 1
                    else:
                        rank = int(np.where(dists[i, :].argsort() == i + self.expLowerLimit)[0]) + 1
                    ranks.append(rank)
                ranks = np.array(ranks)
                rankMetrics = metrics.computeMetrics(Variable(torch.from_numpy(ranks)))
                poolSize = len(self.dataset)
                meanRank = ranks.mean()
                se = ranks.std() / np.sqrt(poolSize)
                meanPercRank = 100 * (1 - (meanRank / poolSize))
                percRankLow = 100 * (1 - ((meanRank + se) / poolSize))
                percRankHigh = 100 * (1 - ((meanRank - se) / poolSize))
                print('%d\t%f\t%f\t%f' % (round, meanPercRank, percRankLow, percRankHigh))
                rankMetrics['percentile'] = meanPercRank
                rankMetricsRounds.append(rankMetrics)

                if self.saveLogs:
                    writer.writerow({"round": round, "meanPercRank": meanPercRank,
                                     "percRankLow": percRankLow, "percRankHigh": percRankHigh})

        self.dataset.split = self.original_split
        return rankMetricsRounds[-1], rankMetricsRounds

    def _getQuestion(self, round, imgkProbs, imgkIdxs, ansParams, gtQuestions, gtQuesLens):
        if self.onlyGuesser or self.qbeamSize == 1:
            # It has to run even under realQA mode
            questions, quesLens, quesLogProbs = self.qBot.forwardDecode(
                inference='greedy', retLogProbs=True)
        elif self.randQ:
            questions, quesLens = self._getRandQQuestions(round)
            quesLogProbs = None
        elif self.gen1Q:
            questions, quesLens = self._getGen1QQuestions(round)
            quesLogProbs = None
        elif self.gtQ:
            questions, quesLens = gtQuestions[:, round], gtQuesLens[:, round]
        else:
            questions, quesLens, quesLogProbs = self.qBot.forwardDecode(
                inference='greedy', beamSize=self.qbeamSize,
                topk=self.numQ, gamma=self.gamma, delta=self.delta, retLogProbs=True)

        # choose question that maximizes information gain
        if not (self.onlyGuesser or self.qbeamSize == 1 or self.gtQ):
            questions = questions.squeeze(0)
            quesLens = quesLens.squeeze(0)
            imgkProbs = imgkProbs.unsqueeze(1)

            questions, quesLens = self._getMaxInfogainQuestions(
                round, questions, quesLens, quesLogProbs,
                imgkIdxs, imgkProbs, ansParams)

        # observe question
        if self.realQA:
            questions, quesLens = gtQuestions[:, round], gtQuesLens[:, round]

        return questions, quesLens

    def _getAnswer(self, round, gtAnswers, gtAnsLens):
        if self.realQA:
            answers, ansLens = gtAnswers[:, round], gtAnsLens[:, round]
        else:
            answers, ansLens = self.aBot.forwardDecode(
                inference='greedy', beamSize=self.beamSize)
        return answers, ansLens

    def _getRandQQuestions(self, round):
        if round == 0 and (self.resampleEveryDialog or not self.sampledRandQuesForDialog):
            questions, quesLens = getRandomQuestions(self.dataset, self.numQ)
            self._questions, self._quesLens = questions.clone(), quesLens.clone()
            self.sampledRandQuesForDialog = True
        else:
            questions, quesLens = self._questions, self._quesLens

        return questions, quesLens

    def _getGen1QQuestions(self, round):
        if round == 0:  # generate only at first turn
            questions, quesLens = self.qBot.forwardDecode(
                inference='greedy', beamSize=self.qbeamSize, topk=self.numQ,
                gamma=self.gamma, delta=self.delta)
            self._questions, self._quesLens = questions.clone(), quesLens.clone()
        else:
            questions, quesLens = self._questions, self._quesLens
        return questions, quesLens

    def _getRandAAnswers(self, round):
        if round == 0 and (self.resampleEveryDialog or not self.sampledRandAnsForDialog):
            randAnswers, randAnsLens = getRandomAnswers(self.dataset, self.numA)
            self._randAnswers, self._randAnsLens = randAnswers.clone(), randAnsLens.clone()
            self.sampledRandAnsForDialog = True
        else:
            randAnswers, randAnsLens = self._randAnswers, self._randAnsLens
        return randAnswers, randAnsLens

    def _getMaxInfogainQuestions(self, round, questions, quesLens, quesLogProbs,
                                 imgkIdxs, imgkProbs, ansParams):
        randAnswers, randAnsLens = self._getRandAAnswers(round) if self.randA else (None, None)

        # calcualte infogain
        p_a = self.qBot.p_a(imgkIdxs, questions, quesLens, ansParams,
                       randAnswers=randAnswers, randAnsLens=randAnsLens,
                       noHistory=self.noHistory, numA=self.numA)

        p_prime_a = torch.sum(imgkProbs * p_a, dim=1, keepdim=True)
        infogain = torch.sum(torch.sum(imgkProbs * p_a * torch.log(p_a / p_prime_a), dim=-1), dim=-1)
        if quesLogProbs is not None:
            a = quesLogProbs.data.squeeze() * self.alpha
            infogain = infogain + a

        # choose question that maximizes infogain
        _, nextQIdx = torch.max(infogain, dim=0)

        questions = questions[nextQIdx, :]
        quesLens = quesLens[nextQIdx]

        return questions, quesLens

    def _getCaption(self, batch):
        if not self.zeroCaption and not self.randomCaption:
            cap, cap_len = batch['cap'], batch['cap_len']

        elif self.zeroCaption:
            cap = torch.zeros_like(batch['cap'])
            cap_len = batch['cap_len']
            gc.collect()

        elif self.randomCaption:
            cap, cap_len = getRandomCaption(self.dataset)
            gc.collect()

        return cap, cap_len

    def runDialog(self, batch_idx, batch, printSummary=False, dialogDump=False):
        # prepare data
        batch = prepareBatch(self.dataset, batch)

        cap, cap_len = self._getCaption(batch)
        caption = Variable(cap, volatile=True)
        captionLens = Variable(cap_len, volatile=True)
        gtQuestions = Variable(batch['ques'], volatile=True)
        gtQuesLens = Variable(batch['ques_len'], volatile=True)
        gtAnswers = Variable(batch['ans'], volatile=True)
        gtAnsLens = Variable(batch['ans_len'], volatile=True)
        image = Variable(batch['img_feat'], volatile=True)

        # 0-th round
        if not self.realQA:
            self.aBot.eval()
            self.aBot.reset()
            self.aBot.observe(-1, image=image, caption=caption, captionLens=captionLens)

        # Even not in realQA mode, questioner still need to observe the dialogue
        # in order to guess
        self.qBot.eval()
        self.qBot.reset()
        self.qBot.observe(-1, caption=caption, captionLens=captionLens)

        predFeature = self.qBot.predictImage()
        self.qBot.setPrior(predFeature, self.gtImgFeatures, self.lda)

        if not self.onlyGuesser:
            imgkLogProbs, imgkIdxs = self.qBot.prior.topk(self.numImg, dim=0)
            imgkProbs = normProb(imgkLogProbs)  # (numImg, )
            predDist = self.qBot.prior.unsqueeze(0).repeat(self.numRounds + 1, 1)  # (numRounds + 1, numAllImg)
        else:
            imgkLogProbs, imgkIdxs = None, None
            imgkProbs = None

        if self.slGuesser:
            self.roundwiseFeaturePreds[0].append(predFeature)

        ansParams = None
        prior = None

        if dialogDump:
            imgIds = [self.getImgId(idx) for idx in batch["index"]]
            dialog = [{"dialog": [], "image_id": imgId} for imgId in imgIds]

            for j in range(self.batchSize):
                caption_str = self.to_str_pred(caption[j], captionLens[j])
                dialog[j]["caption"] = caption_str

        if self.showQA or printSummary:
            print("[batch %4s] %s" % (batch_idx, "=" * 50))
            print(" >> image   | %s" % self.getImgURL(batch_idx))
            print(" >> caption | %s" % self.to_str_pred(caption[0], captionLens[0])
                if not self.realQA else self.to_str_gt(caption[0]))

        rounds_info = {}

        start_t = timer()

        for round in range(self.numRounds):
            round_start_t = timer()

            # get question (aqmBot question)
            questions, quesLens = self._getQuestion(round, imgkProbs, imgkIdxs, ansParams, gtQuestions, gtQuesLens)

            # observe question
            self.qBot.observe(round, ques=questions, quesLens=quesLens)
            if not self.realQA:
                self.aBot.observe(round, ques=questions, quesLens=quesLens)

            # get real answer (aBot answer)
            answers, ansLens = self._getAnswer(round, gtAnswers, gtAnsLens)

            # observe real answer
            self.qBot.observe(round, ans=answers, ansLens=ansLens)
            if not self.realQA:
                self.aBot.observe(round, ans=answers, ansLens=ansLens)

            # calculations for guesser
            # if onlyGuesser, it calculates predDist at once after the dialog
            if not self.onlyGuesser:
                ansParams, prior = self.qBot.guess_sigRnd(
                    ansParams, prior=prior, round=round, noHistory=self.noHistory)

                imgkLogProbs, imgkIdxs = prior.topk(self.numImg, dim=0)
                imgkProbs = normProb(imgkLogProbs)  # p_reg_c
                predDist[round + 1, :] = prior

            if self.slGuesser:   # for one round...
                predFeature = self.qBot.predictImage()
                self.roundwiseFeaturePreds[round + 1].append(predFeature)

            # prepare logs
            q_str = (self.to_str_pred(questions[0], quesLens[0])
                if not self.realQA else self.to_str_gt(questions[0]))
            a_str = (self.to_str_pred(answers[0], ansLens[0])
                if not self.realQA else self.to_str_gt(answers[0]))

            if self.showQA:
                print("[round %2s] %s >> %s" % (round + 1, q_str, a_str))

            if dialogDump:
                for j in range(self.batchSize):
                    dialog[j]["dialog"].append({
                        "question": q_str + " ",
                        "answer": a_str,
                    })

            round_end_t = timer()

            if self.saveLogs:
                round_info = {
                    "question": q_str,
                    "answer": a_str,
                    "rank": int(getRank(prior.unsqueeze(0), batch_idx)) if (not self.onlyGuesser and not self.slGuesser) else None,
                    "sec": (round_end_t - round_start_t),
                }
                rounds_info.update({round + 1: round_info})

        # dialog evaluation

        if self.onlyGuesser:
            predDist = self.qBot.guess(noHistory=self.noHistory) # (numRounds + 1, numAllImgs)

        if printSummary:
            print("[summary] %s" % ("-" * 53))

        if not self.slGuesser:
            rank = getRank(predDist, batch_idx)
            if printSummary:
                print(" >> rank | %s" % rank)

            _rank = np.expand_dims(rank, 1)
            self.ranks = _rank if (self.ranks is None) else np.concatenate([self.ranks, _rank], 1)
        else:
            rank = None

        end_t = timer()
        dialog_rate = (end_t - start_t)
        if printSummary:
            print(" >> rate | %5.2fs" % dialog_rate)

        if self.saveLogs:
            posteriors = dict()
            for round in range(predDist.size(0)):
                posteriorProb = normProb(predDist[round, :])
                posteriors[round] = {
                    _batch_idx: posteriorProb[_batch_idx]
                        for _batch_idx in range(predDist.size(1))
                }

            json.dump(posteriors,
                      open(os.path.join(self.logsDir, "posterior_%s.json" % batch_idx), "w"),
                      indent=4)

            dialog_info = {
                "image": self.getImgURL(batch_idx),
                "ranks": rank.tolist() if (rank is not None) else None,
                "sec": dialog_rate,
                "dialog": rounds_info,
            }
            json.dump(dialog_info, open(os.path.join(self.logsDir, "%s.json" % batch_idx), "w"), indent=4)

        if dialogDump:
            return dialog


def getRank(predDist, curIdx):
    '''
    Return:
        (Round, ) : Ranking
    '''
    _, sortedOrder = torch.sort(predDist, -1)
    sortedOrder = np.array(sortedOrder)
    rank = predDist.size()[-1] - np.where(sortedOrder == curIdx)[1]
    return rank


def normProb(logProbs, numFix=True):
    '''
    Augments:
        logProbs : 1D tensor

    Return:
        normProbs: Same size with normalized probability
    '''
    if numFix:
        logProbs -= torch.max(logProbs)
    probs = torch.exp(logProbs)
    sumProbs = torch.sum(probs)
    normProbs = probs / sumProbs
    return normProbs


def getRandomQuestions(dataset, numQ):
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn)
    randomDialogIdxs = sorted(np.random.randint(len(dataloader), size=numQ))

    randomQuestionsList = []
    randomQuesLens = []

    dialogIdx = 0
    for idx, batch in enumerate(dataloader):
        while randomDialogIdxs[dialogIdx] == idx:
            batch = prepareBatch(dataset, batch)

            ques = batch['ques']
            quesLens = batch['ques_len']

            quesIdx = np.random.randint(ques.size(1))

            randomQuestionsList.append(ques[0, quesIdx, :])
            randomQuesLens.append(quesLens[0, quesIdx])

            dialogIdx += 1
            if dialogIdx == numQ:
                break
        if dialogIdx == numQ:
            break

    maxQuesLen = max(randomQuesLens)
    randomQuestions = torch.zeros(numQ, maxQuesLen).long()
    if dataset.useGPU:
        randomQuestions = randomQuestions.cuda()

    for idx, randomQuestion in enumerate(randomQuestionsList):
        randomQuestions[idx, :randomQuesLens[idx]] = randomQuestion[:randomQuesLens[idx]]

    randomQuesLens = torch.LongTensor(randomQuesLens)
    if dataset.useGPU:
        randomQuesLens = randomQuesLens.cuda()

    return Variable(randomQuestions, volatile=True), Variable(randomQuesLens, volatile=True)


def getRandomAnswers(dataset, numA):
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn)
    randomDialogIdxs = sorted(np.random.randint(len(dataloader), size=numA))

    randomAnswersList = []
    randomAnsLens = []

    dialogIdx = 0
    for idx, batch in enumerate(dataloader):
        while randomDialogIdxs[dialogIdx] == idx:
            batch = prepareBatch(dataset, batch)

            ans = batch['ans']
            ansLens = batch['ans_len']

            ansIdx = np.random.randint(ans.size(1))

            randomAnswersList.append(ans[0, ansIdx, :])
            randomAnsLens.append(ansLens[0, ansIdx])

            dialogIdx += 1
            if dialogIdx == numA:
                break
        if dialogIdx == numA:
            break

    maxAnsLen = max(randomAnsLens)
    randomAnswers = torch.zeros(numA, maxAnsLen).long()
    if dataset.useGPU:
        randomAnswers = randomAnswers.cuda()

    for idx, randomAnswer in enumerate(randomAnswersList):
        randomAnswers[idx, :randomAnsLens[idx]] = randomAnswer[:randomAnsLens[idx]]

    randomAnsLens = torch.LongTensor(randomAnsLens)
    if dataset.useGPU:
        randomAnsLens = randomAnsLens.cuda()

    return Variable(randomAnswers, volatile=True), Variable(randomAnsLens, volatile=True)


def getRandomCaption(dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn)

    for idx, batch in enumerate(dataloader):
        caption = batch['cap']
        captionLens = batch['cap_len']
        return caption, captionLens


def getRandomBatch(dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collate_fn)

    for _, batch in enumerate(dataloader):
        batch_idx = batch["index"][0]
        return batch_idx, batch


def prepareBatch(dataset, batch):
    if dataset.useGPU:
        batch = {key: v.cuda() if hasattr(v, 'cuda') else v for key, v in batch.items()}
    else:
        batch = {key: v.contiguous() if hasattr(v, 'cuda') else v for key, v in batch.items()}
    return batch
