import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from utils import utilities as utils


class Encoder(nn.Module):
    def __init__(self,
                 vocabSize,
                 embedSize,
                 rnnHiddenSize,
                 numLayers,
                 useIm,
                 imgEmbedSize,
                 imgFeatureSize,
                 numRounds,
                 isAnswerer,
                 dropout=0,
                 startToken=None,
                 endToken=None,
                 **kwargs):
        super(Encoder, self).__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.rnnHiddenSize = rnnHiddenSize
        self.numLayers = numLayers
        assert self.numLayers > 1, "Less than 2 layers not supported!"
        if useIm:
            self.useIm = useIm if useIm != True else 'early'
        else:
            self.useIm = False
        self.imgEmbedSize = imgEmbedSize
        self.imgFeatureSize = imgFeatureSize
        self.numRounds = numRounds
        self.dropout = dropout
        self.isAnswerer = isAnswerer
        self.startToken = startToken
        self.endToken = endToken

        # modules
        self.wordEmbed = nn.Embedding(
            self.vocabSize, self.embedSize, padding_idx=0)

        # question encoder
        # image fuses early with words
        if self.useIm == 'early':
            quesInputSize = self.embedSize + self.imgEmbedSize
            dialogInputSize = 2 * self.rnnHiddenSize
            self.imgNet = nn.Linear(self.imgFeatureSize, self.imgEmbedSize)
            self.imgEmbedDropout = nn.Dropout(0.5)
        elif self.useIm == 'late':
            quesInputSize = self.embedSize
            dialogInputSize = 2 * self.rnnHiddenSize + self.imgEmbedSize
            self.imgNet = nn.Linear(self.imgFeatureSize, self.imgEmbedSize)
            self.imgEmbedDropout = nn.Dropout(0.5)
        elif self.isAnswerer:
            quesInputSize = self.embedSize
            dialogInputSize = 2 * self.rnnHiddenSize
        else:
            dialogInputSize = self.rnnHiddenSize
        if self.isAnswerer:
            self.quesRNN = nn.LSTM(
                quesInputSize,
                self.rnnHiddenSize,
                self.numLayers,
                batch_first=True,
                dropout=0)

        # history encoder
        self.factRNN = nn.LSTM(
            self.embedSize,
            self.rnnHiddenSize,
            self.numLayers,
            batch_first=True,
            dropout=0)

        # dialog rnn
        self.dialogRNN = nn.LSTMCell(dialogInputSize, self.rnnHiddenSize)
        # A global counter for save and read
        # self.captionEmbedded = False
        self.firstForwarded = False
        self.isLoaded = False

    def reset(self):
        # batchSize is inferred from input
        self.batchSize = 0

        # Input data
        self.image = None
        self.imageEmbed = None

        self.captionTokens = None
        self.captionEmbed = None
        self.captionLens = None

        self.questionTokens = []
        self.questionEmbeds = []
        self.questionLens = []

        self.answerTokens = []
        self.answerEmbeds = []
        self.answerLengths = []

        # Hidden embeddings
        self.factEmbeds = []
        self.questionRNNStates = []
        self.dialogRNNInputs = []
        self.dialogHiddens = []
        # self.captionEmbedded = False
        self.firstForwarded = False
        self.isLoaded = False

    def _initHidden(self):
        '''Initial dialog rnn state - initialize with zeros'''
        # Dynamic batch size inference
        assert self.batchSize != 0, 'Observe something to infer batch size.'
        someTensor = self.dialogRNN.weight_hh.data
        h = someTensor.new(self.batchSize, self.dialogRNN.hidden_size).zero_()
        c = someTensor.new(self.batchSize, self.dialogRNN.hidden_size).zero_()
        return (Variable(h), Variable(c))

    def observe(self,
                round,
                image=None,
                caption=None,
                ques=None,
                ans=None,
                captionLens=None,
                quesLens=None,
                ansLens=None):
        '''
        Store dialog input to internal model storage

        Note that all input sequences are assumed to be left-aligned (i.e.
        right-padded). Internally this alignment is changed to right-align
        for ease in computing final time step hidden states of each RNN
        '''
        if image is not None:
            assert round == -1
            self.image = image
            self.imageEmbed = None
            self.batchSize = len(self.image)
        if caption is not None:
            assert round == -1
            assert captionLens is not None, "Caption lengths required!"
            caption, captionLens = self.processSequence(caption, captionLens)
            self.captionTokens = caption
            self.captionLens = captionLens
            self.batchSize = len(self.captionTokens)
        if ques is not None:
            # assert round == len(self.questionEmbeds)
            assert quesLens is not None, "Questions lengths required!"
            ques, quesLens = self.processSequence(ques, quesLens)
            self.questionTokens.append(ques)
            self.questionLens.append(quesLens)
        if ans is not None:
            # assert round == len(self.answerEmbeds)
            assert ansLens is not None, "Answer lengths required!"
            ans, ansLens = self.processSequence(ans, ansLens)
            self.answerTokens.append(ans)
            self.answerLengths.append(ansLens)
    
    def processSequence(self, seq, seqLen):
        ''' Strip <START> and <END> token from a left-aligned sequence'''
        return seq[:, 1:], seqLen - 1

    def embedInputDialog(self):
        '''
        Lazy embedding of input:
            Calling observe does not process (embed) any inputs. Since
            self.forward requires embedded inputs, this function lazily
            embeds them so that they are not re-computed upon multiple
            calls to forward in the same round of dialog.
        '''
        # Embed image, occurs once per dialog
        if self.isAnswerer and self.imageEmbed is None:
            self.imageEmbed = self.imgNet(self.imgEmbedDropout(self.image))
        # Embed caption, occurs once per dialog
        if not self.isLoaded and self.captionEmbed is None:
            self.captionEmbed = self.wordEmbed(self.captionTokens)
        # Embed questions
        while len(self.questionEmbeds) < len(self.questionTokens):
            idx = len(self.questionEmbeds)
            self.questionEmbeds.append(
                self.wordEmbed(self.questionTokens[idx]))
        # Embed answers
        while len(self.answerEmbeds) < len(self.answerTokens):
            idx = len(self.answerEmbeds)
            self.answerEmbeds.append(self.wordEmbed(self.answerTokens[idx]))

    def embedFact(self, factIdx, debug):
        '''Embed facts i.e. caption and round 0 or question-answer pair otherwise'''
        # Caption
        if factIdx == 0 and not self.isLoaded:
            seq, seqLens = self.captionEmbed, self.captionLens
            factEmbed, states = utils.dynamicRNN(
                self.factRNN, seq, seqLens, returnStates=True)
            # self.captionEmbedded = True
        # QA pairs
        else:
            idx = factIdx if self.isLoaded else factIdx - 1
            quesTokens, quesLens = \
                self.questionTokens[idx], self.questionLens[idx]
            if debug:
                print("quesTokens", quesTokens)

            ansTokens, ansLens = \
                self.answerTokens[idx], self.answerLengths[idx]

            qaTokens = utils.concatPaddedSequences(
                quesTokens, quesLens, ansTokens, ansLens, padding='right')
            qa = self.wordEmbed(qaTokens)
            qaLens = quesLens + ansLens
            qaEmbed, states = utils.dynamicRNN(
                self.factRNN, qa, qaLens, returnStates=True)
            factEmbed = qaEmbed
        factRNNstates = states
        if debug:
            print("Fact", factEmbed, factRNNstates)
        self.factEmbeds.append((factEmbed, factRNNstates))

    def embedQuestion(self, qIdx, debug):
        '''Embed questions'''
        quesIn = self.questionEmbeds[qIdx]
        quesLens = self.questionLens[qIdx].squeeze()
        if self.useIm == 'early':
            image = self.imageEmbed.unsqueeze(1).repeat(1, quesIn.size(1), 1)
            quesIn = torch.cat([quesIn, image], 2)
        qEmbed, states = utils.dynamicRNN(
            self.quesRNN, quesIn, quesLens, returnStates=True)
        quesRNNstates = states
        if debug:
            print("Qeus", qEmbed, quesRNNstates)
        self.questionRNNStates.append((qEmbed, quesRNNstates))

    def concatDialogRNNInput(self, histIdx, debug):
        quesIdx = histIdx if not self.isLoaded else histIdx + 1
        currIns = [self.factEmbeds[histIdx][0]] 
        if self.isAnswerer:
            currIns.append(self.questionRNNStates[quesIdx][0])
        if self.useIm == 'late':
            currIns.append(self.imageEmbed)
        if debug:
            print("RNNInput", currIns)
        hist_t = torch.cat(currIns, -1)
        self.dialogRNNInputs.append(hist_t)

    def embedDialog(self, dialogIdx, debug):
        if dialogIdx == 0 and not self.isLoaded:
            hPrev = self._initHidden()
        else:
            hPrev = self.dialogHiddens[-1]
        inptIdx = dialogIdx if not self.isLoaded else dialogIdx - 1
        inpt = self.dialogRNNInputs[inptIdx]
        hNew = self.dialogRNN(inpt, hPrev)
        if debug:
            print('dialogInput', inpt)
            print('dialogHiddens', hPrev)
            print('hNew', hNew)
        self.dialogHiddens.append(hNew)

    def exportParam(self):
        '''
        Arugment:
            selIdx : (select_nums,) To index batch whose parameters will be returned. None
                     for all parameters.
        Return:
            params : parameters.
        '''
        assert self.isAnswerer
        dialogHiddens = self.dialogHiddens[-1] if len(self.dialogHiddens) else None
        quesToken = self.questionTokens[-1] if len(self.questionTokens) else None
        quesLen = self.questionLens[-1] if len(self.questionLens) else None
        ansToken = self.answerTokens[-1] if len(self.answerTokens) else None
        ansLen = self.answerLengths[-1] if len(self.answerLengths) else None
        captionTokens = self.captionTokens
        captionLens = self.captionLens
        image = self.image
        volatile = not image.requires_grad
        gpuList = [quesToken, quesLen, ansToken, ansLen, image, captionTokens, captionLens]
        for i in range(len(gpuList)):
            if gpuList[i] is not None:
                gpuList[i] = gpuList[i].data
        if dialogHiddens is not None:
            d0, d1 = dialogHiddens
            d0, d1 = d0.data, d1.data
            dialogHiddens = (d0, d1)
        gpuList.append(dialogHiddens)
        retList = [self.firstForwarded, volatile]
        return gpuList + retList

    
    def importParam(self, params, lowerbound=None, upperbound=None, rep=1):
        '''
        Arugment:
            params                  : Parameters from exportParam()
            lowerbound & upperbound : If they're set, only load parameters of batch 
                                      [lowerbound, upperbound)
            rep                     : Set how many times you want your parameter to repeat. For example,
                                      original parameter [[p1], [p2]] will become 
                                      [[p1],[p2],[p1],[p2]] if repeat=2.
        Return:
            params : parameters.
        '''
        assert self.isAnswerer
        if lowerbound is not None: 
            assert upperbound is not None
        self.reset()
        quesToken, quesLen, ansToken, ansLen, \
        self.image, self.captionTokens, self.captionLens, dialogHiddens, \
        self.firstForwarded, volatile = params
        self.isLoaded = self.firstForwarded
        if lowerbound is not None:
            quesToken = quesToken[lowerbound:upperbound, :]
            quesLen = quesLen[lowerbound:upperbound, :]
            ansToken = ansToken[lowerbound:upperbound, :]
            ansLen = ansLen[lowerbound:upperbound, :]
            self.image = self.image[lowerbound:upperbound, :]
            self.captionTokens = self.captionTokens[lowerbound:upperbound, :]
            self.captionLens = self.captionLens[lowerbound:upperbound]
            d0, d1 = dialogHiddens
            d0 = d0[lowerbound:upperbound, :]
            d1 = d1[lowerbound:upperbound, :]
            dialogHiddens = (d0, d1)
        # repeat & preprocessing
        slfList = [self.questionTokens, self.questionLens, \
                    self.answerTokens, self.answerLengths]
        cands = [quesToken, quesLen, ansToken, ansLen]
        for i in range(len(cands)):
            if cands[i] is not None:
                slfList[i].append(Variable(cands[i].repeat(rep, 1), volatile=volatile))
        if dialogHiddens is not None:
            dialogHiddens = (Variable(dialogHiddens[0].repeat(rep, 1), volatile=volatile), \
                           Variable(dialogHiddens[1].repeat(rep, 1), volatile=volatile))
            self.dialogHiddens.append(dialogHiddens)
        self.image = Variable(self.image.repeat(rep, 1), volatile=volatile)
        self.captionTokens = Variable(self.captionTokens.repeat(rep, 1), volatile=volatile)
        self.captionLens = Variable(self.captionLens.repeat(rep), volatile=volatile)
        self.batchSize = len(self.captionTokens)

    def selectParam(self, params, selIdx):
        # selIdx = torch.Tensor(selIdx).long().cuda()
        selIdx = selIdx.data
        quesToken, quesLen, ansToken, ansLen, \
        image, captionTokens, captionLens, dialogHiddens, \
        firstForwarded, volatile = params
        selList = [quesToken, quesLen, ansToken, ansLen, image, \
                    captionTokens, captionLens]
        newParams = []
        for i in range(len(selList)):
            app = None
            if selList[i] is not None:
                app = selList[i].index_select(0, selIdx)
            newParams.append(app)
        app = None
        if dialogHiddens is not None:
            app = (dialogHiddens[0].index_select(0, selIdx), \
                    dialogHiddens[1].index_select(0, selIdx))
        newParams.append(app)
        newParams.append(firstForwarded)
        newParams.append(volatile)
        return newParams

    def mergeParam(self, param1, param2):
        # Merge two parameters with image index (well this design is 
        # really bad), and split it according to batchSize.
        quesToken, quesLen, ansToken, ansLen, \
        image, captionTokens, captionLens, dialogHiddens, \
        firstForwarded, volatile = param1
        assert firstForwarded
        quesToken2, quesLen2, ansToken2, ansLen2, \
        image2, captionTokens2, captionLens2, dialogHiddens2, \
        firstForwarded2, volatile2 = param2
        assert firstForwarded2
        assert volatile == volatile2
        a = [quesToken, quesLen, ansToken, ansLen, \
        image, captionTokens, captionLens]
        b = [quesToken2, quesLen2, ansToken2, ansLen2, \
        image2, captionTokens2, captionLens2]
        c = []
        for i in range(len(a)):
            temp = torch.cat([a[i], b[i]], 0)
            c.append(temp)
        da0, da1 = dialogHiddens
        db0, db1 = dialogHiddens2
        d0 = torch.cat([da0, db0], 0)
        d1 = torch.cat([da1, db1], 0)
        c.append((d0, d1))
        c.append(firstForwarded)
        c.append(volatile)
        return c
    
    def splitParam(self, params, batchSize):
        quesToken, quesLen, ansToken, ansLen, \
        image, captionTokens, captionLens, dialogHiddens, \
        firstForwarded, volatile = params
        extractList = [quesToken, quesLen, ansToken, ansLen, \
                        image, captionTokens, captionLens]
        for i in range(len(extractList)):
            extractList[i] = extractList[i].split(batchSize, 0)
        resParams = [list(a) for a in zip(*extractList)]
        d0, d1 = dialogHiddens
        d0 = d0.split(batchSize, 0)
        d1 = d1.split(batchSize, 0)
        resDiaHidden = list(zip(d0, d1))
        for i, res in enumerate(resParams):
            res.append(resDiaHidden[i])
            res.append(firstForwarded)
            res.append(volatile)
        return resParams

    def forward(self, debug=False):
        '''
        Returns:
            A tuple of tensors (H, C) each of shape (batchSize, rnnHiddenSize)
            to be used as the initial Hidden and Cell states of the Decoder.
            See notes at the end on how (H, C) are computed.
        '''

        # Lazily embed input Image, Captions, Questions and Answers
        self.embedInputDialog()

        if self.isAnswerer:
            # For A-Bot, current round is the number of facts present,
            # which is number of questions observed - 1 (as opposed
            # to len(self.answerEmbeds), which may be inaccurate as
            round = len(self.questionEmbeds) - 1
        else:
            # For Q-Bot, current round is the number of facts present,
            # which is same as the number of answers observed
            round = len(self.answerEmbeds)

        # Lazy computation of internal hidden embeddings (hence the while loops)

        # Infer any missing facts
        factRound = round - 1 if self.isLoaded else round
        while len(self.factEmbeds) <= factRound:
            factIdx = len(self.factEmbeds)
            self.embedFact(factIdx, debug=debug)
        

        # Embed any un-embedded questions (A-Bot only)
        if self.isAnswerer:
            while len(self.questionRNNStates) <= round:
                qIdx = len(self.questionRNNStates)
                self.embedQuestion(qIdx, debug=debug)

        # Concat facts and/or questions (i.e. history) for input to dialogRNN
        dialogRNNRound = round - 1 if self.isLoaded else round
        while len(self.dialogRNNInputs) <= dialogRNNRound:
            histIdx = len(self.dialogRNNInputs)
            self.concatDialogRNNInput(histIdx, debug=debug)

        # Forward dialogRNN one step
        while len(self.dialogHiddens) <= round:
            dialogIdx = len(self.dialogHiddens)
            self.embedDialog(dialogIdx, debug=debug)


        # Latest dialogRNN hidden state
        dialogHidden = self.dialogHiddens[-1][0]
        if debug:
            print('dialogHidden', self.dialogHiddens)
        
        self.firstForwarded = True

        '''
        Return hidden (H_link) and cell (C_link) states as per the following rule:
        (Currently this is defined only for numLayers == 2)
        If A-Bot:
          C_link == Question encoding RNN cell state (quesRNN)
          H_link ==
              Layer 0 : Question encoding RNN hidden state (quesRNN)
              Layer 1 : DialogRNN hidden state (dialogRNN)

        If Q-Bot:
            C_link == Fact encoding RNN cell state (factRNN)
            H_link ==
                Layer 0 : Fact encoding RNN hidden state (factRNN)
                Layer 1 : DialogRNN hidden state (dialogRNN)
        '''
        if self.isAnswerer:
            quesRNNstates = self.questionRNNStates[-1][1]  # Latest quesRNN states
            C_link = quesRNNstates[1]
            H_link = quesRNNstates[0][:-1]
            H_link = torch.cat([H_link, dialogHidden.unsqueeze(0)], 0)
        else:
            factRNNstates = self.factEmbeds[-1][1]  # Latest factRNN states
            C_link = factRNNstates[1]
            H_link = factRNNstates[0][:-1]
            H_link = torch.cat([H_link, dialogHidden.unsqueeze(0)], 0)
        

        return H_link, C_link
