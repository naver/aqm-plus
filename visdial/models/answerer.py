import copy
import torch
import torch.nn as nn
from visdial.models.agent import Agent
import visdial.models.encoders.hre as hre_enc
import visdial.models.decoders.gen as gen_dec
from utils import utilities as utils
from torch.autograd import Variable


class Answerer(Agent):
    # initialize
    def __init__(self, encoderParam, decoderParam, verbose=1):
        '''
            A-Bot Model

            Uses an encoder network for input sequences (questions, answers and
            history) and a decoder network for generating a response (answer).
        '''
        super(Answerer, self).__init__()
        self.encType = encoderParam['type']
        self.decType = decoderParam['type']

        # Encoder
        if verbose:
            print('Encoder: ' + self.encType)
            print('Decoder: ' + self.decType)
        if 'hre' in self.encType:
            self.encoder = hre_enc.Encoder(**encoderParam)
        else:
            raise Exception('Unknown encoder {}'.format(self.encType))

        # Decoder
        if 'gen' == self.decType:
            self.decoder = gen_dec.Decoder(**decoderParam)
        else:
            raise Exception('Unkown decoder {}'.format(self.decType))

        # Share word embedding parameters between encoder and decoder
        self.decoder.wordEmbed = self.encoder.wordEmbed

        # Initialize weights
        utils.initializeWeights(self.encoder)
        utils.initializeWeights(self.decoder)
        self.reset()

    def reset(self):
        '''Delete dialog history.'''
        self.caption = None
        self.answers = []
        self.encoder.reset()
    
    def exportParams(self, deepcopy=False):
        ans = [self.answers[-1].data] if len(self.answers) else []
        ret = (self.caption, ans, self.encoder.exportParam())
        return ret if not deepcopy else copy.deepcopy(ret)
    
    def importParams(self, params, lowerbound=None, upperbound=None, rep=1, deepcopy=False):
        if lowerbound is not None:
            assert upperbound is not None
        if deepcopy:
            params = copy.deepcopy(params)
        self.caption, ans = params[0], params[1]
        if lowerbound is not None:
            assert upperbound is not None
            self.caption = self.caption[lowerbound:upperbound, :]
            ans = [a[lowerbound:upperbound, :] for a in ans]
        self.caption = self.caption.repeat(rep, 1)
        self.answers = [Variable(a.repeat(rep, 1)) for a in ans]
        self.encoder.importParam(params[2], lowerbound=lowerbound,
                            upperbound=upperbound, rep=rep)
    
    def selectParam(self, params, selIdx):
        caption, ans, encParams = params
        try:
            selIdx = Variable(torch.cuda.LongTensor(selIdx))
        except:
            selIdx = Variable(torch.LongTensor(selIdx))
        newParams = []
        newParams.append(caption.index_select(0, selIdx))
        newParams.append([a.index_select(0, selIdx.data) for a in ans])
        newParams.append(self.encoder.selectParam(encParams, selIdx))
        return newParams

    def mergeParam(self, param1, param2):
        caption, ans, encParams = param1
        caption2, ans2, encParams2 = param2
        newCaption = torch.cat([caption, caption2], 0)
        assert len(ans) == len(ans2)
        newAns = [torch.cat([ans[-1], ans2[-1]], 0)] if len(ans) > 0 else []
        newEnc = self.encoder.mergeParam(encParams, encParams2)
        return (newCaption, newAns, newEnc)
    
    def splitParam(self, params, batchSize):
        caption, ans, encParams = params
        newParams = []
        caption = caption.split(batchSize, 0)
        sigAns = None
        if len(ans) > 0:
            sigAns = ans[-1].split(batchSize, 0)
        encParams = self.encoder.splitParam(encParams, batchSize)
        for i in range(len(caption)):
            if sigAns is None:
                newParams.append([caption[i], 
                                    [], encParams[i]])
            else:
                newParams.append([caption[i], 
                                    [sigAns[i]], encParams[i]])
        return newParams


    def observe(self, round, ans=None, caption=None, **kwargs):
        '''
        Update Q-Bot percepts. See self.encoder.observe() in the corresponding
        encoder class definition (hre).
        '''
        if caption is not None:
            assert round == -1, "Round number should be -1 when observing"\
                                " caption, got %d instead"
            self.caption = caption
        if ans is not None:
            # assert round == len(self.answers),\
                # "Round number does not match number of answers observed"
            self.answers.append(ans)
        self.encoder.observe(round, ans=ans, caption=caption, **kwargs)
    

    def forward(self, debug=False):
        '''
        Forward pass the last observed answer to compute its log
        likelihood under the current decoder RNN state.
        '''
        encStates = self.encoder(debug=debug)
        if len(self.answers) > 0:
            decIn = self.answers[-1]
        elif self.caption is not None:
            decIn = self.caption
        else:
            raise Exception('Must provide an input sequence')

        logProbs = self.decoder(encStates, inputSeq=decIn)
        return logProbs

    def forwardDecode(self, inference='sample', beamSize=1, maxSeqLen=20, topk=1, retLogProbs=False):
        '''
        Decode a sequence (answer) using either sampling or greedy inference.
        An answer is decoded given the current state (dialog history). This
        can be called at every round after a question is observed.

        Arguments:
            inference : Inference method for decoding
                'sample' - Sample each word from its softmax distribution
                'greedy' - Always choose the word with highest probability
                           if beam size is 1, otherwise use beam search.
            beamSize  : Beam search width
            maxSeqLen : Maximum length of token sequence to generate
        '''
        encStates = self.encoder()
        return self.decoder.forwardDecode(
            encStates,
            maxSeqLen=maxSeqLen,
            inference=inference,
            beamSize=beamSize,
            topk=topk,
            retLogProbs=retLogProbs)

    def evalOptions(self, options, optionLens, scoringFunction):
        '''
        Given the current state (question and conversation history), evaluate
        a set of candidate answers to the question.

        Output:
            Log probabilities of candidate options.
        '''
        states = self.encoder()
        return self.decoder.evalOptions(states, options, optionLens,
                                        scoringFunction)

    def reinforce(self, reward):
        # Propogate reinforce function call to decoder
        return self.decoder.reinforce(reward)
