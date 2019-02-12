"""
Copyright 2019 NAVER Corp.
Licensed under BSD 3-clause
"""

import numpy as np
import torch
from visdial.models.agent import Agent
from visdial.models.questioner import Questioner
from visdial.models.answerer import Answerer
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from torch.distributions import Categorical
import copy
import gc

from utils import utilities as utils
from sklearn.metrics.pairwise import pairwise_distances



class AQMQuestioner(Agent):
    def __init__(self, qEncoderParam=None, qDecoderParam=None, aEncoderParam=None, aDecoderParam=None, imgFeatureSize=0,
                 verbose=1):
        super(AQMQuestioner, self).__init__()
        if qEncoderParam is not None and qDecoderParam is not None:
            self.questioner = Questioner(qEncoderParam, qDecoderParam, imgFeatureSize, verbose)
            self.startToken = qEncoderParam['startToken']
            self.endToken = qEncoderParam['endToken']
        if aEncoderParam is not None and aDecoderParam is not None:
            self.appAnswerer = Answerer(aEncoderParam, aDecoderParam, verbose)
        self.questions = [] # for Guesser
        self.quesLens = [] # for Guesser
        self.answers = [] # for Guesser
        self.ansLens = []
        self.image = None
        self.caption = None
        self.captionLens = None
        self.dataset = None
        self.dataset_split = None
        self.logSoftmax = nn.LogSoftmax(dim=0)
    
    def setQuestioner(self, qBot):
        self.questioner = qBot
        self.startToken = qBot.encoder.startToken
        self.endToken = qBot.encoder.endToken
    
    def setAppAnswerer(self, aBot):
        self.appAnswerer = aBot
        self.startToken = aBot.encoder.startToken
        self.endToken = aBot.encoder.endToken
    
    def setData(self, dataset, dataset_split):
        self.dataset = dataset
        self.dataset_split = dataset_split
        # (Num_images,), each element with index c is p(a_j|image_c, q_j, a_(1:j-1), q_(1:j-1))
    
    def setPrior(self, predFeature, gtFeatures, lda=1):
        '''
        Arguments:
            (Image,) : Score for each image
            lda : lambda for scaling score
        '''
        diff = predFeature - gtFeatures
        score = -torch.sum(diff * diff, 1).sqrt()
        self.prior = lda * self.logSoftmax(score * 10).data

    def reset(self):
        self.questioner.reset()
        self.appAnswerer.reset()
        self.questions = [] # for Guesser
        self.quesLens = [] # for Guesser
        self.answers = [] # for Guesser
        self.ansLens = [] # for Guesser
        self.image = None
        self.caption = None
        self.captionLens = None
        self.prior = None

    def observe(self, round, ques=None, ans=None, image=None, **kwargs):
        """
        Round == -1:
        A observe image & caption
        Q observe caption
        Round > -1:
        A observe GT Q&A, forward and get loss
        Q observe GT Question, forward and get loss, then observe A
        Feature net predict and compute loss with mse

        How A work:
        First round: 
        embed image, caption, question & answer
        embed fact(first round is caption, following is QA pairs)
        embed un-embedded question(the new one)
        embed fact, new q and image to dialog RNN
        get state 0, 1: question RNN cell & hidden state, 2: DialogRNN hidden state

        encodeStates & caption->decoder
        Following:
        encodeState & last answer -> decoder
        
        Guessor need P(a|c) -> reset Q -> set caption C & image 0 -> Generate answer as well as prob
        Can sample from the return matrix, and use sum to compuate the log(P(a_j|c)), for speicifc round 
        of answer

        In our code (Eval):
        A, Q observe generated caption and image
        Guesser in Q has belief
        Q generate question
        A observe
        A generate answer
        Q observe
        Guess in Q has belief
        """

        if ques is not None:
            self.questions.append(ques)
            self.quesLens.append(kwargs['quesLens'])
        if ans is not None:
            self.answers.append(ans)
            self.ansLens.append(kwargs['ansLens'])
        if image is not None:
            self.image = image
        if 'captionLens' in kwargs:
            self.captionLens = kwargs['captionLens']
        if 'caption' in kwargs:
            self.caption = kwargs['caption']

        self.questioner.observe(round, ques, ans=ans, **kwargs)
        if self.training:
            self.appAnswerer.observe(round, ans, **kwargs, image=image, ques=ques)

    def p_a(self, candImgIdxs, candQues, candQuesLens, ansParams=None, batchSize=20, returnCandAnswers=False,
            randAnswers=None, randAnsLens=None, noHistory=False, numA=None):
        '''
        Calculate p_reg(a|c,q_t,history) (Normalized)

        Return:
            probArray : (quesSize, imageSize, ansSize), probArray[i, j, k] = P(a(q(i), c(k))|q(i), c(j))
        '''

        assert len(self.answers) == len(self.questions), "Didn't observe full QA pairs!"
        assert self.dataset is not None, "Please set dataset!"
        assert len(candQues) == len(candQuesLens)
        # assert len(candImgIdxs) % batchSize == 0 or batchSize > len(candImgIdxs)
        if not len(candImgIdxs) % batchSize == 0 or batchSize > len(candImgIdxs):
            batchSize = len(candImgIdxs)

        # (quesSize, 1), to fit the observe() function
        candQuesLens = candQuesLens.unsqueeze(1)

        original_split = self.dataset.split
        self.dataset.split = self.dataset_split
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=self.dataset.collate_fn)

        imgList = []
        batchSize = len(candImgIdxs) if len(candImgIdxs) < batchSize else batchSize
        iterTime = len(candImgIdxs) // batchSize
        candImgIdxs = torch.sort(candImgIdxs, dim=0)[0].cpu().numpy()
        imgidxIdx = 0 # To index candImgIdxs

        # assert iterTime == 1, "Deepcopy bug is not fixed!"
        myParams = []

        if numA is None:
            numA = batchSize
            aBeamSize = 1
        elif batchSize < numA:
            aBeamSize = numA // batchSize + (numA % batchSize > 0)
        elif batchSize >= numA:
            aBeamSize = 1

        if ansParams is None or noHistory:
            for idx, batch in enumerate(dataloader):
                if candImgIdxs[imgidxIdx] == idx:
                    if self.dataset.useGPU:
                        batch = {key: v.cuda() for key, v in batch.items() \
                                                        if hasattr(v, 'cuda')}
                    else:
                        batch = {key: v.contiguous() for key, v in batch.items() \
                                                        if hasattr(v, 'cuda')}

                    imgList.append(Variable(batch['img_feat'], volatile=True)) # (batch_size, img_embed)
                    imgidxIdx += 1
                    if imgidxIdx == len(candImgIdxs):
                        break
            imgList = torch.cat(imgList, 0)
        else:
            maxIdx = 0
            for bs, params in ansParams:
                maxIdx += bs
                selList = []
                while imgidxIdx < len(candImgIdxs) and candImgIdxs[imgidxIdx] < maxIdx:
                    selList.append(candImgIdxs[imgidxIdx] - maxIdx + bs)
                    imgidxIdx += 1
                if len(selList) > 0:
                    myParams.append(self.appAnswerer.selectParam(params, np.array(selList)))
            if len(myParams) > 1:
                params = myParams[0]
                for i in range(1, len(myParams)):
                    params = self.appAnswerer.mergeParam(params, myParams[i])
                myParams = self.appAnswerer.splitParam(params, batchSize)

        numRounds = len(self.questions)
        probArray = torch.FloatTensor(len(candQues), len(candImgIdxs), numA).fill_(0)
        if self.dataset.useGPU:
            probArray = probArray.cuda()

        candAns = []
        candAnsLens = []
        numQues = len(candQues)

        # Conditioning
        if ansParams is None or noHistory:
            for i in range(iterTime):
                image = imgList[i*batchSize:(i+1)*batchSize, :]
                caption = self.caption.repeat(batchSize, 1)
                captionLens = self.captionLens.repeat(batchSize)
                abotParams = None
                self.appAnswerer.eval(), self.appAnswerer.reset()
                self.appAnswerer.observe(-1, image=image, caption=caption, captionLens=captionLens)

                if not noHistory:
                    for round in range(len(self.questions)):
                        ques = self.questions[round].repeat(batchSize, 1)
                        ans = self.answers[round].repeat(batchSize, 1)
                        quesLens = self.quesLens[round].unsqueeze(0).repeat(batchSize, 1)
                        ansLens = self.ansLens[round].unsqueeze(0).repeat(batchSize, 1)
                        self.appAnswerer.observe(round, ques=ques, quesLens=quesLens)
                        self.appAnswerer.observe(round, ans=ans, ansLens=ansLens)
                        self.appAnswerer.forward()

                abotParams = self.appAnswerer.exportParams()
                myParams.append(abotParams)


        # First Round: P(a(q(i),c(j))|q(i),c(j))
        ques = candQues.repeat(1, batchSize).view(batchSize*numQues, -1)
        quesLens = candQuesLens.repeat(1, batchSize).view(batchSize*numQues, -1)
        for i in range(iterTime):
            self.appAnswerer.importParams(myParams[i], rep=numQues)
            self.appAnswerer.observe(len(self.questions), ques=ques, quesLens=quesLens)

            if randAnswers is None or randAnsLens is None:
                if aBeamSize != 1:
                    appAns, appAnsLens = self.appAnswerer.forwardDecode(inference='greedy',
                                                                        beamSize=aBeamSize, topk=aBeamSize)
                    if appAns.dim() == 2:
                        appAns = appAns.unsqueeze(1)
                        appAnsLens = appAnsLens.unsqueeze(1)

                    appAns = torch.cat(appAns.unsqueeze(1).split(batchSize, 0), 1)  # [numImg, numQ, aBeamSize, L]
                    appAns = appAns.permute(1, 2, 0, 3)  # [numQ, aBeamSize, numImg, L]
                    appAns = appAns.contiguous().view(numQues, aBeamSize*batchSize, -1)[:, :numA, :]    # [numQ, numA, L]
                    appAns = appAns.transpose(0, 1) # [numA, numQ, L]

                    appAnsLens = torch.cat(appAnsLens.unsqueeze(1).split(batchSize, 0), 1)  # [numImg, numQ, aBeamSize]
                    appAnsLens = appAnsLens.permute(1, 2, 0)    # [numQ, aBeamSize*numImg]
                    appAnsLens = appAnsLens.contiguous().view(numQues, aBeamSize*batchSize)[:, :numA]
                    appAnsLens = appAnsLens.transpose(0, 1) # [numA, numQ]

                else:   # aBeamSize == 1
                    appAns, appAnsLens = self.appAnswerer.forwardDecode(inference='greedy')
                    appAns = torch.cat(appAns.unsqueeze(1).split(batchSize, 0), 1)  # [numImg, numQ, L]
                    appAnsLens = torch.cat(appAnsLens.unsqueeze(1).split(batchSize, 0), 1)  # [numImg, numQ]
                    appAns = appAns[:numA, :, :]    # [numA, numQ, L]
                    appAnsLens = appAnsLens[:numA, :]    # [numA, numQ]

            else:   # randA
                appAns, appAnsLens = randAnswers, randAnsLens
                appAns = appAns.unsqueeze(1).repeat(1, numQues, 1)  # [numA, numQ, L]
                appAnsLens = appAnsLens.unsqueeze(1).repeat(1, numQues) # [numA, numQ]

            candAns.append(appAns)
            candAnsLens.append(appAnsLens)
            myParams[i] = self.appAnswerer.exportParams()
        
        candAns = torch.cat(candAns, dim=0) # (numA, queSize, maxLen), candAns[i, j, :] = a(c(i),q(j))
        candAnsLens = torch.cat(candAnsLens, dim=0) # (numA, queSize)


        # Second Round: P(a(q(i),c(k))|q(i),c(j))
        for i in range(iterTime):
            for aIdx in range(numA):    # p(a | Q, C)
                self.appAnswerer.importParams(myParams[i])
                ans = candAns[aIdx, :, :].repeat(1, batchSize).view(batchSize*numQues, -1)
                ansLens = candAnsLens[aIdx, :].unsqueeze(1).repeat(1, batchSize).view(batchSize*numQues, -1)
                self.appAnswerer.observe(len(self.questions), ans=ans, ansLens=ansLens)
                ansLogProbs = self.appAnswerer.forward()
                
                # Remove  <Start> from ans
                padColumn = ans.data.new(batchSize*numQues, 1).fill_(0)
                padColumn = Variable(padColumn)
                target = torch.cat([ans, padColumn], dim=1)[:, 1:]

                ansLogProbs = torch.gather(ansLogProbs, 2, target.unsqueeze(2)).squeeze(2)
                mask = Variable(torch.ByteTensor(ansLogProbs.size()).fill_(0))
                # have to find an elegant way
                cmpIdx = Variable(torch.LongTensor(1))
                if self.dataset.useGPU:
                    cmpIdx = cmpIdx.cuda()
                    mask = mask.cuda()
                for maskIdx in range(ansLogProbs.size()[1]):
                    cmpIdx.data.fill_(maskIdx)
                    mask[:, maskIdx] = torch.ge(cmpIdx, ansLens).byte()
                ansLogProbs.masked_fill_(mask, 0)
                curP = torch.sum(ansLogProbs, dim=1, keepdim=True).data
                curP = torch.cat(curP.split(batchSize, 0), 1).t()
                probArray[:, i*batchSize:(i+1)*batchSize, aIdx] = curP


        # Normalize
        probArray = torch.exp(probArray)
        probSum = torch.sum(probArray, dim=2, keepdim=True)
        probArray /= probSum
        self.dataset.split = original_split
        if returnCandAnswers:
            return probArray, candAns, candAnsLens
        return probArray    # [numQues, numImg, numA]

    def guess_sigRnd(self, ansParams=None, prior=None, round=None, batchSize=2048,
                     noHistory=False):
        # Guess image for single round using latest question & answer.
        
        # Arugments:
        #     appAnswerers: 
        #         A list of answerer objects returned from last round's guess.
        #         If it's None, new answerer will be used.
        #     prior:  
        #         (Image, ), Probability distribution from last round. 
        #         If it's None, self.prior will be used.
        #     batchSize: 
        #         # images are used in a batch to calculate the posterior.

        # Return:
        #     appAnswerers:
        #         A list of answerer objects storing necessary prior information 
        #         for next round.
        #     posterior: 
        #         (Image, ), the posterior probability for each image.
        if prior is None: 
            posterior = self.prior.clone()
            if self.dataset.useGPU:
                posterior = posterior.cuda()
        else:
            posterior = prior
        assert len(self.answers) == len(self.questions), "Didn't observe full QA pairs!"
        assert self.dataset is not None, "Please set dataset!"

        original_split = self.dataset.split
        self.dataset.split = self.dataset_split
        dataloader = DataLoader(
            self.dataset,
            batch_size=batchSize,
            shuffle=False,
            num_workers=0,
            collate_fn=self.dataset.collate_fn)

        accuIdx = 0
        if ansParams is None or noHistory:
            ansParams = []
            for idx, batch in enumerate(dataloader):
                if self.dataset.useGPU:
                    batch = {key: v.cuda() for key, v in batch.items() \
                                                    if hasattr(v, 'cuda')}
                else:
                    batch = {key: v.contiguous() for key, v in batch.items() \
                                                    if hasattr(v, 'cuda')}

                image = Variable(batch['img_feat'], volatile=True) # (batch_size, img_embed)

                # To avoid dimension mismatch at the end of epoch
                batchSize = image.shape[0]

                caption = self.caption.repeat(batchSize, 1)
                captionLens = self.captionLens.repeat(batchSize)
                
                self.appAnswerer.eval(), self.appAnswerer.reset()
                self.appAnswerer.observe(-1, image=image, caption=caption, captionLens=captionLens)

                ques = self.questions[-1].repeat(batchSize, 1)
                ans = self.answers[-1].repeat(batchSize, 1)
                quesLens = self.quesLens[-1].unsqueeze(0).repeat(batchSize, 1)
                ansLens = self.ansLens[-1].unsqueeze(0).repeat(batchSize, 1)
                self.appAnswerer.observe(round, ques=ques, quesLens=quesLens)
                self.appAnswerer.observe(round, ans=ans, ansLens=ansLens)
                ansLogProbs = self.appAnswerer.forward(debug=False)
                ansParams.append([batchSize, self.appAnswerer.exportParams(deepcopy=False)])

                # Remove  <Start> from ans
                padColumn = ans.data.new(batchSize, 1).fill_(0)
                padColumn = Variable(padColumn)
                target = torch.cat([ans, padColumn], dim=1)[:, 1:]

                ansLogProbs = torch.gather(ansLogProbs, 2, target.unsqueeze(2)).squeeze(2)
                curP = torch.sum(ansLogProbs[:, :self.ansLens[round].data[0]], dim=1).data
                posterior[accuIdx:accuIdx+batchSize] += curP
                accuIdx += batchSize
        else:
            for idx, bp in enumerate(ansParams):
                batchSize, params = bp
                self.appAnswerer.importParams(params)

                ques = self.questions[-1].repeat(batchSize, 1)
                ans = self.answers[-1].repeat(batchSize, 1)
                quesLens = self.quesLens[-1].unsqueeze(0).repeat(batchSize, 1)
                ansLens = self.ansLens[-1].unsqueeze(0).repeat(batchSize, 1)
                self.appAnswerer.observe(round, ques=ques, quesLens=quesLens)
                self.appAnswerer.observe(round, ans=ans, ansLens=ansLens)
                ansLogProbs = self.appAnswerer.forward(debug=False)
                ansParams[idx][1] = self.appAnswerer.exportParams(deepcopy=False)

                # Remove  <Start> from ans
                padColumn = ans.data.new(batchSize, 1).fill_(0)
                padColumn = Variable(padColumn)
                target = torch.cat([ans, padColumn], dim=1)[:, 1:]

                ansLogProbs = torch.gather(ansLogProbs, 2, target.unsqueeze(2)).squeeze(2)
                # print('ansLogProbs', ansLogProbs)
                curP = torch.sum(ansLogProbs[:, :self.ansLens[round].data[0]], dim=1).data
                posterior[accuIdx:accuIdx+batchSize] += curP
                accuIdx += batchSize

        self.dataset.split = original_split
        return ansParams, posterior

    def guess(self, batchSize=2048, noHistory=False):
        '''
        Assume that guesser will be called after observation is done

        Arguments:

        Return:
            (Round, Images) For each round, p(c|a)
        '''
        assert len(self.answers) == len(self.questions), "Didn't observe full QA pairs!"
        assert self.dataset is not None, "Please set dataset!"

        original_split = self.dataset.split
        self.dataset.split = self.dataset_split
        dataloader = DataLoader(
            self.dataset,
            batch_size=batchSize,
            shuffle=False,
            num_workers=0,
            collate_fn=self.dataset.collate_fn)

        numRounds = len(self.questions)

        p_a_history = torch.zeros(
            (numRounds + 1, self.dataset.numDataPoints[self.dataset_split]))
        if self.dataset.useGPU:
            p_a_history = p_a_history.cuda()
        if self.prior is not None:
            p_a_history[0, :] = self.prior
        
        accuIdx = 0
        for idx, batch in enumerate(dataloader):
            if self.dataset.useGPU:
                batch = {key: v.cuda() for key, v in batch.items() \
                                                if hasattr(v, 'cuda')}
            else:
                batch = {key: v.contiguous() for key, v in batch.items() \
                                                if hasattr(v, 'cuda')}

            image = Variable(batch['img_feat'], volatile=True) # (batch_size, img_embed)

            # To avoid dimension mismatch at the end of epoch
            batchSize = image.shape[0] 

            caption = self.caption.repeat(batchSize, 1)
            captionLens = self.captionLens.repeat(batchSize)

            if not noHistory:
                self.appAnswerer.eval(), self.appAnswerer.reset()
                self.appAnswerer.observe(-1, image=image, caption=caption, captionLens=captionLens)
                for round in range(numRounds):
                    ques = self.questions[round].repeat(batchSize, 1)
                    ans = self.answers[round].repeat(batchSize, 1)
                    quesLens = self.quesLens[round].unsqueeze(0).repeat(batchSize, 1)
                    ansLens = self.ansLens[round].unsqueeze(0).repeat(batchSize, 1)
                    self.appAnswerer.observe(round, ques=ques, quesLens=quesLens)
                    self.appAnswerer.observe(round, ans=ans, ansLens=ansLens)
                    ansLogProbs = self.appAnswerer.forward(debug=False)

                    # Remove  <Start> from ans
                    padColumn = ans.data.new(batchSize, 1).fill_(0)
                    padColumn = Variable(padColumn)
                    target = torch.cat([ans, padColumn], dim=1)[:, 1:]

                    ansLogProbs = torch.gather(ansLogProbs, 2, target.unsqueeze(2)).squeeze(2)
                    curP = torch.sum(ansLogProbs[:, :self.ansLens[round].data[0]], dim=1).data
                    p_a_history[round+1, accuIdx:accuIdx+batchSize] = p_a_history[round, accuIdx:accuIdx+batchSize] + curP
            else:
                for round in range(numRounds):
                    self.appAnswerer.eval(), self.appAnswerer.reset()
                    self.appAnswerer.observe(-1, image=image, caption=caption, captionLens=captionLens)
                    ques = self.questions[round].repeat(batchSize, 1)
                    ans = self.answers[round].repeat(batchSize, 1)
                    quesLens = self.quesLens[round].unsqueeze(0).repeat(batchSize, 1)
                    ansLens = self.ansLens[round].unsqueeze(0).repeat(batchSize, 1)
                    self.appAnswerer.observe(round, ques=ques, quesLens=quesLens)
                    self.appAnswerer.observe(round, ans=ans, ansLens=ansLens)
                    ansLogProbs = self.appAnswerer.forward(debug=False)

                    # Remove  <Start> from ans
                    padColumn = ans.data.new(batchSize, 1).fill_(0)
                    padColumn = Variable(padColumn)
                    target = torch.cat([ans, padColumn], dim=1)[:, 1:]

                    ansLogProbs = torch.gather(ansLogProbs, 2, target.unsqueeze(2)).squeeze(2)
                    curP = torch.sum(ansLogProbs[:, :self.ansLens[round].data[0]], dim=1).data
                    p_a_history[round + 1, accuIdx:accuIdx + batchSize] = p_a_history[round,
                                                                          accuIdx:accuIdx + batchSize] + curP

            accuIdx += batchSize
            
        self.dataset.split = original_split
        return p_a_history
        
    def forward(self):
        return self.questioner.forward() 
    
    def aForward(self):
        return self.appAnswerer.forward()
    
    def predictImage(self):
        return self.questioner.predictImage()
    
    def forwardDecode(self, inference='sample', beamSize=1, maxSeqLen=20, topk=1, retLogProbs=False, gamma=0, delta=0):
        return self.questioner.forwardDecode(inference=inference, beamSize=beamSize, maxSeqLen=maxSeqLen, topk=topk, retLogProbs=retLogProbs, gamma=gamma, delta=delta)

