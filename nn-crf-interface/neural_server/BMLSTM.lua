include 'PyTorchFastLSTM.lua'

local BMLSTM, parent = torch.class('BMLSTM', 'AbstractNeuralNetwork')

function BMLSTM:__init(doOptimization, gpuid)
    parent.__init(self, doOptimization)
    self.data = {}
    self.gpuid = gpuid
end

function BMLSTM:initialize(javadata, ...)
    self.data = {}
    local data = self.data
    data.sentences = listToTable(javadata:get("nnInputs"))
    data.hiddenSize = javadata:get("hiddenSize")
    data.embeddingSize = javadata:get("embeddingSize")
    data.numLabels = javadata:get("numLabels")
    local isTraining = javadata:get("isTraining")

    self.input = self:prepare_input()
    self.numSent = #data.sentences
    if self.net == nil and isTraining then
        -- means is initialized process and we don't have the input yet.
        self:createNetwork()
        print(self.net)
    end

    self.output = torch.Tensor()
    self.x1Tab = {}
    self.x1 = torch.LongTensor()
    if self.gpuid >= 0 then
        self.x1 = self.x1:cuda()
    end
    self.gradOutput = {}
    local outputAndGradOutputPtr = {... }
    if #outputAndGradOutputPtr > 0 then
        self.outputPtr = torch.pushudata(outputAndGradOutputPtr[1], "torch.DoubleTensor")
        self.gradOutputPtr = torch.pushudata(outputAndGradOutputPtr[2], "torch.DoubleTensor")
        return self:obtainParams()
    end
end

function BMLSTM:createNetwork()
    local data = self.data
    local hiddenSize = data.hiddenSize
    local embeddingSize = data.embeddingSize
    --- vocabulary x hiddenSize

    local embeddingLayer = nn.LookupTable(self.vocabSize, embeddingSize)
    self.lt = embeddingLayer
    --embeddingLayer.weight:fill(0.5)
    --embeddingLayer.weight[1]:fill(0)
    print("embeddingSize")
    print(embeddingLayer.weight:size())

    local fwdLSTM = PyTorchFastLSTM(embeddingSize, hiddenSize)--:maskZero(1)

    local fwd = nn.Sequential():add(embeddingLayer)
                :add(fwdLSTM)
    local fwdSeq = nn.Sequencer(fwd)
    local bwd = nn.Sequential():add(embeddingLayer:sharedClone())
                :add(PyTorchFastLSTM(embeddingSize, hiddenSize))
    local bwdSeq = nn.Sequential()
            :add(nn.ReverseTable())
            :add(nn.Sequencer(bwd))
            :add(nn.ReverseTable())

    local merge = nn.JoinTable(1, 1)
    local mergeSeq = nn.Sequencer(merge)

    local concat = nn.ConcatTable()
    concat:add(fwdSeq)
    concat:add(bwdSeq)
    local brnn = nn.Sequential()
       :add(concat)
       :add(nn.ZipTable())
       :add(mergeSeq)
    local lastLinear = nn.Linear(2 * hiddenSize, data.numLabels)
    local rnn = nn.Sequential()
        :add(brnn) 
        :add(nn.Sequencer(lastLinear))

    self.net = rnn
end

function BMLSTM:obtainParams()
    --make sure we will not replace this variable
    self.params, self.gradParams = self.net:getParameters()
    self.params:copy(torch.Tensor(self.params:size(1)):fill(2))
    print("Number of parameters: " .. self.params:nElement())
    self.params:retain()
    self.paramsPtr = torch.pointer(self.params)
    self.gradParams:retain()
    self.gradParamsPtr = torch.pointer(self.gradParams)
    return self.paramsPtr, self.gradParamsPtr
end

function BMLSTM:prepare_input()
    local data = self.data
    local sentences = data.sentences
    local sentence_toks = {}
    local maxLen = 0
    for i=1,#sentences do
        local tokens = stringx.split(sentences[i]," ")
        table.insert(sentence_toks, tokens)
        if #tokens > maxLen then
            maxLen = #tokens
        end
    end

    --note that inside if the vocab is already created
    --just directly return
    self:buildVocab(sentences, sentence_toks)    

    local inputs = {}
    local inputs_rev = {}
    for step=1,maxLen do
        inputs[step] = torch.LongTensor(#sentences)
        for j=1,#sentences do
            local tokens = sentence_toks[j]
            if step > #tokens then
                inputs[step][j] = 0 --padding token
            else
                local tok = sentence_toks[j][step]
                local tok_id = self.word2idx[tok]
                if tok_id == nil then
                    tok_id = self.word2idx['<UNK>']
                end
                inputs[step][j] = tok_id
            end
        end
    end
    print("max sentencen length:"..maxLen)
    self.maxLen = maxLen
    return inputs
end

function BMLSTM:buildVocab(sentences, sentence_toks)
    if self.idx2word ~= nil then
        --means the vocabulary is already created
        return 
    end
    self.idx2word = {}
    self.word2idx = {}
    self.word2idx['<PAD>'] = 0
    self.idx2word[0] = '<PAD>'
    --self.word2idx['<UNK>'] = 1
    --self.idx2word[1] = '<UNK>'
    self.vocabSize = 0
    for i=1,#sentences do
        local tokens = sentence_toks[i]
        for j=1,#tokens do
            local tok = tokens[j]
            local tok_id = self.word2idx[tok]
            if tok_id == nil then
                self.vocabSize = self.vocabSize+1
                self.word2idx[tok] = self.vocabSize
                self.idx2word[self.vocabSize] = tok
            end
        end
    end
    --for i,k in ipairs(self.idx2word) do
    --    print(i.." "..k)
    --end
    print("number of unique words:" .. self.vocabSize)
end

function BMLSTM:getForwardInput(isTraining, batchInputIds)
    if isTraining then
        if batchInputIds ~= nil then
            batchInputIds:add(1) -- because the sentence is 0 indexed.
            self.batchInputIds = batchInputIds
            self.x1 = torch.cat(self.x1, self.input, 2):index(1, batchInputIds)
            self.x1:resize(self.x1:size(1)*self.x1:size(2))
            torch.split(self.x1Tab, self.x1, batchInputIds:size(1), 1)
            self.batchInput = self.x1Tab
            return self.batchInput
        else
            return self.input
        end
    else
        return self.testInput
    end
end

function BMLSTM:forward(isTraining, batchInputIds)
    local nnInput = self:getForwardInput(isTraining, batchInputIds)
    local output_table = self.net:forward(nnInput)
    self.output = torch.cat(self.output, output_table, 1)
    if not self.outputPtr:isSameSizeAs(self.output) then
        self.outputPtr:resizeAs(self.output)
    end
    self.outputPtr:copy(self.output)
    
end

function BMLSTM:getBackwardInput()
    if self.batchInputIds ~= nil then
        return self.batchInput
    else
        return self.input
    end
end

function BMLSTM:getBackwardSentNum()
    if self.batchInputIds ~= nil then
        return self.batchInputIds:size(1)
    else
        return self.numSent
    end
end

function BMLSTM:backward()
    self.gradParams:zero()
    local nnInput = self:getBackwardInput()
    local gradOutputTensor = self.gradOutputPtr
    local backwardSentNum = self:getBackwardSentNum()
    torch.split(self.gradOutput, gradOutputTensor, backwardSentNum, 1)
    self.net:backward(nnInput, self.gradOutput)
    -- back propagation
    -- backward
end


