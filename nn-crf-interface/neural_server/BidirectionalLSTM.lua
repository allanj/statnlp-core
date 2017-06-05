local BidirectionalLSTM, parent = torch.class('BidirectionalLSTM', 'AbstractNeuralNetwork')

function BidirectionalLSTM:__init(doOptimization, gpuid)
    parent.__init(self, doOptimization)
    self.data = {}
    self.gpuid = gpuid
end

function BidirectionalLSTM:initialize(javadata, ...)
    local gpuid = self.gpuid

    -- numInputList, inputDimList, embSizeList, outputDim,
    -- numLayer, hiddenSize, activation, dropout
    -- vocab

    self.data = {}
    local data = self.data
    data.sentences = listToTable(javadata:get("sentences"))
    data.hiddenSize = javadata:get("hiddenSize")
    data.optimizer = javadata:get("optimizer")

    local isTraining = javadata:get("isTraining")
    local outputAndGradOutputPtr = {... }
    if isTraining then
        self.outputPtr = torch.pushudata(outputAndGradOutputPtr[1], "torch.DoubleTensor")
        self.gradOutputPtr = torch.pushudata(outputAndGradOutputPtr[2], "torch.DoubleTensor")
    end

    -- what to forward
    if isTraining then self.word2idx = {} end
    self.x = self:prepare_input()

    if isTraining then
        self:createNetwork()
        print(self.net)

        self.params, self.gradParams = self.net:getParameters()
        if doOptimization then
            self:createOptimizer()
            -- no return array if optim is done here
        else
            self.params:retain()
            self.paramsPtr = torch.pointer(self.params)
            self.gradParams:retain()
            self.gradParamsPtr = torch.pointer(self.gradParams)
            return self.paramsPtr, self.gradParamsPtr
        end
    else
        self:createDecoderNetwork()
    end
end

function BidirectionalLSTM:createNetwork()
    local data = self.data
    local gpuid = self.gpuid

    local hiddenSize = data.hiddenSize

    local sharedLookupTable = nn.LookupTableMaskZero(#(self.word2idx), hiddenSize)

    -- forward rnn
    local fwd = nn.Sequential()
       :add(sharedLookupTable)
       :add(nn.FastLSTM(hiddenSize, hiddenSize):maskZero(1))

    -- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
    local fwdSeq = nn.Sequencer(fwd)

    -- backward rnn (will be applied in reverse order of input sequence)
    local bwd = nn.Sequential()
       :add(sharedLookupTable:sharedClone())
       :add(nn.FastLSTM(hiddenSize, hiddenSize):maskZero(1))
    local bwdSeq = nn.Sequencer(bwd)

    -- merges the output of one time-step of fwd and bwd rnns.
    -- You could also try nn.AddTable(), nn.Identity(), etc.
    local merge = nn.JoinTable(1, 1) 
    local mergeSeq = nn.Sequencer(merge)

    -- Assume that two input sequences are given (original and reverse, both are right-padded).
    -- Instead of ConcatTable, we use ParallelTable here.
    local parallel = nn.ParallelTable()
    parallel:add(fwdSeq):add(bwdSeq)
    local brnn = nn.Sequential()
       :add(parallel)
       :add(nn.ZipTable())
       :add(mergeSeq)
end

function MultiLayerPerceptron:createOptimizer()
    local data = self.data

    -- set optimizer. If nil, optimization is done by caller.
    print(string.format("Optimizer: %s", data.optimizer))
    self.doOptimization = data.optimizer ~= nil and data.optimizer ~= 'none'
    if self.doOptimization == true then
        if data.optimizer == 'sgd' then
            self.optimizer = optim.sgd
            self.optimState = {learningRate=data.learningRate}
        elseif data.optimizer == 'adagrad' then
            self.optimizer = optim.adagrad
            self.optimState = {learningRate=data.learningRate}
        elseif data.optimizer == 'adam' then
            self.optimizer = optim.adam
            self.optimState = {learningRate=data.learningRate}
        elseif data.optimizer == 'adadelta' then
            self.optimizer = optim.adadelta
            self.optimState = {learningRate=data.learningRate}
        elseif data.optimizer == 'lbfgs' then
            self.optimizer = optim.lbfgs
            self.optimState = {tolFun=10e-10, tolX=10e-16}
        end
    end
end

function BidirectionalLSTM:prepare_input()
    local gpuid = self.gpuid
    local data = self.data

    local sentences = data.sentences
    local maxLen = 0
    for i=1,#sentences do
        local len = #(stringx.split(sentences[i]," "))
        if len > maxLen then
            maxLen = len
        end
    end

    local sentence_toks = {}
    if self.isTraining then
        self.word2idx['<UNK>'] = 1
        for i=1,#sentences do
            local tokens = stringx.split(sentences[j]," ")
            table.insert(sentence_toks, tokens)
            for j=1,#tokens do
                local tok = tokens[j]
                self.word2idx[tok] = #(self.word2idx)+1
            end
        end
    end

    local inputs, inputs_rev
    for step=1,maxLen do
        inputs[step] = torch.LongTensor(#sentences)
        for j=1,#sentences do
            local tokens = sentences_toks[j]
            if step > #tokens then
                inputs[step][j] = 0
            else
                local tok = sentences_toks[j][step]
                local tok_id = self.word2idx[tok]
                if tok_id == nil then
                    tok_id = self.word2idx['<UNK>']
                end
                inputs[step][j] = tok_id
            end
        end
    end
    for step=1,maxLen do
        inputs_rev[step] = torch.LongTensor(#sentences)
        for j=1,#sentences do
            local tokens = sentences_toks[j]
            if step <= #tokens then
                inputs_rev[step][j] = inputs[#tokens-step+1][j]
            else
                inputs_rev[step][j] = 0
            end
        end
    end

    return {inputs, inputs_rev}
end
