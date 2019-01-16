local TagBiLSTM, parent = torch.class('TagBiLSTM', 'AbstractNeuralNetwork')

function TagBiLSTM:__init(doOptimization, gpuid)
    parent.__init(self, doOptimization)
    self.data = {}
    self.gpuid = gpuid
end

function TagBiLSTM:initialize(javadata, ...)
    self.data = {}
    local data = self.data
    data.sentences = listToTable(javadata:get("nnInputs"))
    data.hiddenSize = javadata:get("hiddenSize")
    data.numLabels = javadata:get("numLabels")
    data.embedding = javadata:get("embedding")
    data.embeddingSize = javadata:get("embeddingSize")
    self.embeddingSize = data.embeddingSize
    self.numLabels = data.numLabels
    local isTraining = javadata:get("isTraining")


    if isTraining then
        self:loadEmbObj()
        self.input = self:prepare_input()
        self.numSent = #data.sentences
        self:buildLookupTable()
        --if we fix embeddings, we forward the input using our current Embedding layer
        if self.fixEmbedding then self.input = self.lt:forward(self.input):clone() end
    end

    if self.net == nil and isTraining then
        self:createNetwork()
        print(self.net)
    end

    if self.net == nil then
        --loading model (this code is used for testing when we save the model and load)
        --you may not be using this.
        self:load_model(modelPath)
    end

    if not isTraining then
        ---prepare the test input
        self.testInput = self:prepare_input()
        if self.fixEmbedding then self.testInput = self.lt:forward(self.testInput):clone() end
    end

    ---below is standard code
    self.output = torch.Tensor()
    self.gradOutput = torch.Tensor()
    local outputAndGradOutputPtr = {... }
    if #outputAndGradOutputPtr > 0 then
        self.outputPtr = torch.pushudata(outputAndGradOutputPtr[1], "torch.DoubleTensor")
        self.gradOutputPtr = torch.pushudata(outputAndGradOutputPtr[2], "torch.DoubleTensor")
        return self:obtainParams()
    end
end

function TagBiLSTM:loadEmbObj()
    local data = self.data
    self.embeddingSize = data.embeddingSize
    if data.embedding == 'glove' then
        --this function is in utils
        self.embeddingObject = loadGloveEmbObj()
        self.embeddingSize = 100
    elseif data.embedding == 'random' then 
        print("using random embedding")
    else
        error('unknown embedding type: '.. data.embedding)
    end
end

function TagBiLSTM:buildLookupTable()
    local sharedLookupTable = nn.LookupTableMaskZero(self.vocabSize, self.embeddingSize)
    ----embeddings are randomly initialized, if you used pretrained embedding, 
    ---you may used the following code
    --if self.data.embedding ~= 'random' then
    --    for i =1, self.vocabSize do
    --        sharedLookupTable.weight[i+1]:copy(self.embeddingObject:word2vec(self.idx2word[i]))
    --    end
    --end
    self.lt = sharedLookupTable
    if self.fixEmbedding then
        --make the gradient of this layer to be 0.
        self.lt.accGradParameters = function() end
        self.lt.parameters = function() end
        if self.gpuid >= 0 then self.lt:cuda() end
    end
end

function TagBiLSTM:createNetwork()
    local data = self.data
    local embeddingSize = self.embeddingSize
    
    print("Word Embedding layer: " .. self.lt.weight:size(1) .. " x " .. self.lt.weight:size(2))
    local rnn = nn.Sequential()
    if not self.fixEmbedding then
        rnn:add(self.lt)
    end
    ---our input is size:  batch x sent_len
    ---after an embedding layer: batch x sent_len x embedding size
    ----nn.SeqBRNN is a BiLSTM
    local brnn = nn.SeqBRNN(embeddingSize, embeddingSize, true, nn.JoinTable(3)) 
    ---after bilstm: batch x sent_len x hidden size
    brnn.batchfirst = true
    brnn.forwardModule.maskzero=true
    brnn.backwardModule.maskzero=true
    rnn:add(brnn)
    rnn:add(nn.SplitTable(2))
    local mapTab = nn.MapTable()
    local mapOp = nn.Sequential():add(nn.Linear(2 * embeddingSize, self.numLabels):noBias())
                    :add(nn.Unsqueeze(2))
    ---after above: batch x sent_len x numLabels
    mapTab:add(mapOp)
    rnn:add(mapTab)
    rnn:add(nn.JoinTable(2))
    rnn:add(nn.Transpose({1,2}))
    ---after above: sent_len x batch x numLabels
    if self.gpuid >= 0 then rnn:cuda() end
    self.net = rnn
    print("finishing buidling the neural network")
end

function TagBiLSTM:obtainParams()
    ----This function is standard, just copy to your code base
    self.params, self.gradParams = self.net:getParameters()
    print("Number of parameters: " .. self.params:nElement())
    if self.doOptimization then
        self:createOptimizer()
        -- no return array if optim is done here
    else
        if self.gpuid >= 0 then
            -- since the the network is gpu network.
            self.paramsDouble = self.params:double()
            self.paramsDouble:retain()
            self.params:retain()
            self.paramsPtr = torch.pointer(self.paramsDouble)
            self.gradParamsDouble = self.gradParams:double()
            self.gradParamsDouble:retain()
            self.gradParams:retain()
            self.gradParamsPtr = torch.pointer(self.gradParamsDouble)
            return self.paramsPtr, self.gradParamsPtr
        else
            self.params:retain()
            self.paramsPtr = torch.pointer(self.params)
            self.gradParams:retain()
            self.gradParamsPtr = torch.pointer(self.gradParams)
            return self.paramsPtr, self.gradParamsPtr
        end
    end
end

function TagBiLSTM:prepare_input()
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

    local inputs = torch.IntTensor(#sentences, maxLen)
    self:fillInputs(#sentences, inputs, maxLen, sentence_toks)
    if self.gpuid >= 0 then 
        inputs = inputs:cuda()
    end
    print("number of sentences: "..#sentences)
    print("max sentence length: "..maxLen)
    return inputs
end

function TagBiLSTM:fillInputs(numSents, inputTensor, maxLen, toks)
    for sId=1,numSents do
        local tokens = toks[sId]
        for step=1,maxLen do
            if step > #tokens then
                inputTensor[sId][step] = 0 ---padding token, always zero-padding
            else 
                local tok = tokens[step]
                local tok_id = self.word2idx[tok]
                if tok_id == nil then
                    tok_id = self.word2idx[self.unkToken]
                end
                inputTensor[sId][step] = tok_id
            end
        end
    end
end

function TagBiLSTM:buildVocab(sentences, sentence_toks)
    if self.idx2word ~= nil then
        --means the vocabulary is already created
        return 
    end
    self.idx2word = {}
    self.word2idx = {}
    self.word2idx['<PAD>'] = 0
    self.idx2word[0] = '<PAD>'
    self.word2idx['<UNK>'] = 1
    self.idx2word[1] = '<UNK>'
    self.vocabSize = 2
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
    print("number of unique words:" .. #self.idx2word)
end

function TagBiLSTM:forward(isTraining, batchInputIds)
    if self.gpuid >= 0 and not self.doOptimization then
        self.params:copy(self.paramsDouble:cuda())
    end
    if isTraining then
        self.net:training()
    else
        self.net:evaluate()
    end
    local nnInput = self:getForwardInput(isTraining, batchInputIds)
    local lstmOutput
    if isTraining then
        lstmOutput = self.net:forward(nnInput)
        -- print(lstmOutput:size())
    else
        -- lstmOutput = self.net:forward(nnInput)
        lstmOutput = torch.Tensor()
        if self.gpuid >=0 then lstmOutput = lstmOutput:cuda() end
        local instSize = nnInput:size(1) --number of sentences 
        local testBatchSize = 10   ---test batch size = 10
        --- forward 10 by 10, this number can be changed, 
        --- depends on your gpu workload
        for i = 1, instSize, testBatchSize do
            if i + testBatchSize - 1 > instSize then testBatchSize =  instSize - i + 1 end
            local tmpOut = self.net:forward(nnInput:narrow(1, i, testBatchSize))
            lstmOutput = torch.cat(lstmOutput, tmpOut, 2)
        end
    end
    if self.gpuid >= 0 then
        lstmOutput = lstmOutput:double()
    end 
    self.output = lstmOutput
    if not self.outputPtr:isSameSizeAs(self.output) then
        self.outputPtr:resizeAs(self.output)
    end
    self.outputPtr:copy(self.output)
    
end

function TagBiLSTM:getForwardInput(isTraining, batchInputIds)
    if isTraining then
        if batchInputIds ~= nil then
            --if batch training, we need to select part of the full input
            batchInputIds:add(1) -- because the sentence is 0 indexed.
            self.batchInputIds = batchInputIds
            ---index: select the first dimension.
            self.batchInput = self.input:index(1, batchInputIds)
            return self.batchInput
        else
            --not batch: then use the full input
            return self.input
        end
    else
        ---if testing 
        return self.testInput
    end
end

function TagBiLSTM:backward()
    self.gradParams:zero()
    local gradOutputTensor = self.gradOutputPtr
    local backwardInput = self:getBackwardInput()  --since backward only happen in training
    self.gradOutput = gradOutputTensor
    if self.gpuid >= 0 then
        self.gradOutput = self.gradOutput:cuda()
    end
    self.net:training()
    self.net:backward(backwardInput, self.gradOutput)

    if self.gpuid >= 0 then
        self.gradParamsDouble:copy(self.gradParams:double())
    end
end

function TagBiLSTM:getBackwardInput()
    if self.batchInputIds ~= nil then
        return self.batchInput
    else
        return self.input
    end
    --no need the test input, because backward always happen in training
end


