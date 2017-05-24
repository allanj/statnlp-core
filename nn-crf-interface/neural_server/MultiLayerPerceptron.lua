local MultiLayerPerceptron, parent = torch.class('MultiLayerPerceptron', 'AbstractNetwork')

function MultiLayerPerceptron:__init(doOptimization, gpuid)
    parent.__init(self, doOptimization)
    self.data = {}
    self.gpuid = gpuid
end

function MultiLayerPerceptron:initialize(javadata, ...)
    local gpuid = self.gpuid

    local outputAndGradOutputPtr = {... }

    self.outputPtr = torch.pushudata(outputAndGradOutputPtr[1], "torch.DoubleTensor")
    self.gradOutputPtr = torch.pushudata(outputAndGradOutputPtr[2], "torch.DoubleTensor")

    -- for i=1,#outputAndGradOutputPtr do
    --     local ptr = outputAndGradOutputPtr[i]
    --     if i <= #outputAndGradOutputPtr/2 then
    --         table.insert(self.outputPtr, torch.pushudata(ptr, "torch.DoubleTensor"))
    --     else    
    --         table.insert(self.gradOutputPtr, torch.pushudata(ptr, "torch.DoubleTensor"))
    --     end
    -- end

    -- numInputList, inputDimList, embSizeList, outputDim,
    -- numLayer, hiddenSize, activation, dropout
    -- vocab

    local data = self.data
    data.vocab = listToTable2D(javadata:get("vocab"))
    data.numInputList = listToTable(javadata:get("numInputList"))
    data.embedding = listToTable(javadata:get("embedding"))
    data.embSizeList = listToTable(javadata:get("embSizeList"))
    data.fixEmbedding = javadata:get("fixEmbedding")
    data.wordList = listToTable(javadata:get("wordList"))
    data.inputDimList = listToTable(javadata:get("inputDimList"))
    data.numLayer = javadata:get("numLayer")
    data.hiddenSize = javadata:get("hiddenSize")
    data.activation = javadata:get("activation")
    data.dropout = javadata:get("dropout")
    data.optimizer = javadata:get("optimizer")

    -- what to forward
    self.x = self:prepare_input()
    self.fixEmbedding = data.fixEmbedding
    self.wordList = data.wordList
    self.word2idx = {}
    local wordList = self.wordList
    local word2idx = self.word2idx
    for i=1,#wordList do
        word2idx[wordList[i]] = i
    end
    self.numInput = self.x[1]:size(1)

    -- input layer
    local pt = nn.ParallelTable()
    local totalInput = 0
    local totalDim = 0
    for i=1,#data.inputDimList do
        local inputDim = data.inputDimList[i]
        local lt
        if data.embSizeList[i] == 0 then
            lt = OneHot(inputDim)
            totalDim = totalDim + data.numInputList[i] * inputDim
        else
            if data.embedding ~= nil then
                if data.embedding[i] == 'senna' then
                    lt = loadSenna()
                elseif data.embedding[i] == 'glove' then
                    lt = loadGlove(data.wordList, data.embSizeList[i])
                elseif data.embedding[i] == 'polyglot' then
                    lt = loadPolyglot(data.wordList, data.lang)
                elseif data.embedding[i] == 'bansal' then
                    lt = loadBansal(data.wordList)
                else -- unknown/no embedding, defaults to random init
                    lt = nn.LookupTable(inputDim, data.embSizeList[i])
                end
            else
                lt = nn.LookupTable(inputDim, data.embSizeList[i])
            end
            totalDim = totalDim + data.numInputList[i] * data.embSizeList[i]
        end
        if data.fixEmbedding then
            lt.accGradParameters = function() end
        end
        pt:add(nn.Sequential():add(lt):add(nn.View(self.numInput,-1)))
        totalInput = totalInput + data.numInputList[i]
    end

    local jt = nn.JoinTable(2)

    self.net = nn.Sequential()
    local mlp = self.net
    if data.fixEmbedding then
        self.inputLayer = nn.Sequential()
        self.inputLayer:add(pt)
        self.inputLayer:add(jt)
    else
        mlp:add(pt)
        mlp:add(jt)
    end
   
    -- self.outputDimList = data.outputDimList
    -- local ct = nn.ConcatTable()
    -- self.numNetworks = #outputDimList
    -- for n=1,numNetworks do
        -- local middleLayers = nn.Sequential()
        local middleLayers = mlp
        -- hidden layer
        for i=1,data.numLayer do
            if data.dropout ~= nil and data.dropout > 0 then
                middleLayers:add(nn.Dropout(data.dropout))
            end

            local ll
            if i == 1 then
                ll = nn.Linear(totalDim, data.hiddenSize)
            else
                ll = nn.Linear(data.hiddenSize, data.hiddenSize)
            end
            middleLayers:add(ll)

            local act
            if data.activation == nil or data.activation == "relu" then
                act = nn.ReLU()
            elseif data.activation == "tanh" then
                act = nn.Tanh()
            elseif data.activation == "hardtanh" then
                act = nn.HardTanh()
            elseif data.activation == "identity" then
                -- do nothing
            else
                error("activation " .. activation .. " not supported")
            end
            if act ~= nil then
                middleLayers:add(act)
            end
        end

        -- if data.dropout ~= nil and data.dropout > 0 then
        --     middleLayers:add(nn.Dropout(data.dropout))
        -- end

        -- output layer (passed to CRF)
        -- local outputDim = data.outputDimList[n]
        -- local lastInputDim
        -- if data.numLayer == 0 then
        --     lastInputDim = totalDim
        -- else
        --     lastInputDim = data.hiddenSize
        -- end
        -- if data.useOutputBias then
        --     middleLayers:add(nn.Linear(lastInputDim, outputDim))
        -- else
        --     middleLayers:add(nn.Linear(lastInputDim, outputDim):noBias())
        -- end
        -- ct:add(middleLayers)
    -- end
    -- mlp:add(ct)

    if gpuid >= 0 then
        if data.fixEmbedding then inputLayer:cuda() end
        mlp:cuda()
    end

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

    self.params, self.gradParams = mlp:getParameters()

    if data.fixEmbedding then print(self.inputLayer) end
    print(mlp)

    if not doOptimization then -- no return array if optim is done here
        self.params:retain()
        self.paramsPtr = torch.pointer(self.params)
        self.gradParams:retain()
        self.gradParamsPtr = torch.pointer(self.gradParams)
        return self.paramsPtr, self.gradParamsPtr    
    end
end

function MultiLayerPerceptron:prepare_input()
    local gpuid = self.gpuid
    local data = self.data
    local vocab = data.vocab
    local numInputList = data.numInputList
    local embSizeList = data.embSizeList
    local result = {}
    local startIdx = 0
    for i=1,#numInputList do
        table.insert(result, torch.Tensor(#vocab, numInputList[i]))
        for j=1,#vocab do
            for k=1,numInputList[i] do
                result[i][j][k] = vocab[j][startIdx+k]
            end
        end
        startIdx = startIdx + numInputList[i]
        if gpuid >= 0 then result[i] = result[i]:cuda() end
    end
    return result
end

function MultiLayerPerceptron:forward(isTraining)
    local mlp = self.net
    if isTraining then
        mlp:training()
    else
        mlp:evaluate()
    end
    local x = self.x
    local input_x = x
    if self.fixEmbedding then
        input_x = self.inputLayer:forward(x)
    end
    local output = mlp:forward(input_x)
    self.outputPtr:copy(output)
    -- for i=1,#output do
    --     self.outputPtr[i]:copy(output[i])
    -- end
end

function MultiLayerPerceptron:backward()
    self.gradParams:zero()
    local x = self.x
    local input_x = x
    if self.fixEmbedding then
        input_x = self.inputLayer:forward(x)
    end
    self.net:backward(input_x, self.gradOutputPtr)
    if self.doOptimization then
        self.optimizer(self.feval, self.params, self.optimState)
    end
end

function MultiLayerPerceptron:save_model(prefix)
    local obj = {}
    obj["mlp"] = self.net
    obj["word2idx"] = self.word2idx
    torch.save("model/" .. prefix .. ".t7",obj)
end

function MultiLayerPerceptron:load_model(path)
    local saved_obj = torch.load("model/" .. prefix .. ".t7")
    local saved_mlp = saved_obj.mlp
    local saved_word2idx = saved_obj.word2idx
    local saved_lt = saved_mlp:get(1):get(1):get(1)
    local mlp = self.net
    local orig_lt = mlp:get(1):get(1):get(1)
    local wordList = self.wordList
    for i=1,#wordList do
        saved_word_idx = saved_word2idx[wordList[i]]
        if saved_word_idx ~= nil then
            orig_lt.weight[i]:copy(saved_lt.weight[saved_word_idx])
        end
    end
    saved_lt.weight = orig_lt.weight
    mlp = saved_mlp
    mlp:get(1):get(1):get(2):resetSize(self.numInput,-1)
end
