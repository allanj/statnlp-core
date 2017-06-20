local ContinuousFeature, parent = torch.class('ContinuousFeature', 'AbstractNeuralNetwork')


function ContinuousFeature:__init(doOptimization, gpuid)
    parent.__init(self, doOptimization)
    self.data = {}
    self.gpuid = gpuid
end

function ContinuousFeature:initialize(javadata, ...)
    local gpuid = self.gpuid

    self.data = {}
    local data = self.data
    self.numLabels = javadata:get("numLabels")
    
    local isTraining = javadata:get("isTraining")
    self.isTraining = isTraining
    local outputAndGradOutputPtr = {...}
    if isTraining then
        self.outputPtr = torch.pushudata(outputAndGradOutputPtr[1], "torch.DoubleTensor")
        self.gradOutputPtr = torch.pushudata(outputAndGradOutputPtr[2], "torch.DoubleTensor")
    end
    self.x = torch.pushudata(outputAndGradOutputPtr[3], "torch.DoubleTensor")
    self.x:retain()
    self.output = torch.Tensor()
    self.gradOutput = {}
    if isTraining then
        self:createNetwork()
        print(self.net)
        self.params, self.gradParams = self.net:getParameters()
        print("Number of parameters: " .. self.params:nElement())
        self.params:retain()
        self.paramsPtr = torch.pointer(self.params)
        self.gradParams:retain()
        self.gradParamsPtr = torch.pointer(self.gradParams)
        return self.paramsPtr, self.gradParamsPtr
    end
end


function ContinuousFeature:createNetwork()
    local fwd = nn.Sequential():add(nn.Linear(1, self.numLabels):noBias())
    self.net = fwd
end

function ContinuousFeature:forward(isTraining)
    --local output_table = self.net:forward(self.x)
    self.output = self.net:forward(self.x)
    --- this is to converting the table into tensor.
    --self.output = torch.cat(self.output, output_table, 1)
    if not self.outputPtr:isSameSizeAs(self.output) then
        self.outputPtr:resizeAs(self.output)
    end
    self.outputPtr:copy(self.output)
end

function ContinuousFeature:backward()
    self.gradParams:zero()
    self.gradOutput = self.gradOutputPtr
    --torch.split(self.gradOutput, self.gradOutputPtr, self.x:size(2), 1)
    self.net:backward(self.x, self.gradOutput)
end


function ContinuousFeature:save_model(path)
    torch.save(path,self.net)
end

function ContinuousFeature:load_model(path)
    self.net = torch.load(path)
end