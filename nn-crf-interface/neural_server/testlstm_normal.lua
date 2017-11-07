require 'nn'
require 'rnn'
include 'PyTorchLSTM.lua'

local lstm = PyTorchLSTM(5, 4)
torch.manualSeed(1)

lstm.inputGate:get(2):get(1):get(2).weight:uniform(-1, 1)
lstm.inputGate:get(2):get(1):get(2).bias:uniform(-1, 1)
lstm.inputGate:get(2):get(2):get(2).weight:uniform(-1, 1)
lstm.inputGate:get(2):get(2):get(2).bias:uniform(-1, 1)



lstm.forgetGate:get(2):get(1):get(2).weight:uniform(-1, 1)
lstm.forgetGate:get(2):get(1):get(2).bias:uniform(-1, 1)
lstm.forgetGate:get(2):get(2):get(2).weight:uniform(-1, 1)
lstm.forgetGate:get(2):get(2):get(2).bias:uniform(-1, 1)


lstm.hiddenLayer:get(2):get(1):get(2).weight:uniform(-1, 1)
lstm.hiddenLayer:get(2):get(1):get(2).bias:uniform(-1, 1)
lstm.hiddenLayer:get(2):get(2):get(2).weight:uniform(-1, 1)
lstm.hiddenLayer:get(2):get(2):get(2).bias:uniform(-1, 1)


lstm.outputGate:get(2):get(1):get(2).weight:uniform(-1, 1)
lstm.outputGate:get(2):get(1):get(2).bias:uniform(-1, 1)
lstm.outputGate:get(2):get(2):get(2).weight:uniform(-1, 1)
lstm.outputGate:get(2):get(2):get(2).bias:uniform(-1, 1)



-- print("inputgtea")
-- print(lstm.inputGate:get(2):get(1):get(2).weight)
-- print(lstm.inputGate:get(2):get(1):get(2).bias)
-- print(lstm.inputGate:get(2):get(2):get(2).weight)
-- print(lstm.inputGate:get(2):get(2):get(2).bias)
-- print("separator")


-- print("forgetgate")
-- print(lstm.forgetGate:get(2):get(1):get(2).weight)
-- print(lstm.forgetGate:get(2):get(1):get(2).bias)
-- print(lstm.forgetGate:get(2):get(2):get(2).weight)
-- print(lstm.forgetGate:get(2):get(2):get(2).bias)


-- print("hiddenLayer")
-- print(lstm.hiddenLayer:get(2):get(1):get(2).weight)
-- print(lstm.hiddenLayer:get(2):get(1):get(2).bias)
-- print(lstm.hiddenLayer:get(2):get(2):get(2).weight)
-- print(lstm.hiddenLayer:get(2):get(2):get(2).bias)


print("outputGate")
print(lstm.outputGate:get(2):get(1):get(2).weight)
print(lstm.outputGate:get(2):get(1):get(2).bias)
print(lstm.outputGate:get(2):get(2):get(2).weight)
print(lstm.outputGate:get(2):get(2):get(2).bias)
local x = torch.Tensor(5):uniform(-1,1)
print(x)
lstm:clearState()
print("forget toutput")
local outputGateOutput = lstm.outputGate:forward({x, torch.Tensor(4):zero()})
print(outputGateOutput)
local y = lstm:forward(x)
print(y)