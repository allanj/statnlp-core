require 'nn'
require 'optim'

include 'nn-crf-interface/neural_server/AbstractNetwork.lua'
include 'nn-crf-interface/neural_server/MultiLayerPerceptron.lua'
include 'nn-crf-interface/neural_server/OneHot.lua'
include 'nn-crf-interface/neural_server/Utils.lua'

local SEED = 1337
torch.manualSeed(SEED)

-- GPU setup
local gpuid = -1
if gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. gpuid .. '...')
        cutorch.setDevice(gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(SEED)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        gpuid = -1 -- overwrite user setting
    end
else
    print("CPU mode")
end

local net
function initialize(javadata, ...)
    local timer = torch.Timer()

    -- re-seed
    torch.manualSeed(SEED)
    if gpuid >= 0 then cutorch.manualSeed(SEED) end

    local networkClass = javadata:get("class")
    if networkClass == "MultiLayerPerceptron" then
        net = MultiLayerPerceptron(false, gpuid)
    else
        error("Unsupported network class " .. networkClass)
    end
    local outputAndGradOutputPtr = {... }
    local paramsPtr, gradParamsPtr = net:initialize(javadata, unpack(outputAndGradOutputPtr))
    local time = timer:time().real
    print(string.format("Init took %.4fs", time))
    if paramsPtr ~= nil and gradParamsPtr ~= nil then
        return paramsPtr, gradParamsPtr
    end
end

function forward(training)
    local timer = torch.Timer()
    net:forward(training)
    local time = timer:time().real
    print(string.format("Forward took %.4fs", time))
end

function backward()
    local timer = torch.Timer()
    net:backward()
    local time = timer:time().real
    print(string.format("Backward took %.4fs", time))
end

function save_model(prefix)
    local timer = torch.Timer()
    net:save_model(prefix)
    local time = timer:time().real
    print(string.format("Saving model took %.4fs", time))
end

function load_model(prefix)
    local timer = torch.Timer()
    net:load_model(prefix)
    local time = timer:time().real
    print(string.format("Loading model took %.4fs", time))
end
