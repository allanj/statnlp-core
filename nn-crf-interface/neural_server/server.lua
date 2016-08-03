require "nn"
require "OneHot"

cmd = torch.CmdLine()
cmd:text()
cmd:text('Neural Network Server')
cmd:text()
cmd:option('-port', 5556, 'port number')
cmd:option('-gpuid', -1, 'which GPU to use (>= 0, -1 = CPU)')
cmd:text()
opt = cmd:parse(arg)
portNumber = opt.port
print("listening on port " .. opt.port)

local json = require ("dkjson")
local zmq = require "lzmq"
local context = zmq.init(1)

-- GPU setup
gpuid = opt.gpuid
if gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. gpuid .. '...')
        cutorch.setDevice(gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(1337)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        gpuid = -1 -- overwrite user setting
    end
else
    print("CPU mode")
end

local socket = context:socket(zmq.REP)
socket:bind("tcp://*:" .. portNumber)

local ret -- return value to client
local mlp -- our neural net
local params, gradParams -- mlp's params
local x -- input to nn, which is fixed
local inputVocabSize
local outputDim

function init_MLP(inputDim, outputDim)
    local mlp = nn.Sequential()
                :add(OneHot(inputDim))
                :add(nn.Linear(inputDim, outputDim):noBias()) -- no bias
    local x = torch.Tensor(inputDim)
    local i = 0
    x:apply(function() i = i + 1 return i end)
    if gpuid >= 0 then
        mlp:cuda()
        x:cuda()
    end
    params, gradParams = mlp:getParameters()
    return mlp, x
end

function fwd_MLP(mlp, x, newParams)
    params:copy(newParams)
    return mlp:forward(x)
end

function bwd_MLP(mlp, x, gradOutput)
    gradParams:zero()
    mlp:backward(x, gradOutput)
end

function serialize(data)
    local timer = torch.Timer()
    local ret = data:view(-1):totable()
    local time = timer:time().real
    print(string.format("Serializing took %.4fs", time))
    return ret
end

function deserialize(data, row, col)
    local timer = torch.Timer()
    local ret
    if row == 1 then
        ret = torch.Tensor(data)
    else
        ret = torch.Tensor(data):view(row, col)
    end
    if gpuid >= 0 then ret = ret:cuda() end
    local time = timer:time().real
    print(string.format("Deserializing took %.4fs", time))
    return ret
end

while true do
    --  Wait for next request from client
    local request = socket:recv()
    -- print("Received Hello [" .. request .. "]")
    -- print(request)
    if request ~= nil then
        request = json.decode(request, 1, nil)
        if request.cmd == "init" then
            timer = torch.Timer()
            inputVocabSize = request.inputVocabSize
            print("inputVocabSize",request.inputVocabSize)
            outputDim = request.outputDim
            print("outputDim",request.outputDim)
            mlp, x = init_MLP(request.inputVocabSize, request.outputDim)
            print(mlp)
            ret = {params:nElement()}
            time = timer:time().real
            print(string.format("Init took %.4fs", time))
        elseif request.cmd == "fwd" then
            local timer = torch.Timer()
            local newParams = deserialize(request.weights, 1, -1)
            local fwd_out = fwd_MLP(mlp, x, newParams)
            ret = serialize(fwd_out)
            time = timer:time().real
            print(string.format("Forward took %.4fs", time))
        elseif request.cmd == "bwd" then
            local timer = torch.Timer()
            local gradOut = deserialize(request.grad, inputVocabSize, outputDim)
            bwd_MLP(mlp, x, gradOut)
            ret = serialize(gradParams)
            time = timer:time().real
            print(string.format("Backward took %.4fs", time))
        end
        ret = json.encode (ret, { indent = true })
        socket:send(ret)
    end
end
--  We never get here but if we did, this would be how we end
socket:close()
context:term()
