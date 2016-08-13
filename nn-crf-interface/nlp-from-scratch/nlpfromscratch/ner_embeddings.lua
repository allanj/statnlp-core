require 'torch'
require 'nn'

numLabel = 9

-- initial vector creation assuming an input vector with 10 values, 5 of which are word ids and 5 are caps ids

ltw = nn.LookupTable(130000, 50)

-- initialize lookup table with embeddings
--path = '/Users/nlp/Documents/workspace/semantic/statnlp-core/nn-crf-interface/nlp-from-scratch/senna-torch/'
path = '../senna-torch/'
embeddingsFile = torch.DiskFile(path .. 'senna/embeddings/embeddings.txt');
embedding = torch.DoubleStorage(50)

embeddingsFile:readDouble(embedding);
for i=2,130000 do 
   embeddingsFile:readDouble(embedding);
   local emb = torch.Tensor(50)
   for j=1,50 do 
      emb[j] = embedding[j]
   end
   ltw.weight[i-1] = emb;
end

ltc = nn.LookupTable(5, 5)
pt = nn.ParallelTable()
pt:add(ltw)
pt:add(ltc)
jt = nn.JoinTable(2)
rs2 = nn.Reshape(275)

mlp = nn.Sequential()
mlp:add(pt)
mlp:add(jt)
mlp:add(rs2)

-- the NN layers
ll1 = nn.Linear(275, 300)
hth = nn.HardTanh()
ll2 = nn.Linear(300, numLabel)
lsm = nn.LogSoftMax()

mlp:add(ll1)
mlp:add(hth)
mlp:add(ll2)

-- let's separate
mlp2 = nn.Sequential()
mlp2:add(mlp)
mlp2:add(lsm)

--trainSize = 172389
--testSize = 44462
trainSize = 203621
testSize = 46435
-- create training data set
inputFile = torch.DiskFile('ner_training.txt', 'r')
inputLine = torch.IntStorage(10)

dataset = {}
function dataset:size() return trainSize end

for i=1,dataset:size() do 
   inputFile:readInt(inputLine)
   local input = torch.Tensor(10)
   for j=1,10 do 
      input[j] = inputLine[j]
   end

   local newInput = nn.SplitTable(1):forward(nn.Reshape(2,5):forward(input))
   local label = inputFile:readInt()

   dataset[i] = {newInput, label}

end

inputFile:close()

criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(mlp2, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)

torch.save("ner_embeddings.model",mlp)


-- Testing

outputFile = torch.DiskFile('ner_results_embedding.txt', 'w')

testFile = torch.DiskFile('ner_testing.txt', 'r')
inputLine = torch.IntStorage(10)

for i=1,testSize do 
   testFile:readInt(inputLine)
   local label = testFile:readInt()
   local input = torch.Tensor(10)
   for j=1,10 do 
      input[j] = inputLine[j]
   end
   local newInput = nn.SplitTable(1):forward(nn.Reshape(2,5):forward(input))
   output = mlp2:forward(newInput)

   local outputLabel = 1;
   local outputValue = -1000;
   for k=1,numLabel do
      if output[k] > outputValue then
	 outputLabel = k;
	 outputValue = output[k];
      end
   end

   outputFile:writeInt(outputLabel);
      
end

testFile:close()
outputFile:close()
