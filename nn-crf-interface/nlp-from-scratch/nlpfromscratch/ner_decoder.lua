require 'torch'
require 'nn'
mlp = torch.load('ner_embedding.model')
lsm = nn.LogSoftMax()
print(mlp)
mlp2 = nn.Sequential()
mlp2:add(mlp)
mlp2:add(lsm)

-- Testing

outputFile = torch.DiskFile('ner_results_embedding.txt', 'w')

testFile = torch.DiskFile('ner_testing.txt', 'r')
inputLine = torch.IntStorage(10)

testSize = 46435
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
   for k=1,17 do
      if output[k] > outputValue then
	 outputLabel = k;
	 outputValue = output[k];
      end
   end

   outputFile:writeInt(outputLabel);
      
end

testFile:close()
outputFile:close()
