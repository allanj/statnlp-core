require 'nn'
mlp = torch.load('ner_embeddings.model')

path = '/home/raymondhs/Workspace/semantic/sp+nn/statnlp-core/model/'
featFile = torch.DiskFile(path .. 'neural.txt')
featSize = 213324
inputLine = torch.IntStorage(10)
numLabel = 9
for i=1,featSize do
   local input = torch.Tensor(10)
   featFile:readInt(inputLine)
   for j=1,10 do 
      input[j] = inputLine[j]
   end
   local newInput = nn.SplitTable(1):forward(nn.Reshape(2,5):forward(input))
   output = mlp:forward(newInput)
   res = ""
   for j=1,10 do
      res = res .. " " .. input[j]
   end
   for j=1,numLabel do
      res = res .. " " .. output[j]
   end
   print(res)
end
