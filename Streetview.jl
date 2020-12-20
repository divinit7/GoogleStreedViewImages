using Flux, Metalhead, Statistics
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, flatten
using Metalhead: trainimgs
using Parameters: @with_kw
using Images: channelview
using Statistics: mean
using Base.Iterators: partition
using Images, DataFrames
using CUDAapi

if has_cuda()
    @info "CUDA is on"
    import CUDA
    CUDA.allowscalar(false)
end

@with_kw mutable struct Args
    batchsize::Int = 128
    throttle::Int = 10
    lr::Float64 = 3e-4
    epochs::Int = 50
    splitr_::Float64 = 0.1
end

#typeData could be either "train" or "test.
#labelsInfo should contain the IDs of each image to be read
#The images in the trainResized and testResized data files
#are 20x20 pixels, so imageSize is set to 400.
#path should be set to the location of the data files.

function read_data(typeData, labelsInfo, imageSize, path)
 #Intialize x matrix
 x = zeros(size(labelsInfo, 1), imageSize)

 for (index, idImage) in enumerate(labelsInfo["ID"])
  #Read image file
  nameFile = "$(path)/$(typeData)Resized/$(idImage).Bmp"
  img = imread(nameFile)

  #Convert img to float values
  temp = float32sc(img)

  #Convert color images to gray images
  #by taking the average of the color scales.
  if ndims(temp) == 3
   temp = mean(temp.data, 1)
  end

  #Transform image matrix to a vector and store
  #it in data matrix
  x[index, :] = reshape(temp, 1, imageSize)
 end
 return x
end

imageSize = 400 # 20 x 20 pixel

#Set location of data files, folders
path = "C:/Users/chauh/Documents/Github/GoogleStreedViewImages"

#Read information about training data , IDs.
labelsInfoTrain = readtable("$(path)/trainLabels.csv")

#Read training matrix
xTrain = read_data("train", labelsInfoTrain, imageSize, path)

#Read information about test data ( IDs ).
labelsInfoTest = readtable("$(path)/sampleSubmission.csv")

#Read test matrix
xTest = read_data("test", labelsInfoTest, imageSize, path)
