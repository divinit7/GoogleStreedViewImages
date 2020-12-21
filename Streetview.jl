using Flux, Metalhead, Statistics
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, flatten
using Metalhead: trainimgs
using Parameters: @with_kw
using Images: channelview
using Statistics: mean
using Base.Iterators: partition
using Images, DataFrames
using CUDA
using CSV

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

 for (index, idImage) in enumerate(labelsInfo[!,"ID"])
  #Read image file
  nameFile = "$(path)/$(typeData)Resized/$(idImage).Bmp"
  img = load(nameFile)

  #Convert img to float values
  temp = float64.(img)

  #Convert color images to gray images
  #by taking the average of the color scales.
  if ndims(temp) == 3
   temp = mean.(temp.data, 1)
  end

  #Transform image matrix to a vector and store
  #it in data matrix
  temp = Gray.(temp)
  x[index, :] = convert(Array{Float64}, temp)
 end
 return x
end

imageSize = 400 # 20 x 20 pixel

#Set location of data files, folders
path = "C:/Users/chauh/Documents/Github/GoogleStreedViewImages"

#Read information about training data , IDs.
labelsInfoTrain = CSV.read("$(path)/trainLabels.csv", DataFrame)

#Read training matrix
xTrain = read_data("train", labelsInfoTrain, imageSize, path)

#Read information about test data ( IDs ).
labelsInfoTest = CSV.read("$(path)/sampleSubmission.csv", DataFrame)

#Read test matrix
xTest = read_data("test", labelsInfoTest, imageSize, path)

function vgg16()
    return Chain(
    Conv((3, 3), 3 => 64, relu, pad = (1, 1), stride = 1),
    BatchNorm(64),
    Conv((3, 3), 64 => 64, relu, pad = (1, 1), stride = 1),
    BatchNorm(64),
    MaxPool((2, 2)),
    Conv((3, 3), 64 => 128, relu, pad = (1, 1), stride = 1),
    BatchNorm(128),
    Conv((3, 3), 128 => 128, relu, pad = (1, 1), stride = 1),
    BatchNorm(128),
    MaxPool((2, 2)),
    Conv((3, 3), 128 => 256, relu, pad = (1, 1), stride = 1),
    BatchNorm(256),
    Conv((3, 3), 256 => 256, relu, pad = (1, 1), stride = 1),
    BatchNorm(256),
    MaxPool((2, 2)),
    Conv((3, 3),256 => 512, relu, pad = (1, 1), stride = 1),
    BatchNorm(512),
    Conv((3, 3), 512 => 512, relu, pad = (1, 1), stride = 1),
    BatchNorm(512),
    Conv((3, 3), 512 => 512, relu, pad = (1, 1), stride = 1),
    BatchNorm(512),
    MaxPool((2, 2)),
    Conv((3, 3), 512 => 512, relu, pad = (1, 1), stride = 1),
    BatchNorm(512),
    Conv((3, 3), 512 => 512, relu, pad = (1, 1), stride = 1),
    BatchNorm(512),
    MaxPool((2, 2)),
    Conv((3, 3), 512 => 512, relu, pad = (1, 1), stride = 1),
    BatchNorm(512),
    Conv((3, 3), 512 => 512, relu, pad = (1, 1), stride = 1),
    BatchNorm(512),
    MaxPool((2, 2)),
    flatten,
    Dense(512, 4096, relu),
    Dropout(0.4),
    Dense(4096, 10)) |> gpu
end

accuracy(x, y, m) = mean(onecold(cpu(m(x)), 1:10) .== onecold(cpu(y), 1:10))

function get_processed_data(args)
    #Fetching the train and validation and getting them into proper shape
    X = trainimgs(xTrain)
    imgs = [getarray(X[i].img) for i in 1:40000]

    # onehot encode the labels
    labels = onehotbatch(X[i].ground_truth class for i in )

function train(; kws...)
    # Initialize the hyperparameters
    args = Args(; kws...)

    # Load the train, validation data
    train, val = get
