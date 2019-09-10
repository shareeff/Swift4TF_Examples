import Foundation
import Path
import TensorFlow


let learningRate: Float = 0.0002
let batchSize = 16
let testBachSize = 1
let imgSize = 256
let cycLambda: Float = 10
let identityLambda: Float = 5
let epochCount = 1000
//let outputFolder = Path.cwd/"Output"
let outputFolder = "./output/"
let testImageGridSize = 2
let path = Path.cwd/"Data"
let fname = "horse2zebra"
downloadAndExtractFile(path, fname)

let trainAPath = path/fname/"trainA"
let trainBPath = path/fname/"trainB"
let testAPath = path/fname/"testA"
let testBPath = path/fname/"testB"

let datasetA = GANDATASET(trainAPath, testAPath)
let datasetB = GANDATASET(trainBPath, testBPath)

var discA = Discriminator()
var discB = Discriminator()
var genA2B = Generator()
var genB2A = Generator()

let optDiscA = Adam(for: discA, learningRate: learningRate, beta1: 0.5)
let optDiscB = Adam(for: discB, learningRate: learningRate, beta1: 0.5)
let optGenA2B = Adam(for: genA2B, learningRate: learningRate, beta1: 0.5)
let optGenB2A = Adam(for: genB2A, learningRate: learningRate, beta1: 0.5)

let testABatches = datasetA.testDataset.batched(testBachSize)
let testBBatches = datasetB.testDataset.batched(testBachSize)

var ittTestA = testABatches.makeIterator()
var ittTestB = testBBatches.makeIterator()

func saveImageGrid(_ testImage: Tensor<Float>, name: String) throws {
    assert(testImage.shape.count == 4, "Image tensor should have 4 dim")
    let paddedImage = testImage.padded(forSizes: [(0, 0), (2, 2), (2, 2), (0, 0)], with: 1)
    let imageHeight = paddedImage.shape[1]
    let imageWidth = paddedImage.shape[2]
    let imageChannel = paddedImage.shape[3]
    var gridImage = paddedImage.reshaped(
        to: [
            testImageGridSize, testImageGridSize,
            imageHeight, imageWidth, imageChannel
        ])
    // Add padding.
    //gridImage = gridImage.padded(forSizes: [(0, 0), (0, 0), (1, 1), (1, 1)], with: 1)
    // Transpose to create single image.
    gridImage = gridImage.transposed(withPermutations: [0, 2, 1, 3, 4])
    gridImage = gridImage.reshaped(
        to: [
            imageHeight * testImageGridSize,
            imageWidth * testImageGridSize,
            imageChannel
        ])
    // Convert [-1, 1] range to [0, 1] range.
    gridImage = (gridImage + 1) / 2

    try saveImage(
        gridImage, size: (gridImage.shape[0], gridImage.shape[1], gridImage.shape[2]), directory: outputFolder,
        name: name)
}


print("Start training...")


// Start training loop.
for epoch in 0...epochCount {
    // Start training phase.
    Context.local.learningPhase = .training
    let trainingAShuffled = datasetA.trainingDataset.shuffled(
        sampleCount: 1000, randomSeed: Int64(epoch))
    let trainingBShuffled = datasetB.trainingDataset.shuffled(
        sampleCount: 1000, randomSeed: Int64(epoch))

    var lossGenA2B: Tensor<Float> = Tensor<Float>(zeros: [1])
    var lossGenB2A: Tensor<Float> = Tensor<Float>(zeros: [1])

    for (batchA, batchB) in zip(trainingAShuffled.batched(batchSize), trainingBShuffled.batched(batchSize)){
        let imagesA = batchA.data
        let imagesB = batchB.data

        //assert(imagesA.shape == imagesB.shape, "imageA (\(imagesA.shape)) and imageB (\(imagesB.shape)) batchSize missmatch")

        if imagesA.shape != imagesB.shape {
            break
        }

        let genA2BOutput = genA2B(imagesA)
        let genB2AOutput = genB2A(imagesB)

        let reconstructedA = genB2A(genA2BOutput)
        let reconstructedB = genA2B(genB2AOutput)

        let discAFakeOutput = discA(genB2AOutput)
        let discBFakeOutput = discB(genA2BOutput)


        let (lossGA2B, 𝛁genA2B) = genA2B.valueWithGradient { genA2B -> Tensor<Float> in
            
            let loss = generatorLoss(fakeLogits: discBFakeOutput) + 
                cycleConsistencyLoss(dataA: imagesA,  dataB: imagesB, reconstructedDataA: reconstructedA, 
                    reconstructedDataB: reconstructedB, cycLambda: cycLambda) + 
                    identityLoss(dataA: imagesA, dataB: imagesB,
                        genA2BOutput: genA2BOutput, genB2AOutput: genB2AOutput, identityLambda: identityLambda)

            return loss
        }
        
        optGenA2B.update(&genA2B, along: 𝛁genA2B)
        lossGenA2B = lossGenA2B.concatenated(with: lossGA2B.expandingShape(at:0))

        let (lossGB2A, 𝛁genB2A) = genB2A.valueWithGradient { genB2A -> Tensor<Float> in

            let loss = generatorLoss(fakeLogits: discAFakeOutput) + 
                cycleConsistencyLoss(dataA: imagesA,  dataB: imagesB, reconstructedDataA: reconstructedA, 
                    reconstructedDataB: reconstructedB, cycLambda: cycLambda) + 
                    identityLoss(dataA: imagesA, dataB: imagesB,
                        genA2BOutput: genA2BOutput, genB2AOutput: genB2AOutput, identityLambda: identityLambda)

            return loss

        }
        
        optGenB2A.update(&genB2A, along: 𝛁genB2A)
        lossGenB2A = lossGenB2A.concatenated(with: lossGB2A.expandingShape(at:0))

        let genA2BFakeOutput = genA2B(imagesA)
        let genB2AFakeOutput = genB2A(imagesB)
       

        let 𝛁discA = discA.gradient { discA -> Tensor<Float> in 

            let discARealOutput = discA(imagesA)
            let discAFakeOutput = discA(genB2AFakeOutput)
            let loss = discriminatorLoss(realLogits: discARealOutput, fakeLogits: discAFakeOutput)
            return loss
        }

        optDiscA.update(&discA, along: 𝛁discA)

        let 𝛁discB = discB.gradient { discB -> Tensor<Float> in

            let discBRealOutput = discB(imagesB)
            let discBFakeOutput = discB(genA2BFakeOutput)
            let loss = discriminatorLoss(realLogits: discBRealOutput, fakeLogits: discBFakeOutput)
            return loss
        }

        optDiscB.update(&discB, along: 𝛁discB)

    }

    // Start inference phase.
    Context.local.learningPhase = .inference

   

    if Int(epoch) % 10 == 0 {
        let testA = ittTestA.next()! 

        let testB = ittTestB.next()! 

        let testImagesA = testA.data
        let testImagesB = testB.data

        let genTestA2BOutput = genA2B(testImagesA)
        let genTestB2AOutput = genB2A(testImagesB)
        var imageToSave = testImagesA[0].expandingShape(at:0)
        imageToSave = imageToSave.concatenated(with: testImagesB[0].expandingShape(at:0))
        imageToSave = imageToSave.concatenated(with: genTestA2BOutput[0].expandingShape(at:0))
        imageToSave = imageToSave.concatenated(with: genTestB2AOutput[0].expandingShape(at:0)) 
        do {
            try saveImageGrid(imageToSave, name: "\(fname)_epoch_\(epoch)")
        } catch {
            print("Could not save image grid with error: \(error)")
        }
    }

    print("[Epoch: \(epoch)] Loss-GenA: \(lossGenA2B.mean()) Loss-GenB: \(lossGenB2A.mean())")

    
}

