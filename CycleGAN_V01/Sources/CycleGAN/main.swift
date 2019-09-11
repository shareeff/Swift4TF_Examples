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



var discA = Discriminator()
var discB = Discriminator()
var genA2B = Generator()
var genB2A = Generator()

let optDiscA = Adam(for: discA, learningRate: learningRate, beta1: 0.5)
let optDiscB = Adam(for: discB, learningRate: learningRate, beta1: 0.5)
let optGenA2B = Adam(for: genA2B, learningRate: learningRate, beta1: 0.5)
let optGenB2A = Adam(for: genB2A, learningRate: learningRate, beta1: 0.5)

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

func train(minibatchA: Tensor<Float>, minibatchB: Tensor<Float>, step: Int) -> (lossGA2B: Tensor<Float>, lossGB2A: Tensor<Float>, 
                                            lossDiscA: Tensor<Float>, lossDiscB: Tensor<Float>){

    Context.local.learningPhase = .training

    let genA2BOutput = genA2B(minibatchA)
    let genB2AOutput = genB2A(minibatchB)

    let reconstructedA = genB2A(genA2BOutput)
    let reconstructedB = genA2B(genB2AOutput)

    let discAFakeOutput = discA(genB2AOutput)
    let discBFakeOutput = discB(genA2BOutput)


    let (lossGA2B, 𝛁genA2B) = genA2B.valueWithGradient { genA2B -> Tensor<Float> in
            
        let loss = generatorLoss(fakeLogits: discBFakeOutput) + 
                cycleConsistencyLoss(dataA: minibatchA,  dataB: minibatchB, reconstructedDataA: reconstructedA, 
                    reconstructedDataB: reconstructedB, cycLambda: cycLambda) + 
                identityLoss(dataA: minibatchA, dataB: minibatchB,
                    genA2BOutput: genA2BOutput, genB2AOutput: genB2AOutput, identityLambda: identityLambda)

            return loss
        }
        
        optGenA2B.update(&genA2B, along: 𝛁genA2B)
        

        let (lossGB2A, 𝛁genB2A) = genB2A.valueWithGradient { genB2A -> Tensor<Float> in

            let loss = generatorLoss(fakeLogits: discAFakeOutput) + 
                cycleConsistencyLoss(dataA: minibatchA,  dataB: minibatchB, reconstructedDataA: reconstructedA, 
                    reconstructedDataB: reconstructedB, cycLambda: cycLambda) + 
                identityLoss(dataA: minibatchA, dataB: minibatchB,
                    genA2BOutput: genA2BOutput, genB2AOutput: genB2AOutput, identityLambda: identityLambda)

            return loss

        }
        
        optGenB2A.update(&genB2A, along: 𝛁genB2A)
        
        let genA2BFakeOutput = genA2B(minibatchA)
        let genB2AFakeOutput = genB2A(minibatchB)
       

        let (lossDiscA, 𝛁discA) = discA.valueWithGradient { discA -> Tensor<Float> in 

            let discARealOutput = discA(minibatchA)
            let discAFakeOutput = discA(genB2AFakeOutput)
            let loss = discriminatorLoss(realLogits: discARealOutput, fakeLogits: discAFakeOutput)
            return loss
        }

        optDiscA.update(&discA, along: 𝛁discA)

        let (lossDiscB, 𝛁discB) = discB.valueWithGradient { discB -> Tensor<Float> in

            let discBRealOutput = discB(minibatchB)
            let discBFakeOutput = discB(genA2BFakeOutput)
            let loss = discriminatorLoss(realLogits: discBRealOutput, fakeLogits: discBFakeOutput)
            return loss
        }

        optDiscB.update(&discB, along: 𝛁discB)

    

    return (lossGA2B, lossGB2A, lossDiscA, lossDiscB)


}


let trainImageLoaderA = try ImageLoader(imageDirectory: URL(fileURLWithPath: trainAPath.string))
let trainImageLoaderB = try ImageLoader(imageDirectory: URL(fileURLWithPath: trainBPath.string))
let testImageLoaderA = try ImageLoader(imageDirectory: URL(fileURLWithPath: testAPath.string))
let testImageLoaderB = try ImageLoader(imageDirectory: URL(fileURLWithPath: testBPath.string))

print("Start training...")


// Start training loop.
for step in 1... {

    let minibatchA = trainImageLoaderA.minibatch(size: batchSize, imageSize: (imgSize, imgSize))
    let minibatchB = trainImageLoaderB.minibatch(size: batchSize, imageSize: (imgSize, imgSize))

    let (lossGA2B, lossGB2A, lossDiscA, lossDiscB) = train(minibatchA: minibatchA, minibatchB: minibatchB, step: step)

    print("[Setp: \(step)] Loss-GenA: \(lossGA2B) Loss-GenB: \(lossGB2A) Loss-DiscA: \(lossDiscA) Loss-DiscB: \(lossDiscB)")

    if step % 1000 == 0 {
       
        let testImagesA = testImageLoaderA.minibatch(size: testBachSize, imageSize: (imgSize, imgSize))
        let testImagesB = testImageLoaderB.minibatch(size: testBachSize, imageSize: (imgSize, imgSize))

        let genTestA2BOutput = genA2B(testImagesA)
        let genTestB2AOutput = genB2A(testImagesB)
        var imageToSave = testImagesA[0].expandingShape(at:0)
        imageToSave = imageToSave.concatenated(with: testImagesB[0].expandingShape(at:0))
        imageToSave = imageToSave.concatenated(with: genTestA2BOutput[0].expandingShape(at:0))
        imageToSave = imageToSave.concatenated(with: genTestB2AOutput[0].expandingShape(at:0)) 
        do {
            try saveImageGrid(imageToSave, name: "\(fname)_step_\(step)")
        } catch {
            print("Could not save image grid with error: \(error)")
        }
    }

    

    
}

