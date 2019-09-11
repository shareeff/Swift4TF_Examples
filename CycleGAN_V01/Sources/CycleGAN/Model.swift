import Foundation
import TensorFlow


/// Layer for padding with reflection over mini-batch of images
/// Expected input layout is BxHxWxC
@differentiable
func RelectionPad2d(_ input: Tensor<Float>, _ padding: (Int, Int)) -> Tensor<Float> {
    return input.paddedWithReflection(forSizes: [
            (0, 0),
            padding,
            padding,
            (0, 0)
        ])
}


@differentiable
func leakyReLU(_ tensor: Tensor<Float>, negativeSlope: Float) -> Tensor<Float> {
  let zeros = Tensor<Float>(zeros: tensor.shape)
  let minimum = min(zeros, tensor)
  let maximum = max(zeros, tensor)
  let output = maximum + negativeSlope * minimum
  return output
}

public struct Encoder: Layer {

    public var conv1: Conv2D<Float>
    public var conv2: Conv2D<Float>
    public var conv3: Conv2D<Float>
    public var bn1: BatchNorm<Float>
    public var bn2: BatchNorm<Float>
    public var bn3: BatchNorm<Float>
    @noDerivative public let rPadding: (Int, Int) = (3, 3)
   
    public init(
        featureCounts: (Int, Int, Int, Int),
        kernelSizes: (Int, Int, Int)
    ){
        self.conv1 = Conv2D(
            filterShape: (kernelSizes.0, kernelSizes.0, featureCounts.0, featureCounts.1),
            strides: (1, 1),
            padding: .valid
            
            )

        self.conv2 = Conv2D(
            filterShape: (kernelSizes.1, kernelSizes.1, featureCounts.1, featureCounts.2),
            strides: (2, 2),
            padding: .same
        )
        self.conv3 = Conv2D(
            filterShape: (kernelSizes.2, kernelSizes.2, featureCounts.2, featureCounts.3),
            strides: (2, 2),
            padding: .same
        )
        self.bn1 = BatchNorm(featureCount: featureCounts.1)
        self.bn2 = BatchNorm(featureCount: featureCounts.2)
        self.bn3 = BatchNorm(featureCount: featureCounts.3)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        //x = RelectionPad2d(x, rPadding)
        x = x.padded(forSizes: [(0, 0), rPadding, rPadding, (0, 0)], with: 0)
        x = bn1(conv1(x))
        x = relu(x)
        x = bn2(conv2(x))
        x = relu(x)
        x = bn3(conv3(x))
        x = relu(x)
        
        return x
    }

}

public struct Residual: Layer {

    public var conv1: Conv2D<Float>
    public var conv2: Conv2D<Float>
    public var bn1: BatchNorm<Float>
    public var bn2: BatchNorm<Float>
    @noDerivative public let rPadding: (Int, Int) = (1, 1)
   
    public init(
        featureCounts: (Int, Int, Int),
        kernelSizes: (Int, Int)
    ){
        self.conv1 = Conv2D(
            filterShape: (kernelSizes.0, kernelSizes.0, featureCounts.0, featureCounts.1),
            strides: (1, 1),
            padding: .valid
            
            )

        self.conv2 = Conv2D(
            filterShape: (kernelSizes.1, kernelSizes.1, featureCounts.1, featureCounts.2),
            strides: (1, 1),
            padding: .valid
        )
        self.bn1 = BatchNorm(featureCount: featureCounts.1)
        self.bn2 = BatchNorm(featureCount: featureCounts.2)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        //x = RelectionPad2d(x, rPadding)
        x = x.padded(forSizes: [(0, 0), rPadding, rPadding, (0, 0)], with: 0)
        x = bn1(conv1(x))
        x = relu(x)
        //x = RelectionPad2d(x, rPadding)
        x = x.padded(forSizes: [(0, 0), rPadding, rPadding, (0, 0)], with: 0)
        x = bn2(conv2(x))
        
        return x + input
    }

}


public struct Decoder: Layer {
    public var conv1: TransposedConv2D<Float>
    public var conv2: TransposedConv2D<Float>
    public var conv3: Conv2D<Float>
    public var bn1: BatchNorm<Float>
    public var bn2: BatchNorm<Float>
    public var bn3: BatchNorm<Float>
    @noDerivative public let rPadding: (Int, Int) = (3, 3)
   
    public init(
        featureCounts: (Int, Int, Int, Int),
        kernelSizes: (Int, Int, Int)
    ){
        self.conv1 = TransposedConv2D(
            filterShape: (kernelSizes.0, kernelSizes.0, featureCounts.1, featureCounts.0),
            strides: (2, 2),
            padding: .same
        )
        self.conv2 = TransposedConv2D(
            filterShape: (kernelSizes.1, kernelSizes.1, featureCounts.2, featureCounts.1),
            strides: (2, 2),
            padding: .same
        )
        self.conv3 = Conv2D(
            filterShape: (kernelSizes.2, kernelSizes.2, featureCounts.2, featureCounts.3),
            strides: (1, 1),
            padding: .valid
            
            )

        self.bn1 = BatchNorm(featureCount: featureCounts.1)
        self.bn2 = BatchNorm(featureCount: featureCounts.2)
        self.bn3 = BatchNorm(featureCount: featureCounts.3)

    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        x = relu(bn1(conv1(x)))
        x = relu(bn2(conv2(x)))
        //x = RelectionPad2d(x, rPadding)
        x = x.padded(forSizes: [(0, 0), rPadding, rPadding, (0, 0)], with: 0)
        x = tanh(bn3(conv3(x)))
    

        return x

    }
}


public struct Generator: Layer {
    public var encoder = Encoder(
        featureCounts: (3, 32, 64, 128),
        kernelSizes: (7, 3, 3)
    )
    public var res1 = Residual(
        featureCounts: (128, 128, 128),
        kernelSizes: (3, 3)
    )
    public var res2 = Residual(
        featureCounts: (128, 128, 128),
        kernelSizes: (3, 3)
    )
    public var res3 = Residual(
        featureCounts: (128, 128, 128),
        kernelSizes: (3, 3)
    )
    public var res4 = Residual(
        featureCounts: (128, 128, 128),
        kernelSizes: (3, 3)
    )
    public var res5 = Residual(
        featureCounts: (128, 128, 128),
        kernelSizes: (3, 3)
    )
    public var res6 = Residual(
        featureCounts: (128, 128, 128),
        kernelSizes: (3, 3)
    )
    public var res7 = Residual(
        featureCounts: (128, 128, 128),
        kernelSizes: (3, 3)
    )
    public var res8 = Residual(
        featureCounts: (128, 128, 128),
        kernelSizes: (3, 3)
    )
    public var res9 = Residual(
        featureCounts: (128, 128, 128),
        kernelSizes: (3, 3)
    )

    public var decoder = Decoder(
        featureCounts: (128, 64, 32, 3),
        kernelSizes: (3, 3, 7)
    )

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        x = encoder(x)
        x = res1(x)
        x = res2(x)
        x = res3(x)
        x = res4(x)
        x = res5(x)
        x = res6(x)
        x = res7(x)
        x = res8(x)
        x = res9(x)
        x = decoder(x)
        
        return x
        
    }
}

public struct Discriminator: Layer {
    public var conv1 = Conv2D<Float>(
            filterShape: (4, 4, 3, 64),
            strides: (2, 2),
            padding: .same
        )
    public var conv2 = Conv2D<Float>(
            filterShape: (4, 4, 64, 128),
            strides: (2, 2),
            padding: .same
        )

    public var conv3 = Conv2D<Float>(
            filterShape: (4, 4, 128, 256),
            strides: (2, 2),
            padding: .same
     )

    public var conv4 = Conv2D<Float>(
            filterShape: (4, 4, 256, 512),
            strides: (1, 1),
            padding: .same
        )

    public var conv5 = Conv2D<Float>(
            filterShape: (4, 4, 512, 1),
            strides: (1, 1),
            padding: .same
        )

    public var bn1 = BatchNorm<Float>(featureCount: 128)
    public var bn2 = BatchNorm<Float>(featureCount: 256)
    public var bn3 = BatchNorm<Float>(featureCount: 512)

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var output: Tensor<Float>
        output = leakyReLU(conv1(input),  negativeSlope: 0.2)
        output = leakyReLU(bn1(conv2(output)),  negativeSlope: 0.2)
        output = leakyReLU(bn2(conv3(output)),  negativeSlope: 0.2)
        output = leakyReLU(bn3(conv4(output)),  negativeSlope: 0.2)
        output = conv5(output)
        
        return output
        
    }
}



// Loss functions

@differentiable
func generatorLoss(fakeLogits: Tensor<Float>) -> Tensor<Float> {
    meanSquaredError(
        predicted: fakeLogits,
        expected: Tensor(ones: fakeLogits.shape))
}

@differentiable
func discriminatorLoss(realLogits: Tensor<Float>, fakeLogits: Tensor<Float>) -> Tensor<Float> {
    let realLoss =  meanSquaredError(
        predicted: realLogits,
        expected: Tensor(ones: realLogits.shape))
    let fakeLoss =  meanSquaredError(
        predicted: fakeLogits,
        expected: Tensor(zeros: fakeLogits.shape))
    return realLoss + fakeLoss
}


@differentiable
func cycleConsistencyLoss(dataA: Tensor<Float>, dataB: Tensor<Float>,
    reconstructedDataA: Tensor<Float>, reconstructedDataB: Tensor<Float>, cycLambda: Float = 10) -> Tensor<Float> {
    
    let loss = (abs(dataA - reconstructedDataA) + abs(dataB - reconstructedDataB)).mean()
                            
    return cycLambda * loss                       
}


@differentiable
func identityLoss(dataA: Tensor<Float>, dataB: Tensor<Float>,
    genA2BOutput: Tensor<Float>, genB2AOutput: Tensor<Float>, identityLambda: Float = 5) -> Tensor<Float> {
    
    let loss = (abs(dataA - genB2AOutput) + abs(dataB - genA2BOutput)).mean()
                            
    return identityLambda * loss                       
}