
import TensorFlow

public struct GANDatasetExample: TensorGroup {
    public var data: Tensor<Float>

    public init(data: Tensor<Float>) {
        self.data = data
    }

    public init<C: RandomAccessCollection>(
        _handles: C
    ) where C.Element: _AnyTensorHandle {
        precondition(_handles.count == 1) //2
        //let labelIndex = _handles.startIndex
        //let dataIndex = _handles.index(labelIndex, offsetBy: 1)
        let dataIndex = _handles.startIndex
        //label = Tensor<Int32>(handle: TensorHandle<Int32>(handle: _handles[labelIndex]))
        data = Tensor<Float>(handle: TensorHandle<Float>(handle: _handles[dataIndex]))
    }
}
