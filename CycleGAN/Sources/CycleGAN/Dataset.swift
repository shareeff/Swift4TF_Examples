import Foundation
import Just
import Path
import TensorFlow

//let imgSize = 256

enum FileError: Error {
    case file_not_found
}

public struct GANDATASET {
    public let trainingDataset: Dataset<GANDatasetExample>
    public let testDataset: Dataset<GANDatasetExample>

    public init(_ trainPath: Path, _ testPath: Path) {
        self.trainingDataset = Dataset<GANDatasetExample>(elements: loadGANDataFiles(trainPath))
        self.testDataset = Dataset<GANDatasetExample>(elements: loadGANDataFiles(testPath))
    }
}

public extension String {
    @discardableResult
    func shell(_ args: String...) -> String
    {
        let (task,pipe) = (Process(),Pipe())
        task.executableURL = URL(fileURLWithPath: self)
        (task.arguments,task.standardOutput) = (args,pipe)
        do    { try task.run() }
        catch { 
            print("Unexpected error: \(error).") 
            exit(1)
        }

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: String.Encoding.utf8) ?? ""
    }
}

public func downloadFile(_ url: String, dest: String? = nil, force: Bool = false) {
    let dest_name = dest ?? (Path.cwd/url.split(separator: "/").last!).string
    let url_dest = URL(fileURLWithPath: (dest ?? (Path.cwd/url.split(separator: "/").last!).string))
    if !force && Path(dest_name)!.exists { return }

    print("Downloading \(url)...")

    if let cts = Just.get(url).content {
        do    {try cts.write(to: URL(fileURLWithPath:dest_name))}
        catch {print("Can't write to \(url_dest).\n\(error)")}
    } else {
        print("Can't reach \(url)")
    }
}

public func downloadAndExtractFile(_ path: Path, _ fname: String) {
    let baseUrl = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/"
    let file = path/fname
     if !file.exists {
        downloadFile("\(baseUrl)\(fname).zip", dest:(path/"\(fname).zip").string)
        "/usr/bin/unzip".shell((path/"\(fname).zip").string, "-d", path.string)
    }
}

public func loadJpegAsTensor(from file: String) throws -> Tensor<Float> {
    guard FileManager.default.fileExists(atPath: file) else {
        throw FileError.file_not_found
    }
    let imgData = Raw.readFile(filename: StringTensor(file))
    return Tensor<Float>(Raw.decodeJpeg(contents: imgData, channels: 3, dctMethod: "")) / 255
}

public func loadData(_ path: Path) -> Tensor<Float> {
    var files: [String] = []
    do {
        files = try FileManager.default.contentsOfDirectory(atPath: path.string)
    } catch {
        print(error)
        exit(-1)
    }
    //print(files)
    var data: Tensor<Float> = Tensor<Float>(zeros: [1, imgSize, imgSize, 3])
    for file in files{
    
        let imPath = path/file
        guard var imageTensor = try? loadJpegAsTensor(from: imPath.string) else {
            print("Error: Failed to load image \(files[0]). Check file exists and has JPEG format")
            exit(1)
        }
        imageTensor = imageTensor.expandingShape(at:0)
        if file == files[0] {
            data = imageTensor
        } else {
            data = data.concatenated(with: imageTensor)
        }
    }
    return data
}

public func loadGANDataFiles(_ path: Path) -> GANDatasetExample{

    let data = loadData(path)

    return GANDatasetExample(data: data)
}