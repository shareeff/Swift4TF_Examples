import Foundation
import Just
import Path
import TensorFlow

//let imgSize = 256


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
