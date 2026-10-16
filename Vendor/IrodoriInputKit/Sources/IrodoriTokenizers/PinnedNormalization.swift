import CTokenizers
import Foundation
import IrodoriInputLogic

func unicode15NFKC(_ text: String) throws -> String {
    let bytes = Array(text.utf8)
    var output = IroBytes()
    var error: UnsafeMutablePointer<CChar>?
    let status = bytes.withUnsafeBufferPointer { iro_nfkc($0.baseAddress, $0.count, &output, &error) }
    defer {
        iro_bytes_free(output)
        iro_error_free(error)
    }
    guard status == 0, output.len == 0 || output.data != nil else {
        throw IrodoriInputError.normalization(error.map { String(cString: $0) } ?? "NFKC failed")
    }
    return String(decoding: UnsafeBufferPointer(start: output.data, count: output.len), as: UTF8.self)
}
