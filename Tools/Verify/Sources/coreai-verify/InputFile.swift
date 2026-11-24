import Darwin
import Foundation

enum InputFile {
    static func read(_ url: URL, label: String, expectedBytes: Int?) throws -> Data {
        let descriptor = open(url.path(percentEncoded: false), O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else {
            throw errno == ELOOP
                ? FunctionError.notRegularFile(label)
                : FunctionError.cannot("read \(label)", String(cString: strerror(errno)))
        }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        var status = stat()
        guard fstat(descriptor, &status) == 0, status.st_mode & S_IFMT == S_IFREG else {
            throw FunctionError.notRegularFile(label)
        }
        let size = Int(status.st_size)
        if let expectedBytes, size != expectedBytes {
            throw FunctionError.fileSize(label, expected: expectedBytes, actual: size)
        }
        let data = try handle.read(upToCount: size) ?? Data()
        guard data.count == size else { throw FunctionError.fileSize(label, expected: size, actual: data.count) }
        return data
    }
}
