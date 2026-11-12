import ChronosFeature
import Darwin
import Foundation

enum ChronosJournalBoundary {
    static func folder(_ name: String, in parent: Int32) throws -> Int32 {
        let descriptor = openat(parent, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw ChronosDemoFailure.invalidData }
        return descriptor
    }

    static func names(_ descriptor: Int32) throws -> Set<String> {
        let copy = dup(descriptor)
        guard copy >= 0 else { throw ChronosDemoFailure.persistence }
        guard let directory = fdopendir(copy) else {
            close(copy)
            throw ChronosDemoFailure.persistence
        }
        defer { closedir(directory) }
        rewinddir(directory)
        var names: Set<String> = []
        while true {
            errno = 0
            guard let entry = readdir(directory) else {
                guard errno == 0 else { throw ChronosDemoFailure.persistence }
                return names
            }
            let name = withUnsafePointer(to: &entry.pointee.d_name) {
                $0.withMemoryRebound(to: CChar.self, capacity: Int(MAXNAMLEN) + 1) { String(cString: $0) }
            }
            if name != ".", name != ".." { names.insert(name) }
        }
    }

    static func read(_ name: String, in folder: Int32, limit: Int) throws -> Data {
        try Task.checkCancellation()
        let descriptor = openat(folder, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard descriptor >= 0 else { throw ChronosDemoFailure.invalidData }
        let file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? file.close() }
        var before = stat()
        guard fstat(descriptor, &before) == 0, before.st_mode & S_IFMT == S_IFREG,
            before.st_nlink == 1, before.st_size > 0, before.st_size <= limit
        else { throw ChronosDemoFailure.invalidData }
        var data = Data()
        while true {
            try Task.checkCancellation()
            let count = try autoreleasepool {
                let chunk = try file.read(upToCount: min(1_048_576, limit + 1 - data.count)) ?? Data()
                data.append(chunk)
                return chunk.count
            }
            guard data.count <= limit else { throw ChronosDemoFailure.invalidData }
            if count == 0 { break }
        }
        var after = stat()
        guard fstat(descriptor, &after) == 0, data.count == before.st_size, before.st_size == after.st_size,
            before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
            before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
            before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec,
            before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec
        else { throw ChronosDemoFailure.invalidData }
        return data
    }
}
