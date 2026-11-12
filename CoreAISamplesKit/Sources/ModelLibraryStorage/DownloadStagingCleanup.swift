import Darwin
import Foundation

func discardDownloadStaging(at area: URL, paths: [String]) {
    let root = open(area.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
    guard root >= 0 else { return }
    defer { close(root) }
    for path in Set(paths + paths.map { $0 + ".part" }) {
        discardStagingEntry(root, path: path, directory: false)
    }
    let directories = Set(
        paths.flatMap { path in
            let parts = path.split(separator: "/")
            return parts.indices.dropLast().map { parts[...$0].joined(separator: "/") }
        })
    for path in directories.sorted(by: { $0.count > $1.count }) {
        discardStagingEntry(root, path: path, directory: true)
    }
}

private func discardStagingEntry(_ root: Int32, path: String, directory: Bool) {
    let parts = path.split(separator: "/", omittingEmptySubsequences: false)
    guard let first = parts.first, let last = parts.last,
        parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
        UUID(uuidString: String(first)) == nil
    else { return }
    var parent = root
    var opened: [Int32] = []
    defer { for descriptor in opened { close(descriptor) } }
    for part in parts.dropLast() {
        let next = openat(parent, String(part), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard next >= 0 else { return }
        opened.append(next)
        parent = next
    }
    var info = stat()
    guard fstatat(parent, String(last), &info, AT_SYMLINK_NOFOLLOW) == 0,
        info.st_mode & S_IFMT == (directory ? S_IFDIR : S_IFREG)
    else { return }
    _ = unlinkat(parent, String(last), directory ? AT_REMOVEDIR : 0)
}
