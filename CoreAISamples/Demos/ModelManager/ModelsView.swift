import DecideFeature
import FeatureStore
import SwiftUI
import UniformTypeIdentifiers

struct ModelsView: View {
    @Environment(ModelsViewModel.self) private var vm
    @Environment(Store<DecideFeature>.self) private var decide
    @Environment(ChatViewModel.self) private var chat
    @Environment(ConversationHost.self) private var host
    @Environment(DemoNavigator.self) private var navigator
    @State private var importing = false
    @State private var pendingDelete: CatalogRow?

    var body: some View {
        List {
            Section("Catalog") {
                if vm.rows.isEmpty {
                    Text("No published Core AI models yet. Open a bundle folder on this Mac below.")
                        .foregroundStyle(.secondary)
                }
                ForEach(vm.rows) { row in
                    CatalogRowView(
                        row: row,
                        inUse: chat.isUsing(row.directory),
                        onDownload: { vm.startDownload(row) },
                        onCancel: { vm.cancelDownload(row) },
                        onLoad: { load(row.directory) },
                        onDelete: { pendingDelete = row })
                }
            }
            Section("On this Mac") {
                ForEach(vm.folders) { folder in
                    FolderRowView(
                        folder: folder,
                        state: chat.state(of: folder.url),
                        busy: chat.isLoading || chat.isResponding || chat.externallyBusy,
                        onLoad: { load(folder.url) },
                        onClear: { vm.clearPrepared(folder.url) },
                        onForget: { vm.forget(folder) })
                }
                Button("Open Folder…", systemImage: "folder") { importing = true }
                DecideModelButton(store: decide)
                if let message = vm.message {
                    RuntimeMessageText(message)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            Section("Storage") {
                LabeledContent("Downloaded models", value: vm.downloadedSize.formatted(.byteCount(style: .file)))
                LabeledContent("Disk available", value: vm.availableSpace.formatted(.byteCount(style: .file)))
            }
        }
        .demoNavigationTitle("Models")
        .onAppear { vm.refresh() }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result { vm.open(url) }
        }
        .alert(
            "Delete this model?",
            isPresented: Binding(get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } })
        ) {
            Button("Delete", role: .destructive) {
                if let row = pendingDelete { vm.delete(row) }
                pendingDelete = nil
            }
            Button("Cancel", role: .cancel) { pendingDelete = nil }
        } message: {
            Text("This removes the downloaded files and the prepared copy from this Mac.")
        }
    }

    private func load(_ url: URL) {
        navigator.selection = .chat
        Task { await ModelSwitch.choose(url, chat: chat, host: host) }
    }
}

struct FolderRowView: View {
    let folder: ModelsViewModel.Folder
    let state: String?
    let busy: Bool
    let onLoad: () -> Void
    let onClear: () -> Void
    let onForget: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(folder.name ?? folder.url.lastPathComponent)
                .fontWeight(.medium)
            Text(folder.id)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
            HStack {
                Group {
                    if folder.problem != nil || folder.name == nil {
                        RuntimeMessageText(folder.problem ?? "Not a readable bundle")
                    } else {
                        Text(folder.size.formatted(.byteCount(style: .file)))
                    }
                }
                .lineLimit(2)
                if let state {
                    Text(LocalizedStringKey(state)).foregroundStyle(.green)
                }
                Spacer()
                Button("Load", action: onLoad)
                    .buttonStyle(.bordered)
                    .disabled(busy || !folder.canLoad)
                Button("Clear Prepared Copy", action: onClear)
                    .disabled(busy || state != nil || !folder.canClear)
                    .help("Remove the copy this app prepared for the GPU. The next load prepares it again.")
                Button("Forget", action: onForget)
                    .disabled(state != nil)
            }
            .font(.caption)
            .controlSize(.small)
        }
        .padding(.vertical, 4)
    }
}

struct CatalogRowView: View {
    let row: CatalogRow
    let inUse: Bool
    let onDownload: () -> Void
    let onCancel: () -> Void
    let onLoad: () -> Void
    let onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(row.model.name)
                .fontWeight(.medium)
            Text(
                "\(row.model.repo)  ·  \(row.model.sizeBytes.formatted(.byteCount(style: .file)))  ·  memory \(Text(LocalizedStringKey(row.model.memory)))  ·  \(Text(LocalizedStringKey(row.model.platforms)))"
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            if row.isDownloading {
                ProgressView(value: row.progress)
                HStack {
                    Text(
                        row.bytesPerSecond > 0
                            ? "\(Int64(row.bytesPerSecond).formatted(.byteCount(style: .file)))/s" : "Preparing…")
                    Spacer()
                    Text(
                        "\(row.downloadedBytes.formatted(.byteCount(style: .file))) / \(row.totalBytes.formatted(.byteCount(style: .file)))"
                    )
                    Button("Pause", action: onCancel)
                        .buttonStyle(.borderless)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            } else if row.isDownloaded {
                HStack {
                    Label(row.diskSize.formatted(.byteCount(style: .file)), systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                    Spacer()
                    Button("Load", action: onLoad)
                        .buttonStyle(.bordered)
                    Button(role: .destructive, action: onDelete) {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.borderless)
                    .disabled(inUse)
                }
                .font(.caption)
            } else {
                HStack {
                    Text(LocalizedStringKey(row.hasPartial ? "Partly downloaded" : "Not downloaded"))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button(LocalizedStringKey(row.hasPartial ? "Resume" : "Download"), action: onDownload)
                        .buttonStyle(.bordered)
                }
                .font(.caption)
            }
            if let error = row.error, !row.isDownloading {
                Label {
                    RuntimeMessageText(error)
                } icon: {
                    Image(systemName: "exclamationmark.triangle.fill")
                }
                .font(.caption)
                .foregroundStyle(.red)
                .lineLimit(2)
            }
        }
        .controlSize(.small)
        .padding(.vertical, 4)
    }
}
