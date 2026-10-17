import FeatureStore
import ImageSearchFeature
import SwiftUI
import UniformTypeIdentifiers

struct ImageSearchView: View {
    let store: Store<ImageSearchFeature>
    @State private var choosingModel = false
    @State private var choosingImages = false
    @Environment(\.locale) private var locale

    private var state: ImageSearchState { store.state }

    var body: some View {
        VStack(spacing: 0) {
            controls.padding()
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    if let failure = state.failure { failureView(failure) }
                    if state.photos.isEmpty { emptyView } else { gallery }
                }
                .frame(maxWidth: 1_200)
                .frame(maxWidth: .infinity)
                .padding()
            }
        }
        .demoNavigationTitle("EmbeddingGemma 2")
        .fileImporter(isPresented: $choosingModel, allowedContentTypes: [.folder]) { result in
            switch result {
            case .success(let url): store.send(.load(url))
            case .failure(let error): store.send(.selectionFailed(error.localizedDescription))
            }
        }
        .fileImporter(isPresented: $choosingImages, allowedContentTypes: [.image], allowsMultipleSelection: true) {
            result in
            switch result {
            case .success(let urls): store.send(.index(urls))
            case .failure(let error): store.send(.selectionFailed(error.localizedDescription))
            }
        }
        .sheet(item: Binding(get: { state.selection }, set: { store.send(.select($0)) })) { photo in
            OriginalSearchPhoto(photo: photo)
        }
        .onDisappear { store.send(.disappeared) }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 14) {
            ViewThatFits(in: .horizontal) {
                HStack {
                    modelControls
                    Spacer()
                    imageControls
                }
                VStack(alignment: .leading, spacing: 10) {
                    modelControls
                    imageControls
                }
            }
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField(
                    "Describe an image to find", text: Binding(get: { state.query }, set: { store.send(.setQuery($0)) })
                )
                .textFieldStyle(.plain)
                .onSubmit { store.send(.search) }
                .accessibilityIdentifier("image-search-query")
                Button("Search") { store.send(.search) }
                    .buttonStyle(.borderedProminent)
                    .disabled(!state.canSearch)
                    .accessibilityIdentifier("image-search-submit")
            }
            .padding(10)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
            Menu("Example searches") {
                ForEach(ImageSearchAssembly.queries) { query in
                    Button {
                        store.send(.setQuery(query.text))
                    } label: {
                        Text(verbatim: query.text)
                    }
                }
            }
            .disabled(state.isBusy)
            .accessibilityIdentifier("image-search-examples")
            status
        }
    }

    private var modelControls: some View {
        HStack(spacing: 10) {
            Menu {
                Button("Open model folder…") { choosingModel = true }
                if let folder = state.modelFolder {
                    Button("Reload selected model") { store.send(.load(folder)) }
                }
            } label: {
                Label(
                    LocalizedStringKey(state.ready ? "Model ready" : "Open model"),
                    systemImage: state.ready ? "checkmark.circle" : "folder"
                )
            }
            .disabled(state.isBusy)
            .accessibilityIdentifier("image-search-model")
            Text("EmbeddingGemma 2").font(.caption).foregroundStyle(.secondary)
        }
    }

    private var imageControls: some View {
        HStack {
            Button("Choose images…") { choosingImages = true }
                .disabled(!state.ready || state.isBusy)
            Button("Use sample images") { store.send(.index(ImageSearchAssembly.samples)) }
                .disabled(!state.ready || state.isBusy)
                .accessibilityIdentifier("image-search-samples")
        }
    }

    private var status: some View {
        HStack(spacing: 10) {
            if state.isBusy {
                ProgressView().controlSize(.small)
                switch state.phase {
                case .loading: Text("Preparing the model…")
                case .indexing: Text("Indexing images: \(state.completed) of \(state.total)")
                case .searching: Text("Searching selected images…")
                case .stopping: Text("Stopping…")
                case .idle: EmptyView()
                }
                Spacer()
                Button("Stop") { store.send(.stop) }
                    .disabled(state.phase == .stopping)
            } else {
                Text("Search stays on this device.")
                Spacer()
                if let seconds = state.seconds {
                    let value = DemoStrings(locale: locale).number(seconds, fractionDigits: 2...2)
                    if state.hasSearched { Text("Search: \(value) s") } else { Text("Indexing: \(value) s") }
                }
            }
        }
        .font(.caption)
        .foregroundStyle(.secondary)
    }

    private var emptyView: some View {
        ContentUnavailableView {
            Label("Find images with your own words", systemImage: "photo.on.rectangle.angled")
        } description: {
            Text("Open the model, then choose the images you want to search.")
        }
        .frame(minHeight: 280)
    }

    private var gallery: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .firstTextBaseline) {
                if !state.hasSearched {
                    Text("Selected images").font(.title2.bold())
                } else {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Search results").font(.title2.bold())
                        Text(verbatim: state.submittedQuery).foregroundStyle(.secondary)
                    }
                }
                Spacer()
                let count = state.hasSearched ? state.displayedMatches.count : state.photos.count
                Text("\(count) images").font(.subheadline).foregroundStyle(.secondary)
            }
            if state.hasSearched {
                Toggle(
                    "Show all candidates",
                    isOn: Binding(get: { state.showsAllCandidates }, set: { store.send(.showAllCandidates($0)) })
                )
                .accessibilityIdentifier("image-search-all-candidates")
                Text("Similarity compares meaning; it is not a probability.")
                    .font(.caption).foregroundStyle(.secondary)
                if state.displayedMatches.isEmpty {
                    ContentUnavailableView(
                        "No close matches", systemImage: "magnifyingglass",
                        description: Text("Try different wording, choose more images, or show all candidates.")
                    )
                    .accessibilityIdentifier("image-search-no-matches")
                }
            }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 200, maximum: 360), spacing: 16)], spacing: 20) {
                if !state.hasSearched {
                    ForEach(state.photos) { photo in tile(photo, similarity: nil) }
                } else {
                    ForEach(state.displayedMatches) { match in tile(match.photo, similarity: match.similarity) }
                }
            }
        }
    }

    private func tile(_ photo: SearchPhoto, similarity: Double?) -> some View {
        Button {
            store.send(.select(photo))
        } label: {
            VStack(alignment: .leading, spacing: 8) {
                if let image = ImageSearchAssembly.image(data: photo.thumbnail) {
                    Image(decorative: image, scale: 1).resizable().scaledToFit()
                        .frame(maxWidth: .infinity).aspectRatio(4 / 3, contentMode: .fit)
                        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
                }
                Text(verbatim: photo.url.lastPathComponent).font(.subheadline).lineLimit(1)
                if let similarity {
                    HStack {
                        Text("Similarity")
                        Spacer()
                        Text(similarity, format: .number.precision(.fractionLength(3))).monospacedDigit()
                    }
                    .font(.caption).foregroundStyle(.secondary)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(Text("Open original: \(photo.url.lastPathComponent)"))
    }

    private func failureView(_ failure: ImageSearchFailure) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(LocalizedStringKey(failure.message), systemImage: "exclamationmark.triangle")
            if !failure.detail.isEmpty {
                DisclosureGroup("Details") { Text(verbatim: failure.detail).font(.caption).textSelection(.enabled) }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.orange.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
    }
}

private struct OriginalSearchPhoto: View {
    let photo: SearchPhoto
    @State private var image: CGImage?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if let image {
                    Image(decorative: image, scale: 1).resizable().scaledToFit()
                        .accessibilityLabel(Text(verbatim: photo.url.lastPathComponent))
                } else {
                    ContentUnavailableView("The original image could not be opened.", systemImage: "photo")
                }
            }
            .padding()
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .navigationTitle(Text(verbatim: photo.url.lastPathComponent))
            .toolbar { Button("Done") { dismiss() } }
        }
        #if os(macOS)
            .frame(minWidth: 640, minHeight: 500)
        #endif
        .onAppear { image = ImageSearchAssembly.original(photo.url) }
    }
}
