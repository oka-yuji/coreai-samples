import ImageSearchFeature
import SwiftUI
import UniformTypeIdentifiers

struct ImageSearchContent: View {
    let state: ImageSearchState
    let send: @MainActor (ImageSearchAction) -> Void
    let isPermitted: @MainActor () -> Bool
    @State private var choosingImages = false
    @State private var imageRequest = 0
    @Environment(\.locale) private var locale

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            controls
            if let failure = state.failure { failureView(failure) }
            if state.needsIndex && !state.isBusy { indexNotice }
            if state.photos.isEmpty { emptyView } else { gallery }
        }
        .fileImporter(isPresented: $choosingImages, allowedContentTypes: [.image], allowsMultipleSelection: true) {
            result in
            switch result {
            case .success(let urls): send(.imagesChosen(imageRequest, urls))
            case .failure(let error): send(.selectionFailed(imageRequest, error.localizedDescription))
            }
        }
        .sheet(item: Binding(get: { state.selection }, set: { send(.select($0)) })) { photo in
            OriginalSearchPhoto(photo: photo, isPermitted: isPermitted)
        }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Button("Choose images…") {
                    imageRequest = state.session
                    choosingImages = true
                }
                .disabled(!state.canIndex)
                Button("Use sample images") { send(.index(ImageSearchAssembly.samples)) }
                    .disabled(!state.canIndex)
                    .accessibilityIdentifier("image-search-samples")
            }
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField(
                    "Describe an image to find", text: Binding(get: { state.query }, set: { send(.setQuery($0)) })
                )
                .textFieldStyle(.plain)
                .onSubmit { send(.search) }
                .accessibilityIdentifier("image-search-query")
                Button("Search") { send(.search) }
                    .buttonStyle(.borderedProminent)
                    .disabled(!state.canSearch)
                    .accessibilityIdentifier("image-search-submit")
            }
            .padding(10)
            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
            Menu("Example searches") {
                ForEach(ImageSearchAssembly.queries) { query in
                    Button {
                        send(.setQuery(query.text))
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
                Button("Stop") { send(.stop) }
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

    private var indexNotice: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Index the images again to search them.", systemImage: "arrow.clockwise")
            Button("Index again") { send(.index(state.photos.map(\.url))) }
                .buttonStyle(.bordered)
                .disabled(!state.canIndex)
        }
        .font(.callout)
    }

    private var emptyView: some View {
        ContentUnavailableView {
            Label("Find images with your own words", systemImage: "photo.on.rectangle.angled")
        } description: {
            if state.execution == .standalone { Text("Open the model, then choose the images you want to search.") }
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
                        if let identity = state.resultIdentity {
                            Text("Results from \(identity)")
                                .font(.caption).foregroundStyle(.secondary).lineLimit(1).truncationMode(.middle)
                        }
                    }
                }
                Spacer()
                let count = state.hasSearched ? state.displayedMatches.count : state.photos.count
                Text("\(count) images").font(.subheadline).foregroundStyle(.secondary)
            }
            if state.hasSearched {
                Toggle(
                    "Show all candidates",
                    isOn: Binding(get: { state.showsAllCandidates }, set: { send(.showAllCandidates($0)) })
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
            send(.select(photo))
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
    let isPermitted: @MainActor () -> Bool
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
        .onAppear { image = isPermitted() ? ImageSearchAssembly.original(photo.url) : nil }
    }
}
