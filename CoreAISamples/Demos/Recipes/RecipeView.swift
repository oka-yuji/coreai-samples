import AppKit
import FeatureStore
import RecipeCore
import RecipeFeature
import SwiftUI
import UniformTypeIdentifiers

struct RecipeView: View {
    let store: Store<RecipeFeature>
    @Environment(\.locale) private var locale
    @State private var importingPhoto = false
    @State private var importingModel = false
    private var strings: DemoStrings { DemoStrings(locale: locale) }

    var body: some View {
        VStack(spacing: 0) {
            header
            RecipeModelView(
                state: store.state, open: { importingModel = true },
                retry: { store.send(.openModel(store.state.modelPath)) })
            Divider()
            HSplitView {
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        photoSection
                        inventorySection
                        kitchenSection
                    }
                    .padding(20)
                }
                .frame(minWidth: 310, idealWidth: 380)
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        RecipeInspectorView(state: store.state)
                        recipeSection
                        if let report = store.state.report {
                            RecipeReportView(report: report)
                        } else if store.state.recipe != nil {
                            Label("Review your inventory, then check this recipe.", systemImage: "checklist")
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(20)
                }
                .frame(minWidth: 350, idealWidth: 450)
            }
            Divider()
            footer
        }
        .demoNavigationTitle("Recipes")
        .onAppear { store.send(.appeared) }
        .onDisappear { store.send(.ended) }
        .fileImporter(isPresented: $importingPhoto, allowedContentTypes: [.image]) { result in
            receive(result) { store.send(.importPhoto($0)) }
        }
        .fileImporter(isPresented: $importingModel, allowedContentTypes: [.folder]) { result in
            receive(result) { store.send(.openModel($0.path)) }
        }
    }

    private var header: some View {
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Cook with what you have").font(.title2.bold())
                Text("A photo, a recipe, and an inventory you have checked.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Spacer()
            if store.state.busy {
                ProgressView().controlSize(.small)
                Button("Stop") { store.send(.stop) }
                    .disabled(store.state.phase == .stopping)
            } else {
                Button("Check recipe", systemImage: "checkmark.circle") { store.send(.check) }
                    .buttonStyle(.borderedProminent)
                    .disabled(!store.state.canCheck)
            }
        }
        .padding(20)
    }

    private var photoSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("1. Your fridge").font(.headline)
                Spacer()
                Menu("Examples") {
                    ForEach(store.state.examples) { example in
                        Button(LocalizedStringKey(example.title)) { store.send(.selectExample(example)) }
                    }
                }
                .fixedSize()
                Button("Open photo", systemImage: "photo") { importingPhoto = true }
            }
            .disabled(store.state.busy)
            Group {
                if let photo = store.state.photo, let image = NSImage(data: photo.image) {
                    Image(nsImage: image).resizable().scaledToFit()
                        .accessibilityLabel(Text("Selected fridge photo: \(photo.filename)"))
                } else {
                    VStack(spacing: 10) {
                        Image(systemName: "refrigerator").font(.system(size: 44, weight: .light))
                        Text("Start with a fridge photo").font(.headline)
                        Text("Or try one of the five generated examples.").font(.caption)
                    }
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, minHeight: 190)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: 260)
            .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 12))
            if let photo = store.state.photo {
                Text("Photo suggestions are not confirmed inventory.")
                    .font(.caption).foregroundStyle(.secondary)
                if !photo.checks.isEmpty {
                    DisclosureGroup("Photo model notes (original)") {
                        ForEach(Array(photo.checks.enumerated()), id: \.offset) { _, text in
                            Text(verbatim: text).font(.caption).frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.vertical, 3)
                        }
                    }
                }
            }
        }
    }

    private var inventorySection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("2. Review your inventory").font(.headline)
                Spacer()
                Button("Add", systemImage: "plus") { store.send(.addItem) }
                    .disabled(store.state.pantry.count >= 32)
            }
            Text("Check names and amounts. Add pantry staples and anything outside the photo.")
                .font(.caption).foregroundStyle(.secondary)
            ForEach(store.state.pantry) { item in
                HStack(spacing: 8) {
                    Toggle(
                        "Confirm \(strings.ingredient(item.name))",
                        isOn: Binding(
                            get: { item.confirmed }, set: { store.send(.confirm(item.id, $0)) })
                    )
                    .labelsHidden().toggleStyle(.checkbox)
                    .help("I have checked this ingredient and its amount.")
                    TextField(
                        "Ingredient",
                        text: Binding(
                            get: { strings.ingredient(item.name) },
                            set: {
                                store.send(
                                    .editDisplayedName(
                                        item.id, $0, previousDisplay: strings.ingredient(item.name),
                                        originalSource: item.name))
                            })
                    )
                    .accessibilityLabel(Text("Ingredient name \(item.id)"))
                    TextField(
                        "Amount",
                        text: Binding(
                            get: { item.amount }, set: { store.send(.setAmount(item.id, $0)) })
                    )
                    .frame(width: 85).accessibilityLabel(Text("Amount for \(strings.ingredient(item.name))"))
                    Button("Remove \(strings.ingredient(item.name))", systemImage: "minus.circle") {
                        store.send(.removeItem(item.id))
                    }
                    .labelStyle(.iconOnly).buttonStyle(.plain).foregroundStyle(.secondary)
                }
            }
            Text(
                "Use the recipe's units (for example, 1 pcs or 150 g), or enter enough for the whole recipe. Editing an item clears its confirmation."
            )
            .font(.caption).foregroundStyle(.secondary)
            Button("Confirm listed items") { store.send(.confirmListedItems) }
                .disabled(store.state.pantry.isEmpty || store.state.pantry.allSatisfy(\.confirmed))
            Toggle(
                "I have checked the complete inventory, including items outside this photo.",
                isOn: Binding(
                    get: { store.state.inventoryComplete }, set: { store.send(.setInventoryComplete($0)) })
            )
            .toggleStyle(.checkbox).font(.callout)
            Text("Leave this unchecked when unseen ingredients are still unknown.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .disabled(store.state.busy)
        .textFieldStyle(.roundedBorder)
    }

    private var kitchenSection: some View {
        DisclosureGroup("3. Check kitchen equipment") {
            VStack(spacing: 8) {
                ForEach(store.state.kitchen) { item in
                    HStack {
                        Text(LocalizedStringKey(item.name.capitalized)).frame(maxWidth: .infinity, alignment: .leading)
                        Picker(
                            LocalizedStringKey(item.name.capitalized),
                            selection: Binding(
                                get: { item.status }, set: { store.send(.setKitchen(item.name, $0)) }
                            )
                        ) {
                            ForEach(KitchenStatus.allCases, id: \.self) { status in
                                Text(LocalizedStringKey(status.title)).tag(status)
                            }
                        }
                        .labelsHidden().frame(width: 140)
                    }
                }
            }
            .padding(.top, 10)
        }
        .disabled(store.state.busy)
    }

    private var recipeSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("4. Recipe to check (original)").font(.headline)
            HStack {
                TextField(
                    "Paste a Kikkoman recipe URL",
                    text: Binding(
                        get: { store.state.urlText }, set: { store.send(.setURL($0)) })
                )
                .textFieldStyle(.roundedBorder)
                .onSubmit { store.send(.fetchRecipe) }
                Button("Read") { store.send(.fetchRecipe) }.disabled(store.state.urlText.isEmpty)
            }
            .disabled(store.state.busy)
            Text("Supports public kikkoman.co.jp recipe pages. Reading the URL requires internet.")
                .font(.caption).foregroundStyle(.secondary)
            if let recipe = store.state.recipe {
                VStack(alignment: .leading, spacing: 10) {
                    Text(verbatim: recipe.title).font(.title3.bold()).textSelection(.enabled)
                    HStack {
                        Text("Servings: \(recipe.servings)").foregroundStyle(.secondary)
                        Spacer()
                        if let url = URL(string: recipe.source) { Link("Original recipe", destination: url) }
                    }
                    Divider()
                    ForEach(recipe.ingredients) { ingredient in
                        HStack(alignment: .firstTextBaseline) {
                            Text(verbatim: ingredient.name).frame(maxWidth: .infinity, alignment: .leading)
                            Text(verbatim: ingredient.amount).foregroundStyle(.secondary)
                            Button("Add") { store.send(.addRecipeIngredient(ingredient)) }
                                .help("Add this ingredient to your inventory for review.")
                                .disabled(store.state.busy || store.state.pantry.count >= 32)
                        }
                    }
                    DisclosureGroup("Read the method (original)") {
                        ForEach(Array(recipe.steps.enumerated()), id: \.offset) { index, step in
                            Text(verbatim: "\(index + 1). " + step).font(.callout)
                                .frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 5)
                        }
                    }
                    .padding(.top, 4)
                }
                .padding(16)
                .background(.background.secondary, in: RoundedRectangle(cornerRadius: 12))
            }
        }
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let message = store.state.message {
                Label {
                    RuntimeMessageText(message)
                } icon: {
                    Image(systemName: "info.circle")
                }
                .font(.callout).textSelection(.enabled)
            }
            if let problem = store.state.input?.problem {
                Text(LocalizedStringKey(problem)).font(.caption).foregroundStyle(.orange)
            }
            HStack {
                phaseLabel.font(.caption).foregroundStyle(.secondary)
                Spacer()
            }
        }
        .padding(.horizontal, 20).padding(.vertical, 10)
    }

    @ViewBuilder
    private var phaseLabel: some View {
        switch store.state.phase {
        case .analyzing: Text("Reading the photo on this Mac...")
        case .fetching: Text("Reading the recipe URL...")
        case .loading: Text("Opening the decision model...")
        case .checking: Text("Checking recipe requirements and equipment...")
        case .stopping: Text("Waiting for the current operation to stop...")
        default:
            if let name = store.state.modelName {
                Text("On-device photo reading + \(name)")
            } else {
                Text("Photos stay on this Mac. Open a decision bundle to check recipes.")
            }
        }
    }

    private func receive(_ result: Result<URL, Error>, perform: (URL) -> Void) {
        switch result {
        case .success(let url): perform(url)
        case .failure(let error):
            if (error as? CocoaError)?.code != .userCancelled { store.send(.uiError(error.localizedDescription)) }
        }
    }
}
