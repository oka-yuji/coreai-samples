#if os(iOS)
    import FeatureStore
    import Foundation
    import ImageGenerationCore
    import ImageGenerationRuntime
    import ModelLibraryFeature
    import ModelLibraryRuntime

    @MainActor
    final class IOSImageGenerationConnectionCheck {
        typealias Factory =
            @MainActor (
                ModelDemoSession, @escaping ManagedImageGeneration.Permission
            ) -> (service: LiveImageGenerationService, budget: ImageMemoryBudget)?

        private struct Report: Encodable {
            let scope =
                "Normal Models acquisition, preparation, adapter and live-service binding; no image generation or UI interaction"
            var factoryCalls = 0
            var serviceCreated = false
            var modelID: String?
            var artifactID: String?
            var contract: String?
            var revision: String?
            var sourceKind: ModelSourceKind?
            var budget: ImageMemoryBudget?
            var prepared = false
            var permissionAfterBinding: Bool?
            var closed = false
            var permissionAfterClose: Bool?
            var failures: [String] = []
        }

        private let factory: Factory
        private let output: URL
        private var report = Report()
        private var createdSession: ModelDemoSession?
        private var permission: ManagedImageGeneration.Permission?

        init(output: URL, factory: @escaping Factory) {
            self.output = output
            self.factory = factory
        }

        func makeService(
            for session: ModelDemoSession, permission: @escaping ManagedImageGeneration.Permission
        ) -> LiveImageGenerationService? {
            report.factoryCalls += 1
            createdSession = session
            self.permission = permission
            report.modelID = session.resource.modelID
            report.artifactID = session.resource.artifactID
            report.contract = session.resource.contract
            report.revision = session.resource.revision
            report.sourceKind = session.resource.sourceKind
            guard let created = factory(session, permission) else {
                report.failures.append("The production factory did not provide a calibrated image service")
                return nil
            }
            report.serviceCreated = true
            report.budget = created.budget
            return created.service
        }

        func check(_ library: ModelLibraryAssembly.Library) async -> [String] {
            let selection = DemoSelection(modelID: "qwen-image21-turbo", demoID: "image-generation")
            library.store.send(.openDemo(selection))
            guard let session = library.store.state.session,
                session.selection == selection, session.resource.artifactID == QwenImage21Artifact.nativeComplex.id
            else {
                library.store.send(.closeDemo)
                report.failures.append("The ordinary Models store did not open the expected image session")
                return save()
            }
            _ = library.rendering.content(for: session)
            library.store.send(.requestDemoPreparation(session.id))
            let settled = await wait {
                library.store.state.session != session || library.store.state.demoPreparation != .preparing
            }
            report.prepared =
                settled && library.store.state.session == session
                && library.store.state.demoPreparation == .available && library.store.state.failure == nil
            if report.prepared {
                _ = library.rendering.content(for: session)
                report.permissionAfterBinding = permission?()
                if report.factoryCalls != 1 || !report.serviceCreated || createdSession != session
                    || report.permissionAfterBinding != true
                {
                    report.failures.append(
                        "The ordinary renderer did not bind exactly one permitted live image service")
                }
            } else {
                report.failures.append("Image session preparation did not complete successfully")
            }
            library.store.send(.closeDemo)
            let stopped = await wait {
                library.store.state.session == nil || library.store.state.failure != nil
            }
            report.closed =
                stopped && library.store.state.session == nil && library.store.state.pendingDestination == nil
                && library.store.state.failure == nil
            report.permissionAfterClose = permission?()
            if !report.closed || (report.serviceCreated && report.permissionAfterClose != false) {
                report.failures.append(
                    "Closing the image entry did not confirm release and revoke execution permission")
            }
            if Task.isCancelled { report.failures.append("The connection check was cancelled") }
            return save()
        }

        private func wait(_ finished: @escaping @MainActor @Sendable () -> Bool) async -> Bool {
            await withTaskGroup(of: Bool.self) { group in
                group.addTask { await self.observe(finished) }
                group.addTask {
                    try? await Task.sleep(for: .seconds(30))
                    return false
                }
                let result = await group.next() ?? false
                group.cancelAll()
                return result
            }
        }

        private func observe(_ finished: @escaping @MainActor @Sendable () -> Bool) async -> Bool {
            for await value in Observations(finished) where value { return true }
            return false
        }

        private func save() -> [String] {
            do {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(report).write(to: output, options: .atomic)
            } catch {
                report.failures.append("Could not save image connection evidence: \(error)")
            }
            return report.failures
        }
    }
#endif
