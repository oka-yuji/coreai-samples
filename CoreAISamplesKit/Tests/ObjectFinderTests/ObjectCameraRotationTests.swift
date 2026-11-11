import FeatureStore
import Foundation
import ObjectFinderCore
import ObjectFinderFeature
import Testing

@testable import ObjectFinderRuntime

@MainActor
struct ObjectCameraRotationTests {
    private let candidate = ObjectCandidate(
        id: 1, region: .init(x: 0.1, y: 0.2, width: 0.3, height: 0.3), similarity: 0.9)

    private func source(cameraID: String = "rear", generation: Int = 1) -> ObjectCameraRotationSource {
        ObjectCameraRotationSource(session: 1, cameraID: cameraID, generation: generation)
    }

    private func active() -> ObjectFinderState {
        var state = ObjectFinderState(folder: URL(filePath: "/model"))
        state.query = "keys"
        _ = ObjectFinderFeature.reduce(&state, .start)
        _ = ObjectFinderFeature.reduce(&state, .started(1))
        return state
    }

    @Test("CAM-01 Changing the capture angle rejects boxes from the previous capture revision")
    func captureChange() {
        var state = active()
        let source = source()
        _ = ObjectFinderFeature.reduce(&state, .rotationSourceChanged(source))
        let frame = ObjectFinderFrame(candidates: [candidate], seconds: 0.1, captureGeneration: 1)
        _ = ObjectFinderFeature.reduce(&state, .analyzed(1, frame))
        #expect(state.frame == frame)
        let commands = ObjectFinderFeature.reduce(&state, .captureAngleChanged(source, 180))
        #expect(state.frame == nil && state.matches.isEmpty && state.orientationRevision == 1)
        #expect(commands == [.cancel(.notification), .run(.orient(state.cameraRotation), id: .orientation)])
        _ = ObjectFinderFeature.reduce(&state, .analyzed(1, frame))
        #expect(state.frame == nil)
    }

    @Test("CAM-02 Preview-only rotation hides mismatched boxes without requesting capture or analysis changes")
    func previewChange() {
        var state = active()
        let source = source()
        var previews: [Double] = []
        var commands: [Command<ObjectFinderEffect, ObjectFinderEffectID>] = []
        let relay = ObjectCameraRotationRelay(
            sourceChanged: { _ = ObjectFinderFeature.reduce(&state, .rotationSourceChanged($0)) },
            previewChanged: { previews.append($0) },
            captureChanged: { commands += ObjectFinderFeature.reduce(&state, .captureAngleChanged($0, $1)) })
        relay.begin(source, preview: 90, capture: 90)
        let before = state
        relay.receive(0, target: .preview, from: source)
        relay.reapplyPreview()
        #expect(previews == [90, 0, 0] && commands.isEmpty && state == before)
        var gate = ObjectPreviewGate()
        gate.expect(8)
        gate.receive([candidate], generation: 8, validUntil: 12, rotation: 90)
        #expect(gate.visible(at: 10, rotation: 0).isEmpty && gate.expiration(at: 10, rotation: 0) == nil)
        gate.receive([], generation: 7, validUntil: 12, rotation: 0)
        #expect(gate.visible(at: 10, rotation: 90) == [candidate])
        #expect(gate.visible(at: 13, rotation: 90).isEmpty)
    }

    @Test(
        "CAM-03 Invalid and repeated normalized angles leave capture orientation and its revision unchanged",
        arguments: [Double.nan, .infinity, -.infinity, 45, 90.5, 90, 90.4, -270, 450])
    func ignoredAngles(_ angle: Double) {
        var state = active()
        let source = source()
        _ = ObjectFinderFeature.reduce(&state, .rotationSourceChanged(source))
        let before = state
        #expect(ObjectFinderFeature.reduce(&state, .captureAngleChanged(source, angle)).isEmpty)
        #expect(state == before)
    }

    @Test(
        "Camera angles wrap into one turn and round to integral supported quarter turns",
        arguments: [(0.0, 0.0), (360, 0), (-360, 0), (450, 90), (-90, 270), (89.5, 90), (359.5, 0)])
    func normalizedAngles(input: Double, expected: Double) {
        #expect(ObjectCameraOrientation.normalized(input) == expected)
    }

    @Test(
        "Non-quarter and non-finite angles are rejected after normalization",
        arguments: [0.5, 45, 90.5, 359.49, Double.nan, .infinity, -.infinity])
    func invalidAngles(_ input: Double) {
        #expect(ObjectCameraOrientation.normalized(input) == nil)
    }

    @Test("CAM-04 Notifications from a replaced camera or ended observation cannot change current angles")
    func staleSources() {
        var state = active()
        let first = source()
        let second = source(cameraID: "replacement", generation: 2)
        var previews: [Double] = []
        let relay = ObjectCameraRotationRelay(
            sourceChanged: { _ = ObjectFinderFeature.reduce(&state, .rotationSourceChanged($0)) },
            previewChanged: { previews.append($0) },
            captureChanged: { _ = ObjectFinderFeature.reduce(&state, .captureAngleChanged($0, $1)) })
        relay.begin(first, preview: 90, capture: 90)
        relay.begin(second, preview: 180, capture: 180)
        let replaced = state
        relay.receive(270, target: .capture, from: first)
        relay.receive(270, target: .preview, from: first)
        _ = ObjectFinderFeature.reduce(&state, .rotationSourceChanged(first))
        _ = ObjectFinderFeature.reduce(&state, .captureAngleChanged(first, 270))
        #expect(state == replaced && previews == [90, 180])
        relay.end()
        relay.receive(0, target: .capture, from: second)
        relay.receive(0, target: .preview, from: second)
        relay.reapplyPreview()
        #expect(state == replaced && previews == [90, 180])
        _ = ObjectFinderFeature.reduce(&state, .stop)
        let stopped = state
        _ = ObjectFinderFeature.reduce(&state, .rotationSourceChanged(second))
        _ = ObjectFinderFeature.reduce(&state, .captureAngleChanged(second, 270))
        #expect(state == stopped && state.orientation.source == nil)
    }

    @Test("Reconnecting at the same angle clears old boxes without changing the capture revision")
    func sameAngleReconnect() {
        var state = active()
        let first = source()
        let second = source(cameraID: "replacement", generation: 2)
        _ = ObjectFinderFeature.reduce(&state, .rotationSourceChanged(first))
        let frame = ObjectFinderFrame(candidates: [candidate], seconds: 0.1, captureGeneration: 8)
        _ = ObjectFinderFeature.reduce(&state, .analyzed(1, frame))
        _ = ObjectFinderFeature.reduce(&state, .rotationSourceChanged(first))
        #expect(state.frame == frame)
        let commands = ObjectFinderFeature.reduce(&state, .rotationSourceChanged(second))
        #expect(commands == [.cancel(.notification)])
        #expect(state.frame == nil && state.matches.isEmpty && state.orientationRevision == 0)
        #expect(state.orientation.source == second)
        var gate = ObjectPreviewGate()
        gate.expect(8)
        gate.receive([candidate], generation: 8, validUntil: 12)
        gate.begin(9, rotation: 90)
        gate.expect(8)
        gate.receive([candidate], generation: 8, validUntil: 12)
        #expect(gate.visible(at: 10, rotation: 90).isEmpty)
        gate.receive([candidate], generation: 9, validUntil: 12)
        gate.expect(9)
        gate.begin(8, rotation: 90)
        #expect(gate.visible(at: 10, rotation: 90) == [candidate])
    }

    @Test("A delayed layout cannot revive old boxes after capture turns away and back before reconnecting")
    func angleChangeBeforeLayout() {
        var gate = ObjectPreviewGate()
        gate.prepareCapture()
        gate.begin(7, rotation: 90)
        gate.expect(7)
        gate.receive([candidate], generation: 7, validUntil: 12, rotation: 90)
        gate.updateCaptureAngle(90)
        #expect(gate.visible(at: 10, rotation: 90) == [candidate])
        gate.updateCaptureAngle(0)
        gate.updateCaptureAngle(90)
        gate.expect(7)
        gate.receive([candidate], generation: 7, validUntil: 12, rotation: 90)
        #expect(gate.visible(at: 10, rotation: 90).isEmpty)
        gate.begin(9, rotation: 90)
        gate.receive([candidate], generation: 9, validUntil: 12, rotation: 90)
        gate.expect(9)
        #expect(gate.visible(at: 10, rotation: 90) == [candidate])
    }

    @Test("A nondefault capture-start angle seeds observation without invalidating the new stream")
    func captureStartupSeed() {
        var gate = ObjectPreviewGate()
        gate.prepareCapture()
        gate.begin(1, rotation: 0)
        gate.expect(1)
        gate.receive([candidate], generation: 1, validUntil: 12, rotation: 0)
        gate.updateCaptureAngle(0)
        #expect(gate.visible(at: 10, rotation: 0) == [candidate])
    }

    @Test("A previous search session cannot bind an observer after a replacement search starts")
    func staleSession() {
        var state = active()
        let previous = source()
        _ = ObjectFinderFeature.reduce(&state, .stop)
        _ = ObjectFinderFeature.reduce(&state, .stopped(1))
        _ = ObjectFinderFeature.reduce(&state, .start)
        let before = state
        _ = ObjectFinderFeature.reduce(&state, .rotationSourceChanged(previous))
        _ = ObjectFinderFeature.reduce(&state, .captureAngleChanged(previous, 0))
        #expect(state == before && state.session == 2)
    }
}
