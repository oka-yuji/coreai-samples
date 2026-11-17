import Testing
import VoxCPMCore

@Suite("VoxCPM2 diffusion solver")
struct VoxCPMSolverTests {
    @Test("The ten-step time span matches the fixture time span bit for bit")
    func timeSpan() throws {
        let expected = try VoxCPMReference.floats(VoxCPMReference.load().timeSpan)
        try #require(expected.count == 11)
        #expect(VoxCPMReference.bits(VoxCPMSolver.timeSpan) == VoxCPMReference.bits(expected))
    }

    @Test(
        "Feeding the fixture estimator outputs in order gives the fixture latent within a maximum normalized error of 1e-5",
        arguments: [0, 13, 33])
    func latent(patchNumber: Int) throws {
        let reference = try VoxCPMReference.load()
        let patch = try reference.patch(patchNumber)
        var solver = try VoxCPMSolver(
            noise: reference.noise(patchNumber), ditHidden: VoxCPMReference.floats(patch.ditHidden),
            condition: VoxCPMReference.floats(patch.condition))
        for call in patch.calls {
            try #require(solver.estimatorInput != nil)
            try solver.advance(velocity: VoxCPMReference.floats(call.velocity))
        }
        #expect(solver.estimatorInput == nil)
        let expected = try VoxCPMReference.floats(patch.latent)
        #expect(VoxCPMReference.maximumNormalizedError(solver.latent, expected) <= 1e-5)
    }

    @Test("The first of the ten steps uses no estimator output, so the first estimator input carries the noise")
    func firstStep() throws {
        let reference = try VoxCPMReference.load()
        let patch = try reference.patch(0)
        let noise = try reference.noise(0)
        let times = try VoxCPMReference.floats(reference.timeSpan)
        try #require(times.count == 11)
        var solver = try VoxCPMSolver(
            noise: noise, ditHidden: VoxCPMReference.floats(patch.ditHidden),
            condition: VoxCPMReference.floats(patch.condition))
        #expect(VoxCPMReference.bits(solver.latent) == VoxCPMReference.bits(noise))
        let first = try #require(solver.estimatorInput)
        #expect(VoxCPMReference.bits(first.x) == VoxCPMReference.bits(noise + noise))
        #expect(VoxCPMReference.bits(first.t) == VoxCPMReference.bits([times[1], times[1]]))
        var requests = 0
        for call in patch.calls where solver.estimatorInput != nil {
            requests += 1
            try solver.advance(velocity: VoxCPMReference.floats(call.velocity))
        }
        #expect(requests == 9)
        #expect(solver.estimatorInput == nil)
    }

    @Test("Every estimator input matches the fixture input of the same call", arguments: [0, 13, 33])
    func estimatorInputs(patchNumber: Int) throws {
        let reference = try VoxCPMReference.load()
        let patch = try reference.patch(patchNumber)
        let hidden = try VoxCPMReference.floats(patch.ditHidden)
        let condition = try VoxCPMReference.floats(patch.condition)
        var solver = try VoxCPMSolver(noise: reference.noise(patchNumber), ditHidden: hidden, condition: condition)
        for (index, call) in patch.calls.enumerated() {
            let input = try #require(solver.estimatorInput, "call \(index)")
            let x = try VoxCPMReference.floats(call.x)
            if index == 0 {
                #expect(VoxCPMReference.bits(input.x) == VoxCPMReference.bits(x), "call \(index)")
            } else {
                #expect(VoxCPMReference.maximumNormalizedError(input.x, x) <= 1e-5, "call \(index)")
            }
            let mu = hidden + [Float](repeating: 0, count: hidden.count)
            let t = try VoxCPMReference.floats(call.t)
            let dt = try VoxCPMReference.floats(call.dt)
            #expect(VoxCPMReference.bits(input.mu) == VoxCPMReference.bits(mu), "call \(index)")
            #expect(VoxCPMReference.bits(input.t) == VoxCPMReference.bits(t), "call \(index)")
            #expect(VoxCPMReference.bits(input.cond) == VoxCPMReference.bits(condition + condition), "call \(index)")
            #expect(VoxCPMReference.bits(input.dt) == VoxCPMReference.bits(dt), "call \(index)")
            try solver.advance(velocity: VoxCPMReference.floats(call.velocity))
        }
    }

    @Test("Inputs that do not have the VoxCPM2 sizes and a velocity after the last step are rejected")
    func invalidInputs() throws {
        let patch = [Float](repeating: 0, count: 256)
        let hidden = [Float](repeating: 0, count: 2048)
        let velocity = [Float](repeating: 0, count: 512)
        #expect(throws: VoxCPMError.self) {
            try VoxCPMSolver(noise: Array(patch.dropLast()), ditHidden: hidden, condition: patch)
        }
        #expect(throws: VoxCPMError.self) {
            try VoxCPMSolver(noise: patch, ditHidden: Array(hidden.dropLast()), condition: patch)
        }
        #expect(throws: VoxCPMError.self) {
            try VoxCPMSolver(noise: patch, ditHidden: hidden, condition: Array(patch.dropLast()))
        }
        var solver = try VoxCPMSolver(noise: patch, ditHidden: hidden, condition: patch)
        #expect(throws: VoxCPMError.self) { try solver.advance(velocity: Array(velocity.dropLast())) }
        for _ in 0..<9 {
            try solver.advance(velocity: velocity)
        }
        #expect(throws: VoxCPMError.self) { try solver.advance(velocity: velocity) }
    }
}
