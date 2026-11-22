import ArgumentParser
import CoreAIBackend

struct ReloadOptions: ParsableArguments {
    @Option(
        help:
            "Completed reads before reloading at the next new conversation. Zero disables the interval. Own runtime only."
    )
    var reloadInterval: Int?

    @Option(
        help:
            "Process footprint bytes that trigger a reload before reading. Zero disables the threshold. Own runtime only."
    )
    var reloadFootprintBytes: UInt64?

    @Option(
        help:
            "Graphics footprint growth bytes that trigger a reload. Omit for the per-asset default; zero disables. Own runtime only."
    )
    var reloadGraphicsGrowthBytes: UInt64?

    func validate(_ runtime: RuntimeKind) throws {
        if runtime != .own, reloadInterval != nil || reloadFootprintBytes != nil || reloadGraphicsGrowthBytes != nil {
            throw ValidationError(
                "--reload-interval, --reload-footprint-bytes and --reload-graphics-growth-bytes apply to --runtime own only."
            )
        }
        if let reloadInterval, reloadInterval < 0 {
            throw ValidationError("--reload-interval must be zero or greater.")
        }
    }

    var policy: LocalLanguageModel.ReloadPolicy {
        var policy = LocalLanguageModel.ReloadPolicy()
        if let reloadInterval { policy.readInterval = reloadInterval }
        if let reloadFootprintBytes { policy.footprintLimitBytes = reloadFootprintBytes }
        if let reloadGraphicsGrowthBytes { policy.graphicsGrowthLimitBytes = reloadGraphicsGrowthBytes }
        return policy
    }
}
