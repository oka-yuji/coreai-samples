import Foundation
import SwiftUI

struct DemoApp: App {
    @NSApplicationDelegateAdaptor(ConversationLifetime.self) private var lifetime
    var body: some Scene {
        WindowGroup("Core AI Demos") {
            if LaunchArguments.flag("--product-preview") {
                ProductPreviewRoot()
            } else {
                DemoRootView(lifetime: lifetime)
            }
        }
        .defaultSize(width: 1040, height: 720)
    }
}

@main
enum DemoAppMain {
    static func main() async {
        if let directory = LaunchArguments.value("--image-search-eval"),
            let folder = LaunchArguments.value("--embedding-bundle")
        {
            exit(await ImageSearchSelfTest.run(directory: URL(filePath: directory), folder: URL(filePath: folder)))
        }
        if LaunchArguments.flag("--check-localization") {
            DemoLocalizationCheck.run()
            return
        }
        if let directory = LaunchArguments.value("--product-source-check") {
            await ProductSelfTest.run(directory: directory, modelFolder: nil)
            return
        }
        if let directory = LaunchArguments.value("--product-e2e") {
            guard let folder = LaunchArguments.value("--extract") else {
                FileHandle.standardError.write(Data("Set --extract to a GLiNER bundle folder.\n".utf8))
                exit(1)
            }
            await ProductSelfTest.run(directory: directory, modelFolder: folder)
            return
        }
        if let directory = LaunchArguments.value("--recipe-e2e") {
            await RecipeSelfTest.run(directory: directory)
            return
        }
        if let directory = LaunchArguments.value("--decide-e2e") {
            await DecideSelfTest.run(directory: directory)
            return
        }
        if LaunchArguments.flag("--decide-json") {
            await DecideCommand.run()
            return
        }
        if LaunchArguments.flag("--selftest") {
            SelfTest.run()
        }
        if LaunchArguments.flag("--agent-selftest") {
            AgentSelfTest.run()
        }
        if LaunchArguments.flag("--agent-e2e") {
            AgentSelfTest.runEndToEnd()
        }
        if LaunchArguments.value("--extract-eval") != nil {
            ExtractEval.run()
        }
        DemoApp.main()
    }
}
