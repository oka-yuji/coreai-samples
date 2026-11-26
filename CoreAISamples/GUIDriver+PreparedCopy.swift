import CoreAIBackend
import Foundation
import ModelLibraryStorage

extension GUIDriver {
    func checkPreparedCopy(
        _ scenario: String, url: URL, direct: Bool, chat: ChatViewModel, library: MacModelLibrary,
        choice: MacModelChoice
    ) async {
        guard ["recover", "terminal"].contains(scenario),
            scenario != "terminal" || LaunchArguments.flag("--clear-prepared")
        else {
            await step(2, "arguments", "Check the prepared copy diagnostic arguments") {
                (false, "Choose recover or terminal. Clearing a terminal copy requires --clear-prepared.")
            }
            return
        }
        if scenario == "recover" {
            let wasPrepared = ModelStorage.prepared.contains(ModelStorage.key(url))
            async let opening: Void = openModel(url, direct: direct, chat: chat, library: library, choice: choice)
            await step(2, "preparing", "Chat shows preparation while a known copy is being repaired") {
                let observed = await self.wait(seconds: 30) { chat.phase == .loading(preparing: true) }
                return (wasPrepared && observed, "previouslyPrepared=\(wasPrepared) phase=\(chat.phase)")
            }
            await opening
            await step(3, "recovered", "Chat loaded the repaired copy") {
                let ready = await self.wait(seconds: 900) { chat.phase == .ready }
                let recovered = chat.loadStats?.recoveredPreparedCopy == true
                return (ready && recovered, "phase=\(chat.phase) recovered=\(recovered)")
            }
            return
        }
        await openModel(url, direct: direct, chat: chat, library: library, choice: choice)
        let expected = ChatViewModel.preparedCopyMessage(for: PreparedCopyError.recoveryFailed)
        let failed = await wait(seconds: 60) {
            guard case .failed(let message) = chat.phase else { return false }
            return message == expected && (direct || (!choice.state.selecting && !library.isInUse(.qwen)))
        }
        await step(2, "terminal", "Chat explains how to clear the prepared copy") {
            (failed, "phase=\(chat.phase)")
        }
        guard failed else { return }
        await step(3, "ready-after-clear", "Clearing the prepared copy lets Chat prepare it again") {
            do {
                try MacModelLibrary.clearPrepared(url, artifact: LanguageModelArtifacts.qwen())
                await self.openModel(url, direct: direct, chat: chat, library: library, choice: choice)
                let ready = await self.wait(seconds: 900) { chat.phase == .ready }
                let prepared = chat.loadStats?.cached == false && chat.loadStats?.recoveredPreparedCopy == false
                return (ready && prepared, "phase=\(chat.phase) prepared=\(prepared)")
            } catch { return (false, error.localizedDescription) }
        }
    }
}
