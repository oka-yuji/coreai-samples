import Foundation
import OCRCore
import OCRFeature
import OCRRuntime
import Testing

@testable import OCRDemoRuntime

@MainActor
struct OCRDisplayFailureTests {
    @Test("Unlocalized runtime errors become actionable image, model and read messages")
    func messages() throws {
        let url = URL(filePath: "/fixture/image.png")
        #expect(
            OCRRunner.message(OCRError.invalidImage, for: .select(1, url))
                == "The image could not be opened. Choose another image.")
        #expect(
            OCRRunner.message(OCRRuntimeError.invalidBundle("internal detail"), for: .load(1, url))
                == "Model preparation failed. Check the model folder and try again.")
        #expect(
            OCRRunner.message(OCRRuntimeError.invalidOutput("internal detail"), for: .read(1, try pageFixture().input))
                == "The document could not be read. Try another image.")
        #expect(
            OCRRunner.message(OCRRuntimeError.busy, for: .load(1, url))
                == "Another read is still finishing. Try again shortly.")
        let cocoa = CocoaError(.fileReadNoPermission)
        #expect(OCRRunner.message(cocoa, for: .select(1, url)) == cocoa.localizedDescription)
    }
}
