import ModelLibraryFeature
import SwiftUI

@MainActor
public protocol ModelOnDemandDemoAdapter: ModelDemoAdapter {
    func preview(for selection: DemoSelection, access: OnDemandModelAccess?) -> AnyView
}
