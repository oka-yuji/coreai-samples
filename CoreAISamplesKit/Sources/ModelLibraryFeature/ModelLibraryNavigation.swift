public enum ModelLibraryNavigation {
    public enum Column: Equatable, Sendable {
        case sidebar, detail
    }

    public enum Intent: Equatable, Sendable {
        case select(DemoSelection)
        case showSidebar
        case synchronize
    }

    public struct Update: Equatable, Sendable {
        public let column: Column
        public let destination: ModelDestination?
    }

    public static func resolve(_ intent: Intent, state: ModelLibraryState) -> Update {
        if state.pendingDestination != nil { return Update(column: .detail, destination: nil) }
        switch intent {
        case .select(let selection):
            return Update(column: .detail, destination: .demo(selection))
        case .showSidebar:
            return Update(
                column: state.selection == nil ? .sidebar : .detail,
                destination: state.selection == nil ? nil : .library)
        case .synchronize:
            return Update(column: state.selection == nil ? .sidebar : .detail, destination: nil)
        }
    }
}
