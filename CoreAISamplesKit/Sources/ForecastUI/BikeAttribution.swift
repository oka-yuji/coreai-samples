import SwiftUI

struct BikeAttribution: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let source = URL(string: "https://doi.org/10.24432/C5F62R") {
                Link(destination: source) {
                    Text("Source: Seoul Bike Sharing Demand · UCI", bundle: .module)
                }
            }
            Text(
                "Seoul Bike Sharing Demand [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5F62R.",
                bundle: .module)
            if let license = URL(string: "https://creativecommons.org/licenses/by/4.0/") {
                Link(destination: license) {
                    Text("Creative Commons Attribution 4.0 International (CC BY 4.0)", bundle: .module)
                }
            }
            Text(
                "Encoding only: Windows-1252 (CP1252) to UTF-8. CRLF line endings, headers, rows, units and values are unchanged. No imputation or aggregation. The bundled sample works offline.",
                bundle: .module)
        }.font(.caption).foregroundStyle(.secondary)
    }
}
