import ChronosFeature
import MapKit
import SwiftUI

struct ChronosMap: View {
    @Environment(\.locale) private var locale
    let state: ChronosFeature.State
    let select: (String) -> Void
    @State private var position = MapCameraPosition.region(Self.region)

    private static var region: MKCoordinateRegion {
        MKCoordinateRegion(
            center: CLLocationCoordinate2D(latitude: 40.7515, longitude: -73.988),
            span: MKCoordinateSpan(latitudeDelta: 0.058, longitudeDelta: 0.045))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Manhattan · 20 taxi zones", bundle: .module).font(.headline)
                Spacer()
                Text("Tap a zone", bundle: .module).font(.caption).foregroundStyle(.secondary)
            }
            MapReader { proxy in
                Map(position: $position, interactionModes: [.pan, .zoom, .rotate]) {
                    if let catalog = state.catalog {
                        ForEach(catalog.areas) { area in
                            ForEach(Array((area.polygons ?? []).enumerated()), id: \.offset) { _, shape in
                                MapPolygon(polygon(shape))
                                    .foregroundStyle(ChronosStyle.demand(state.displayedValue(areaID: area.id)))
                                    .stroke(
                                        area.id == state.areaID ? Color.primary : Color.white,
                                        lineWidth: area.id == state.areaID ? 3 : 1)
                            }
                        }
                        if let area = state.selectedArea, let center = area.center {
                            Annotation(
                                ChronosCopy.name(area.name, locale: locale), coordinate: coordinate(center),
                                anchor: .bottom
                            ) {
                                VStack(spacing: 2) {
                                    Text(ChronosCopy.name(area.name, locale: locale)).font(.caption2).lineLimit(2)
                                    Text(ChronosStyle.count(state.displayedValue(areaID: area.id)))
                                        .font(.title3.bold()).monospacedDigit()
                                }
                                .padding(8).frame(maxWidth: 150)
                                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                                .overlay(RoundedRectangle(cornerRadius: 12).stroke(.primary.opacity(0.3)))
                            }
                            .annotationTitles(.hidden)
                        }
                    }
                }
                .mapStyle(.standard(elevation: .flat, emphasis: .muted, pointsOfInterest: .excludingAll))
                .mapControls { MapCompass() }
                .onTapGesture { point in
                    guard let location = proxy.convert(point, from: .local) else { return }
                    let point = ChronosGeoPoint(latitude: location.latitude, longitude: location.longitude)
                    if let area = state.catalog?.areas.first(where: {
                        $0.polygons?.contains { $0.contains(point) } == true
                    }) {
                        select(area.id)
                    }
                }
                .overlay(alignment: .topLeading) {
                    Button {
                        position = .region(Self.region)
                    } label: {
                        Image(systemName: "viewfinder").padding(10)
                    }
                    .background(.regularMaterial, in: Circle())
                    .padding(10)
                    .accessibilityLabel(Text("Show all zones", bundle: .module))
                }

            }
            .frame(height: 310)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            HStack(spacing: 6) {
                Text(
                    LocalizedStringKey(state.forecast == nil ? "Observed trips / 30 min" : "Predicted trips / 30 min"),
                    bundle: .module
                )
                .font(.caption2)
                Spacer(minLength: 4)
                ForEach([0, 50, 100, 200, 500], id: \.self) { value in
                    HStack(spacing: 2) {
                        RoundedRectangle(cornerRadius: 2).fill(ChronosStyle.demand(Float(value))).frame(
                            width: 10, height: 10)
                        Text(value == 500 ? "500+" : String(value)).font(.system(size: 10)).monospacedDigit()
                    }
                }
            }
        }
    }

    private func coordinate(_ point: ChronosGeoPoint) -> CLLocationCoordinate2D {
        CLLocationCoordinate2D(latitude: point.latitude, longitude: point.longitude)
    }

    private func polygon(_ shape: ChronosZonePolygon) -> MKPolygon {
        let outer = shape.outer.map(coordinate)
        let holes = shape.holes.map { ring in
            let points = ring.map(coordinate)
            return MKPolygon(coordinates: points, count: points.count)
        }
        return MKPolygon(coordinates: outer, count: outer.count, interiorPolygons: holes)
    }
}
