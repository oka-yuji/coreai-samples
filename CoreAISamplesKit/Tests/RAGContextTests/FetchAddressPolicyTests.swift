import Foundation
import Testing

@testable import RAGContext

struct FetchAddressPolicyTests {
    @Test(
        "Non-public IPv4 ranges are refused",
        arguments: [
            [0, 0, 0, 0], [0, 255, 255, 255], [10, 0, 0, 1], [127, 1, 2, 3], [169, 254, 1, 1],
            [172, 16, 0, 0], [172, 31, 255, 255], [192, 168, 1, 1], [100, 64, 0, 0], [100, 127, 255, 255],
            [224, 0, 0, 0], [239, 255, 255, 255], [240, 0, 0, 0], [255, 255, 255, 255],
        ])
    func privateIPv4(_ address: [UInt8]) {
        #expect(!FetchAddressPolicy.isPublic(address))
    }

    @Test(
        "Public IPv4 boundary addresses remain allowed",
        arguments: [
            [1, 0, 0, 0], [8, 8, 8, 8], [100, 63, 255, 255], [100, 128, 0, 0], [169, 253, 255, 255],
            [169, 255, 0, 0], [172, 15, 255, 255], [172, 32, 0, 0], [192, 167, 255, 255], [192, 169, 0, 0],
            [223, 255, 255, 255],
        ])
    func publicIPv4(_ address: [UInt8]) {
        #expect(FetchAddressPolicy.isPublic(address))
    }

    @Test("Non-public IPv6 and mapped IPv4 addresses are refused")
    func privateIPv6() {
        let zero = [UInt8](repeating: 0, count: 16)
        let mapped = Array(zero.prefix(10)) + [255, 255]
        for address in [
            zero, Array(zero.prefix(15)) + [1], [0xfc, 0] + Array(zero.prefix(14)),
            [0xfd, 0xff] + Array(zero.prefix(14)), [0xfe, 0x80] + Array(zero.prefix(14)),
            [0xfe, 0xbf] + Array(zero.prefix(14)), [0xff, 0] + Array(zero.prefix(14)),
            mapped + [127, 0, 0, 1], mapped + [100, 64, 0, 1], mapped + [192, 168, 0, 1],
        ] {
            #expect(!FetchAddressPolicy.isPublic(address))
        }
        #expect(FetchAddressPolicy.isPublic(mapped + [8, 8, 8, 8]))
        #expect(FetchAddressPolicy.isPublic([0x20, 1] + Array(zero.prefix(14))))
        #expect(FetchAddressPolicy.isPublic([0xfe, 0x7f] + Array(zero.prefix(14))))
        #expect(FetchAddressPolicy.isPublic([0xfe, 0xc0] + Array(zero.prefix(14))))
    }

    @Test("Every resolved address must be public and resolution must not be empty")
    func allAddresses() {
        #expect(FetchAddressPolicy.permits([[8, 8, 8, 8], [1, 1, 1, 1]]))
        #expect(!FetchAddressPolicy.permits([[8, 8, 8, 8], [10, 0, 0, 1]]))
        #expect(!FetchAddressPolicy.permits([]))
        #expect(!FetchAddressPolicy.permits([[1, 2, 3]]))
    }

    @Test("Local hostname suffixes are refused with case and trailing dots")
    func localNames() {
        for name in ["localhost", "localhost.", "sub.LOCALHOST.", "printer.LOCAL.", ""] {
            #expect(FetchAddressPolicy.isLocalName(name))
        }
        for name in ["example.com", "notlocalhost", "local.example.com"] {
            #expect(!FetchAddressPolicy.isLocalName(name))
        }
    }

    @Test("Subresource filters match local literals without blocking ordinary public URLs")
    func contentFilters() throws {
        let filters = try FetchAddressPolicy.subresourceFilters.map {
            try NSRegularExpression(pattern: $0, options: [.caseInsensitive])
        }
        func blocked(_ url: String) -> Bool {
            filters.contains { $0.firstMatch(in: url, range: NSRange(url.startIndex..., in: url)) != nil }
        }
        for host in [
            "localhost.", "sub.local", "127.0.0.1", "0.0.0.0", "10.1.2.3", "172.16.0.1", "172.31.0.1",
            "169.254.1.1", "192.168.1.1", "100.64.0.1", "100.127.255.255", "224.0.0.1", "255.255.255.255",
            "[::]", "[::1]", "[fc00::1]", "[fe80::1]", "[febf::1]", "[ff02::1]", "[::ffff:7f00:1]",
            "[::ffff:a00:1]", "[::ffff:a9fe:1]", "[::ffff:ac10:1]", "[::ffff:c0a8:1]", "[::ffff:6440:1]",
            "[::ffff:f000:1]", "[::ffff:127.0.0.1]", "[fe80::1%25lo0]",
        ] {
            #expect(blocked("https://\(host):8443/path"), "\(host)")
            #expect(blocked("ws://\(host)/path"), "\(host)")
        }
        for host in [
            "example.com", "developer.apple.com", "1.1.1.1", "100.128.0.1", "[2606:4700::1111]", "[::ffff:808:808]",
        ] {
            #expect(!blocked("https://\(host)/path"), "\(host)")
        }
        let rules = try #require(
            JSONSerialization.jsonObject(with: Data(FetchAddressPolicy.contentRules().utf8)) as? [[String: Any]])
        #expect(rules.count == filters.count)
    }
}
