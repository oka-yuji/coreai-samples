import Foundation

public enum FetchAddressPolicy {
    public static func isLocalName(_ host: String) -> Bool {
        let name = host.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "."))
        return name.isEmpty || name == "localhost" || name.hasSuffix(".localhost") || name.hasSuffix(".local")
    }

    public static func permits(_ addresses: [[UInt8]]) -> Bool {
        !addresses.isEmpty && addresses.allSatisfy(isPublic)
    }

    public static func isPublic(_ bytes: [UInt8]) -> Bool {
        if bytes.count == 4 {
            switch (bytes[0], bytes[1]) {
            case (0, _), (10, _), (127, _), (169, 254), (172, 16...31), (192, 168), (100, 64...127), (224...255, _):
                return false
            default:
                return true
            }
        }
        guard bytes.count == 16 else { return false }
        if bytes.prefix(10).allSatisfy({ $0 == 0 }), bytes[10] == 255, bytes[11] == 255 {
            return isPublic(Array(bytes.suffix(4)))
        }
        return !bytes.allSatisfy { $0 == 0 }
            && !(bytes.prefix(15).allSatisfy { $0 == 0 } && bytes[15] == 1)
            && bytes[0] & 0xfe != 0xfc
            && !(bytes[0] == 0xfe && bytes[1] & 0xc0 == 0x80)
            && bytes[0] != 0xff
    }

    public static let subresourceFilters: [String] = {
        let ipv4 = [
            #"0\.[0-9.]+"#, #"10\.[0-9.]+"#, #"127\.[0-9.]+"#, #"169\.254\.[0-9.]+"#,
            #"172\.1[6-9]\.[0-9.]+"#, #"172\.2[0-9]\.[0-9.]+"#, #"172\.3[01]\.[0-9.]+"#,
            #"192\.168\.[0-9.]+"#, #"100\.6[4-9]\.[0-9.]+"#, #"100\.[7-9][0-9]\.[0-9.]+"#,
            #"100\.1[01][0-9]\.[0-9.]+"#, #"100\.12[0-7]\.[0-9.]+"#,
            #"22[4-9]\.[0-9.]+"#, #"2[34][0-9]\.[0-9.]+"#, #"25[0-5]\.[0-9.]+"#,
        ]
        let mapped = [
            #"[0-9a-f][0-9a-f]?:[0-9a-f]+"#, #"a[0-9a-f][0-9a-f]:[0-9a-f]+"#,
            #"7f[0-9a-f][0-9a-f]:[0-9a-f]+"#, #"a9fe:[0-9a-f]+"#, #"ac1[0-9a-f]:[0-9a-f]+"#,
            #"c0a8:[0-9a-f]+"#, #"64[4-7][0-9a-f]:[0-9a-f]+"#, #"[ef][0-9a-f][0-9a-f][0-9a-f]:[0-9a-f]+"#,
        ]
        let ipv6 = ["::", "::1", "f[cd][0-9a-f:]*", "fe[89ab][0-9a-f:]*", "ff[0-9a-f:]*"]
        let hosts =
            [#"([^/:]+\.)*localhost\.?"#, #"([^/:]+\.)+local\.?"#]
            + ipv4.map { $0 + #"\.?"# }
            + ipv6.map { #"\["# + $0 + #"(%[a-z0-9._-]+)?\]"# }
            + (ipv4 + mapped).map { #"\[::ffff:"# + $0 + #"(%[a-z0-9._-]+)?\]"# }
        return hosts.map { #"^[a-z]+://([^/@]*@)?"# + $0 + "[/:]" }
    }()

    public static func contentRules() throws -> String {
        let rules = subresourceFilters.map { filter in
            ["trigger": ["url-filter": filter], "action": ["type": "block"]]
        }
        return String(
            decoding: try JSONSerialization.data(withJSONObject: rules, options: [.sortedKeys]), as: UTF8.self)
    }
}
