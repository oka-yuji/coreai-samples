import re


def pattern(template, dotall=False):
    seen, parts = set(), re.split(r"\{(\w+)\}", template)
    regex = []
    for index, part in enumerate(parts):
        if index % 2 == 0:
            regex.append(re.escape(part))
        elif part in seen:
            regex.append(f"(?P={part})")
        else:
            seen.add(part)
            regex.append(f"(?P<{part}>.+?)")
    return re.compile("".join(regex), re.DOTALL if dotall else 0)


def fields(template, text, dotall=False):
    found = pattern(template, dotall).fullmatch(text)
    if found is None:
        raise ValueError("Text does not follow the template")
    return found.groupdict()


def listed(value, part):
    return [] if value == part["empty"] else value.split(part["join"])


def solve_extract(content, prompts):
    spec = prompts["synth-extract"]
    found = fields(spec["instruction"], content, dotall=True)
    name = next(name for name, part in spec["templates"].items() if part["schema"] == found["schema"])
    part, keys = spec["templates"][name], spec["templates"][name]["keys"]
    rows = []
    for record in found["records"].split(part["separator"]):
        values = fields(part["record"], record)
        if name == "orders":
            goods = [fields(part["item"], piece) for piece in listed(values["goods"], part)]
            rows.append({keys["number"]: values["number"], keys["customer"]: values["customer"],
                         keys["place"]: values["place"],
                         keys["goods"]: [{keys["name"]: good["name"], keys["quantity"]: int(good["quantity"])}
                                         for good in goods]})
        else:
            rows.append({keys["team"]: values["team"], keys["leader"]: values["leader"],
                         keys["members"]: listed(values["members"], part), keys["areas"]: listed(values["areas"], part)})
    return {keys["root"]: rows}


def solve_long(content, prompts):
    spec = prompts["synth-long"]
    found = fields(spec["instruction"], content, dotall=True)
    lines = found["records"].split(spec["separator"])
    for name, part in spec["templates"].items():
        try:
            target = fields(part["question"], found["question"])
        except ValueError:
            continue
        if name == "stock":
            total = int(target["start"])
            for line in lines:
                for kind, sign in (("in", 1), ("out", -1)):
                    try:
                        values = fields(part[kind], line)
                    except ValueError:
                        continue
                    if (values["warehouse"], values["goods"]) == (target["warehouse"], target["goods"]):
                        total += sign * int(values["quantity"])
                    break
                else:
                    raise ValueError("Unknown record")
            return total
        total = 0
        for line in lines:
            values = fields(part["record"], line)
            if (values["person"], values["place"], values["goods"]) == (target["person"], target["place"], target["goods"]):
                total += int(values["quantity"])
        return total
    raise ValueError("Unknown question")


def solve_format(content, prompts):
    spec = prompts["synth-format"]
    found = fields(spec["instruction"], content, dotall=True)
    lines = found["rules"].split(spec["separator"])
    if len(lines) != 5 or lines[4] != spec["rules"]["only"]:
        raise ValueError("Unexpected rules")
    items = fields(spec["rules"]["items"], lines[0])
    limit = fields(spec["rules"]["max_chars"], lines[1])
    style = next(style for style in spec["styles"].values() if style["rule"] == lines[2])
    try:
        words = fields(spec["rules"]["required_two"], lines[3])
        required = [words["first"], words["second"]]
    except ValueError:
        required = [fields(spec["rules"]["required_one"], lines[3])["first"]]
    if items["bullet"] != limit["bullet"]:
        raise ValueError("Bullet differs between rules")
    return {"kind": "format_rules", "bullet": items["bullet"], "items": int(items["items"]),
            "max_chars": int(limit["max_chars"]), "endings": style["endings"], "required": required}
