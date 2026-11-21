import random


def extract(rng, spec, vocabulary, settings, template):
    part = spec["templates"][template]
    keys = part["keys"]
    count = rng.randint(*settings["records"])
    limit = settings["list_items"]
    empty = rng.randrange(count)
    lines, rows = [], []
    if template == "orders":
        numbers = rng.sample(range(100, 1000), count)
        people = rng.sample(vocabulary["people"], count)
        for index in range(count):
            goods = [] if index == empty else rng.sample(vocabulary["goods"], rng.randint(1, limit))
            pairs = [(name, rng.randint(*settings["quantity"])) for name in goods]
            place = rng.choice(vocabulary["places"])
            number = f"K-{numbers[index]}"
            text = part["join"].join(part["item"].format(name=name, quantity=quantity) for name, quantity in pairs)
            lines.append(part["record"].format(number=number, customer=people[index], place=place,
                                               goods=text or part["empty"]))
            rows.append({keys["number"]: number, keys["customer"]: people[index], keys["place"]: place,
                         keys["goods"]: [{keys["name"]: name, keys["quantity"]: quantity} for name, quantity in pairs]})
        used = people
    else:
        teams = rng.sample(vocabulary["teams"], count)
        people = rng.sample(vocabulary["people"], count * (1 + limit))
        blank = rng.choice(["members", "areas"])
        for index in range(count):
            pool = people[count + index * limit:count + (index + 1) * limit]
            members = [] if (index, blank) == (empty, "members") else pool[:rng.randint(1, limit)]
            areas = [] if (index, blank) == (empty, "areas") else rng.sample(vocabulary["places"], rng.randint(1, limit))
            lines.append(part["record"].format(team=teams[index], leader=people[index],
                                               members=part["join"].join(members) or part["empty"],
                                               areas=part["join"].join(areas) or part["empty"]))
            rows.append({keys["team"]: teams[index], keys["leader"]: people[index], keys["members"]: members,
                         keys["areas"]: areas})
        used = people[:count] + [name for row in rows for name in row[keys["members"]]]
    content = spec["instruction"].format(schema=part["schema"], records=part["separator"].join(lines))
    return content, {keys["root"]: rows}, {"kind": "json_exact"}, [template, *sorted(used)]


def long(rng, spec, vocabulary, settings, template, records):
    part = spec["templates"][template]
    count = rng.randint(*settings["relevant"])
    if template == "stock":
        target = (rng.choice(vocabulary["warehouses"]), rng.choice(vocabulary["goods"]))
        start = rng.randint(*settings["start"])
        answer, relevant = start, []
        for _ in range(count):
            if answer == 0 or rng.random() < 0.5:
                quantity, kind = rng.randint(*settings["quantity"]), "in"
                answer += quantity
            else:
                quantity, kind = rng.randint(1, answer), "out"
                answer -= quantity
            relevant.append(part[kind].format(warehouse=target[0], goods=target[1], quantity=quantity))
        question = part["question"].format(warehouse=target[0], goods=target[1], start=start)
        pools = (vocabulary["warehouses"], vocabulary["goods"])
    else:
        target = (rng.choice(vocabulary["people"]), rng.choice(vocabulary["places"]), rng.choice(vocabulary["goods"]))
        quantities = [rng.randint(*settings["quantity"]) for _ in range(count)]
        relevant = [part["record"].format(person=target[0], place=target[1], goods=target[2], quantity=quantity)
                    for quantity in quantities]
        answer = sum(quantities)
        question = part["question"].format(person=target[0], place=target[1], goods=target[2])
        pools = (vocabulary["people"], vocabulary["places"], vocabulary["goods"])
    others = []
    while len(others) < records - count:
        choice = tuple(rng.choice(pool) for pool in pools)
        if choice == target:
            continue
        quantity = rng.randint(*settings["quantity"])
        if template == "stock":
            others.append(part[rng.choice(["in", "out"])].format(warehouse=choice[0], goods=choice[1], quantity=quantity))
        else:
            others.append(part["record"].format(person=choice[0], place=choice[1], goods=choice[2], quantity=quantity))
    positions = {round((0.05 + 0.9 * index / (count - 1)) * (records - 1)): index for index in range(count)}
    if len(positions) != count:
        raise ValueError("Too few records to separate the relevant records")
    filler = iter(others)
    lines = [relevant[positions[index]] if index in positions else next(filler) for index in range(records)]
    content = spec["instruction"].format(records=spec["separator"].join(lines), question=question)
    return content, answer, {"kind": "final_integer"}, [template, *target]


def rules(rng, spec, vocabulary, settings):
    topic = rng.choice(vocabulary["topics"])
    items = rng.choice(settings["items"])
    limit = rng.choice(settings["max_chars"])
    style = rng.choice(sorted(spec["styles"]))
    words = rng.sample(vocabulary["words"], rng.choice(settings["required_words"]))
    bullet, text = spec["bullet"], spec["rules"]
    required = (text["required_one"].format(first=words[0]) if len(words) == 1
                else text["required_two"].format(first=words[0], second=words[1]))
    lines = [text["items"].format(items=items, bullet=bullet), text["max_chars"].format(bullet=bullet, max_chars=limit),
             spec["styles"][style]["rule"], required, text["only"]]
    content = spec["instruction"].format(topic=topic, rules=spec["separator"].join(lines))
    scorer = {"kind": "format_rules", "bullet": bullet, "items": items, "max_chars": limit,
              "endings": spec["styles"][style]["endings"], "required": words}
    return content, None, scorer, [topic, *sorted(words)]


def templates(spec):
    return sorted(spec["templates"]) if "templates" in spec else ["rules"]


def generate(category, index, seed, prompts, vocabulary, records=None):
    settings = category["generator"]
    spec = prompts[settings["prompt"]]
    rng = random.Random(f"{seed}:{category['id']}:{index}")
    template = templates(spec)[index % len(templates(spec))]
    if settings["kind"] == "extract":
        content, answer, scorer, combination = extract(rng, spec, vocabulary, settings, template)
    elif settings["kind"] == "long":
        content, answer, scorer, combination = long(rng, spec, vocabulary, settings, template, records)
    elif settings["kind"] == "format":
        content, answer, scorer, combination = rules(rng, spec, vocabulary, settings)
    else:
        raise ValueError(f"Unknown generator: {settings['kind']}")
    return {"content": content, "answer": answer, "scorer": scorer, "template": template, "combination": combination}
