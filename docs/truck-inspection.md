# Truck daily inspection input

The Japanese truck demo records a person's inspection results. It uses GLiNER2.5-multi for one question at a time, and keeps model suggestions pending until the person confirms them. It does not diagnose a vehicle or determine whether it may operate.

## Checklist and sources

The bundled catalog follows the daily inspection section of the [MLIT vehicle inspection and maintenance guide](https://www.mlit.go.jp/jidosha/anzen/03safety/tebiki.html), checked on 2026-10-15. The [MLIT business vehicle overview](https://www.mlit.go.jp/jidosha/anzen/03safety/inspection.html) and the [Japan Trucking Association handbook](https://jta.or.jp/wp-content/themes/jta_theme/pdf/anzen/kotsuanzen_ichiran/tenkenseibi_handbook202406.pdf) provide context.

The catalog has 23 questions. The hydraulic brake example selects 18, and the large air brake example selects 21. These are example equipment profiles, not a classification of every truck. The vehicle manufacturer's instructions determine the actual equipment, procedure and specified values. The model does not impose a common tire pressure, air pressure, oil quantity or service limit.

| Area | Questions |
| --- | --- |
| Previous operation | Previously noticed problems |
| Cab | Brake pedal travel and effectiveness (hydraulic), parking brake, engine starting and noise, low-speed and acceleration behavior, washer spray, wiper operation |
| Engine compartment | Washer fluid, brake fluid (hydraulic), battery fluid, coolant, engine oil, fan belt |
| Around the vehicle | Lamps and indicators, tire pressure, tire damage and foreign objects, abnormal tire wear, tread depth, wheel mounting (large-truck example) |
| Air brakes | Air pressure rise, brake valve exhaust, air tank condensate, chamber rod travel and drum/lining clearance |

Eleven questions retain the guide's condition-based timing note. They may be checked at an appropriate interval based on distance and operating conditions; being shown in the demo does not mean every such check is due on every trip. A record can explicitly say not inspected or not applicable. Neither is a normal result. The UI must keep unanswered items distinct from both.

Questions, Japanese answer aliases, confirmation vocabulary and source metadata live in `TruckInspectionRuntime/Resources/truck-inspection-ja.json`. Pure data types and interpretation live in `TruckInspectionCore`. Loading bundled resources is a runtime responsibility.

## Answer contract

An explicit button choice or exact item-specific alias precedes model classification. Punctuation, whitespace and character width are normalized for exact matching. Other short answers are sent to `GLiNERExtractor.classify` with the current item's task and the catalog's labels. An empty answer never goes to the model. Answers exceeding 240 characters must be shortened.

A model score is not a calibrated probability of correctness. Every model candidate requires explicit confirmation, including scores above 0.85. This differs from the earlier equipment demo because these records concern vehicle inspections. Invalid scores, an incorrect result count, all-zero scores and tied winners are rejected.

Confirmation uses only complete phrases from the vocabulary, with negative phrases checked first. Ambiguous, contradictory and empty responses remain unknown. Repeated uncertainty leaves the candidate unconfirmed.

An empty final speech result uses the last nonempty partial result. The listening clock allows up to six seconds before ending audio; the silence interval begins after the first changed nonempty partial. After audio ends, the final wait is limited to two seconds. A missing or empty transcript is never a normal inspection result.

## Verification ownership

The implementation work is tracked by issues #120, #121 and #122 under #123. The current user's 2026-10-15 instruction changes the earlier six-field equipment handoff into this truck demo and assigns iPhone validation to the user. Device speed, memory, model accuracy, offline speech availability and Bluetooth operation remain unverified until the user's runs are recorded.
