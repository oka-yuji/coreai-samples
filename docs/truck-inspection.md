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

## iPhone screen

Open **Truck inspection** under the **GLiNER2.5-multi** section of the iPhone app's Demos menu, or launch it with `--truck-inspection`. Choose the equipment profile before starting. Answer buttons work without a model. A free-form answer outside the exact vocabulary needs a model folder in Documents: `contract.json`, one `.aimodel`, and `tokenizer.json` either at the top level or under `tokenizer/`. The app enumerates folders at launch, so relaunch after copying one. The model section lists the available folders; the first is prepared when the screen opens.

The model section reports load milliseconds and the requested compute preference. The app default is GPU; `--compute-units default` requests automatic placement. This label is not evidence of the hardware actually used. Inspect Core AI Instruments on the device to verify execution placement.

Use **Stop** before editing recorded results or changing configuration. Cleanup completes before a new operation starts. The screen releases its model when closed. Returning during cleanup waits for the old operation before preparing the model again. A rejected candidate leaves the same item unanswered. Empty or failed input preserves existing records.

The inspection is kept in memory while the app runs. Share the record before terminating the app. Unanswered, not inspected, not applicable and attention-needed results remain distinct. Diagnostics retain up to 1,000 events, including `decided:` and `timing:` entries. Use **Copy or share diagnostics** and choose Copy in the system share sheet.

## Japanese voice input

From the Demos menu, open **GLiNER2.5-multi → Truck inspection**. After the model is ready, start Japanese voice input. Speech recognition and microphone permissions are requested before the first question. Recognition requires a recognizer whose actual locale is Japanese (Japan), supports on-device recognition, and is currently available. The app never requests a server fallback. An installed Japanese system voice reads the questions.

Questions, answers, confirmations and acknowledgments use the same feature state as tap input. Every model suggestion is confirmed. Explicit choices are repeated before moving on. Two empty answers, two ambiguous confirmations, or two consecutive playback deadlines return to buttons without fabricating a result. Falling back during an acknowledgment selects the next unanswered item and preserves the result already recorded.

The Store owns tasks. The runner waits for every cancelled operation before acknowledging stop; the audio actor owns the engine and session, and a per-listen synchronized owner keeps recognition requests and callbacks separate. The audio session is configured once per voice run. Bluetooth routing waits for the selected input and falls back to the built-in input/speaker if it cannot settle. Recording uses the iOS 27 throwing audio tap API and validates the input format first. The app stops on backgrounding, not on the inactive state used by permission dialogs.

Diagnostics include permissions, selected voice, route for each listen, RMS range, partial and raw final transcripts, final resolution, decisions and timing. The **Copy or share diagnostics** action uses the system share sheet's Copy action. Physical audibility, recognition quality, Bluetooth routing and offline operation require the user's device checks.

## Device measurements

With a model ready and the inspection stopped, expand Model and diagnostics and choose **Measure 20 classifications**. The runner classifies the same short tire-pressure answer once to warm up and then 20 measured times. It bypasses the exact-answer shortcut and leaves inspection records unchanged. Median is the average of sorted samples 10 and 11; p95 uses nearest rank (sample 19). The warmup is reported separately. These timings are inference measurements, not correctness scores.

Logs report requested compute preference, actual asset float type, revision, OS, device, thermal state and low-power mode. Memory entries distinguish resident bytes, peak resident bytes, current footprint and lifetime peak footprint. Available process memory is reported on iOS. Peaks cover the process lifetime, so use a fresh app launch for a clean run. Memory queries occur outside the timed inference interval. The runtime's actual hardware placement still needs Instruments verification.
