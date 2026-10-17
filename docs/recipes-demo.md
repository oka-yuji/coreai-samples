# Try the Recipes demo

This Mac demo reads a fridge photograph, proposes ingredients for you to review, and checks a recipe URL against your confirmed inventory and kitchen equipment. It uses Apple's on-device Foundation Models for the photo and the existing Clef-Flash 9B decision bundle for recipe requirements. Quantities are compared in code.

## Start

Download the [Clef-Flash decision bundle](https://huggingface.co/okayuji/clef-flash-9b-coreai-decide) from the repository root with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```sh
hf download okayuji/clef-flash-9b-coreai-decide --local-dir ./models/clef-flash-9b-decide-v1
```

The model payload is 7,278,548,313 bytes, with small documentation and example files in addition. The [English model guide](clef-flash.md) and [日本語ガイド](clef-flash.ja.md) describe the measured environment, limits and license. Use this bundle with Recipes or Decide through `DecisionModel`.

1. Open `CoreAISamples.xcodeproj`, choose the **CoreAISamples** scheme and **My Mac**, set the Run build configuration to **Release**, and Run. This requires macOS 27, Xcode 27 and Apple Intelligence with vision available on the Mac.
2. Select the **Clef-Flash 9B** sidebar entry whose description is **Can you make this recipe with what you have?**. The **EN / JP** control at the top right changes the interface immediately on all six Mac screens and remembers the choice. Original recipe text and model-generated content keep their source language.
3. The model card above the workflow always names `clef-flash-9b-decide-v1`. Use **Open Clef-Flash bundle** to choose that folder. It shows loading and loaded states; **Retry saved bundle** retries a remembered path when no model is loaded. A successful selection is remembered. `COREAI_DECISION_BUNDLE` can also supply the path when launching the app.
4. Choose **Examples → Three-egg omelette**. This reads the bundled one-egg photo and fetches its recipe. **Open photo** accepts your own image instead.
5. Check the proposed ingredient names and amounts. For the egg, confirm that the amount is `1 pcs`, then check its confirmation box. You can add pantry ingredients using **Add** beside a recipe ingredient and enter their amounts.
6. Click **Check recipe**. For this recipe's original two servings, one confirmed egg leaves **Short by 2.00 pcs**. Other ingredients and equipment can remain unchecked; they are shown as needing review.

The five supplied images are bundled, so importing them separately is optional. Example selection always runs photo recognition and fetches the live page. There are no canned recognition results or offline recipe fallbacks.

The downloaded Clef model does not read the photo. Apple's on-device model proposes the foods, Clef reads the recipe text to classify requirements and equipment, and Swift code compares confirmed quantities. Downloading the model and fetching recipe URLs need internet access. Photo recognition and decision inference run on the Mac, without sending the photo or inventory to the recipe site.

## Review amounts and equipment

- Suggested counts come only from visible, countable foods. Check them yourself. The app never estimates grams or liquid volume from a photograph.
- Enter a quantity in the recipe's units, or explicitly enter `enough` when you have checked that you have enough for the whole recipe. Changing the recipe clears confirmation of `enough`; numeric inventory is retained. Different units remain unknown.
- **Confirm listed items** confirms the visible list after you have reviewed it. Editing a name or amount clears that item's confirmation.
- Check **I have checked the complete inventory…** only after checking ingredients outside the photo too. Otherwise, an unlisted ingredient means unknown rather than absent.
- Expand **Check kitchen equipment** and set each relevant tool to **Have it**, **Do not have it** or **Not checked**.
- Any inventory, recipe or equipment change clears the old report. **Stop** waits for the in-flight operation to end before another can start, including if you leave and reopen Recipes.

## Five examples

| Example | Recipe | What to check |
|---|---|---|
| Chicken & egg rice bowl | [Oyakodon](https://www.kikkoman.co.jp/homecook/search/recipe/00006446/) | The herb and unlabeled bottle need human identification. Explicitly add rice, water and confirmed seasoning. |
| Ginger pork | [Ginger pork](https://www.kikkoman.co.jp/homecook/search/recipe/00056818/) | After confirming the full inventory without ginger, ginger is missing. Carrots and cucumbers do not replace it. |
| Three-egg omelette | [Omelette](https://www.kikkoman.co.jp/homecook/search/recipe/00004002/) | One confirmed egg for a three-egg recipe is two eggs short. |
| Macaroni gratin | [Gratin](https://www.kikkoman.co.jp/homecook/search/recipe/00004825/) | With the basic tools available, set both oven and toaster unavailable, check, then set toaster available and check again. The recipe permits either appliance. |
| Tomato & mozzarella | [Caprese](https://www.kikkoman.co.jp/homecook/search/recipe/00001454/) | The opaque container does not confirm mozzarella. Leave full-inventory confirmation off until its contents have been checked. |

The recipe is not scaled to a different number of servings. For controlled checks of equipment or a single shortage, explicitly enter and confirm the remaining required ingredients; do not infer them from the photograph.

## Inspect the decision

Expand **Developer details** at the top of the recipe column. It separates four stages:

- **Photo:** Apple's `SystemLanguageModel.default` proposes ingredients on this Mac. The panel shows actual photo-processing time and candidate count.
- **URL:** The parser extracts the source ingredients and cooking steps. These counts come from the loaded recipe.
- **Clef:** The decision model classifies ingredient requirements and kitchen equipment. After checking a recipe, the panel shows the result's model and version, input tokens, reused document tokens and check time. Select an ingredient or equipment question to inspect every returned choice, probability, logit and selected result; expand the question to read its original instructions.
- **Code:** Confirmed inventory names and quantities are matched in code. The ordinary result shows missing ingredients and calculated shortages. The panel displays the same `0.60` review threshold used by the report, alongside the other unknown-status conditions.

Check time includes document preparation and result mapping, but excludes the explicit bundle-opening step and queueing. Internal runtime preparation during the check is included. Input tokens include the recipe and all questions. Reuse is zero on the first check and can increase when you change only kitchen equipment and check the same recipe again. Choice probabilities are normalized over the offered choices; they are not calibrated accuracy or evidence that an ingredient is in the photograph.

Opening the panel, selecting a question or changing EN/JP does not run the model again. Check that the recorded time and scores stay the same after switching languages. Editing actual inventory data still clears its confirmation and invalidates the old report.

## Boundaries

Photo and inventory processing stays on the Mac. Loading a recipe performs a public HTTP request, without sending the image or inventory. Version 1 accepts only `https://www.kikkoman.co.jp/homecook/search/recipe/<eight digits>/` and validates redirects. Unsupported sites, unavailable Apple Intelligence and network failures are reported as errors.

In the supplied ginger-pork example, the photo model can propose `zucchini` for the intended cucumbers and `meat` for pork. Correct names before confirming them; a successful demonstration does not mean every label was recognized correctly.

The original Japanese recipe remains visible with its source link. Model scores describe interpretation of a requirement and are uncalibrated; they are not confidence that an ingredient is physically present. Unclear requirements and unverified amounts remain review items. The supplied images are synthetic examples, not a benchmark of real-fridge accuracy.

## Repeat the automated scenario

First acquire the repository's GPU lock and make sure no other model job or demo app is running, as described in `CLAUDE.md`. Then run the built executable:

```sh
COREAI_DECISION_BUNDLE=/path/to/clef-flash-9b-decide-v1 /path/to/CoreAISamples.app/Contents/MacOS/CoreAISamples --recipe-e2e /tmp/recipe-results
```

This uses the same Store, runner and live services as the screen. `five-store-cases.json` keeps raw photo suggestions separate from the explicitly scripted reviewed inventory. `toaster-store-case.json` records the changed appliance condition. These scripted inventories are test inputs, not automatic recognition claims.

The acceptance contract and verification status are in [the feature specification](specs/fridge-recipe-demo.md); measured results and unsuccessful approaches are in [the research record](research/fridge-photo-recognition.md).
