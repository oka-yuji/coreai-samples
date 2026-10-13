# Try the Recipes demo

This Mac demo reads a fridge photograph, proposes ingredients for you to review, and checks a recipe URL against your confirmed inventory and kitchen equipment. It uses Apple's on-device Foundation Models for the photo and the existing Clef-Flash 9B decision bundle for recipe requirements. Quantities are compared in code.

## Start

1. Open `CoreAISamples.xcodeproj`, choose **CoreAISamples / My Mac**, and Run. This requires macOS 27, Xcode 27 and Apple Intelligence with vision available on the Mac.
2. Select **Recipes** in the sidebar.
3. Use **Open Clef-Flash bundle** to choose the `clef-flash-9b-decide-v1` folder. A successful selection is remembered. `COREAI_DECISION_BUNDLE` can also supply the path when launching the app.
4. Choose **Examples → Three-egg omelette**. This reads the bundled one-egg photo and fetches its recipe. **Open photo** accepts your own image instead.
5. Check the proposed ingredient names and amounts. For the egg, confirm that the amount is `1 pcs`, then check its confirmation box. You can add pantry ingredients using **Add** beside a recipe ingredient and enter their amounts.
6. Click **Check recipe**. For this recipe's original two servings, one confirmed egg leaves **Short by 2.00 pcs**. Other ingredients and equipment can remain unchecked; they are shown as needing review.

The five supplied images are bundled, so importing them separately is optional. Example selection always runs photo recognition and fetches the live page. There are no canned recognition results or offline recipe fallbacks.

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
