Here is the markdown formatted for your requested changes. This structure is ideal for a Pull Request description, a Jira/Trello ticket, or a developer spec sheet. 

***

### 📋 Feature Updates: Recipe Card & Landing Page UI Improvements

**Overview**
This update optimizes the layout of the recipe cards, focusing on better data visibility in the Nutrient tab, a more compact action button layout, and improved user control on the landing page.

#### 1. Nutrient Tab Layout Updates
* **Disabled Truncation:** Removed text-overflow constraints (`...`) on all nutrient values to ensure full visibility.
* **Maximized Card Space:** Updated the container layout (e.g., using `justify-content: space-between` and flexible grid/flex columns) to ensure nutrient data expands to fill the entire card symmetrically without leaving awkward empty gaps.
* **Responsive Wrapping:** Allowed longer values to wrap to the next line naturally instead of clipping.

#### 2. Action Buttons Redesign
* **Inline Layout:** Placed the **Website**, **Add to Cart**, and **Not Relevant** buttons on a single horizontal row.
* **Compact Styling:** Reduced button padding, scaled down icons (if applicable), and minimized margins between the buttons to save vertical space.
* **Equal Distribution:** Set the button container to a flex row to ensure they are evenly spaced but tightly grouped.

#### 3. Landing Page Functionality
* **Global "Not Relevant" Button:** Added the **Not Relevant** button directly to the main landing page view. 
* **Refresh Action:** Tied the button to a refresh/skip function so users can immediately dismiss a suggested recipe they dislike and generate a new one without having to navigate into the recipe details.

***

### 💻 Developer Notes / CSS Guidance 
*(Optional implementation details for the front-end team)*

* **Nutrient Values:** Ensure `white-space: nowrap` and `text-overflow: ellipsis` are removed from the nutrient value text classes. Use `flex: 1` on the items to evenly distribute space.
* **Button Container:** Use `display: flex; flex-direction: row; gap: 8px;` for the button wrapper to keep them on one line and tightly packed.
* **Landing Page State:** Ensure the `onDismiss` or `fetchNewRecipe` function is correctly passed down to the landing page component for the "Not Relevant" button.
# Frontend Visual Audit And Fix Plan

## Validation

This document was empty before this pass, so there was no existing proposal to
preserve or reconcile. The audit below is based on the provided Search page
screenshot and the current Streamlit implementation.

## Problems Seen In The Screenshot

1. Floating shopping cart is rendered in normal page flow.
   - The cart appears as overlapping text near the upper left of the content
     area instead of behaving like a small fixed utility near Streamlit's top
     toolbar controls.
   - It creates a large empty bordered region before the search controls.
   - Root cause: the cart used a Streamlit `st.popover` inside a bordered
     container, and the CSS selector did not reliably remove that container
     from layout flow.

2. Search toolbar has too much vertical space.
   - The search block is much taller than the controls it contains.
   - There is a large blank band between the toolbar and the "Popular starting
     points" section.
   - Root cause: a hard-coded spacer was added after the sticky toolbar even
     though the toolbar already participates in page layout.

3. Upload control is visually broken.
   - The upload button text overlaps and appears duplicated.
   - The "50MB per file" helper text is exposed inside a very small area.
   - The upload region is too narrow for the default Streamlit uploader markup.

4. Header/cart text overlaps the search area.
   - The cart title/count and toolbar content collide visually.
   - This makes the first viewport feel unfinished and hard to trust.

5. Sticky toolbar separation is heavy and imprecise.
   - The toolbar reads as a floating card sitting in whitespace rather than a
     compact utility strip.
   - It needs a small shadow/bottom border, but not an extra empty block.

6. State freshness risk in cart count.
   - Adding recipe ingredients can mutate shopping-list state after the floating
     cart has already rendered for that run.
   - The badge can show a stale count until another rerun.

7. Shopping List page metrics can be stale after manual add.
   - Metrics were computed before the manual-add form mutation, so a newly added
     item could appear while metrics still showed the old totals.

## Fix Plan

1. Replace the Streamlit popover cart with a fixed HTML details/summary cart.
   - Keep it out of normal document flow.
   - Place it near the top-right Streamlit toolbar controls.
   - Show count, compact preview, remaining total, and a link to the full
     Shopping List page.

2. Remove the sticky toolbar spacer.
   - Let the toolbar occupy only its natural compact height.
   - Keep a subtle bottom border and shadow for separation.

3. Tighten toolbar CSS.
   - Reduce block gaps and padding.
   - Widen the upload column slightly.
   - Hide Streamlit file-uploader helper text inside the compact toolbar.

4. Force shopping-list reruns after add actions.
   - Store a transient notice in session state.
   - Rerun after mutating the shopping list so the floating count updates
     immediately.

5. Move manual-add handling before Shopping List metrics.
   - Rerun after successful manual add so item counts, progress, and list rows
     agree.

6. Verify with focused checks.
   - Compile touched Python files.
   - Run search UI, card, and shopping-list tests.
   - Recheck the Search page visually in browser after restart.

## Implemented In This Pass

- Replaced `st.popover` cart with a fixed HTML cart menu in
  `app/components/floating_cart.py`.
- Removed the hard-coded sticky-toolbar spacer from `app/pages/1_Search.py`.
- Tightened toolbar padding, gaps, upload width, and uploader helper hiding.
- Added rerun-after-add behavior for recipe-card shopping-list additions.
- Moved Shopping List manual-add handling before metrics and reruns after a
  successful add.
- Fixed the main Streamlit entrypoint import path so the app root loads with
  `src/recipe_discovery` available.

## Follow-Up Frontend Polish Requests

These are additional UI fixes to address after the prior frontend pass. They
focus on keeping the search page compact, making utility controls feel native,
and removing visual glitches around the cart and sidebar controls.

### Problems To Address

1. Floating shopping cart placement still conflicts with the top page chrome.
   - The cart button should not appear in the main content flow or create extra
     layout space.
   - Its final position should sit near, or visually overlay, the Streamlit
     Deploy button area in the top-right controls.

2. Shopping cart popup requires a second cart-button click to close.
   - The popup should dismiss naturally when the user clicks anywhere outside
     the open cart panel.
   - This should make the cart behave like a standard dropdown or popover.

3. Sidebar collapse/expand arrow is rendering incorrectly.
   - The double-arrow left/right keyboard control should display as the intended
     sidebar toggle.
   - It should not appear as an intermittent text box or stray text element.

4. Search page has too much whitespace above the search bar.
   - The search controls should sit closer to the top of the page.
   - Any unnecessary top padding, margin, spacer, or empty container above the
     search bar should be removed.

### Fix Plan

1. Reposition the floating shopping cart as a fixed top-right utility control.
   - Keep it outside normal page layout so it does not push the search UI down.
   - Tune `top`, `right`, and `z-index` values so it aligns with Streamlit's
     top toolbar and avoids overlapping core controls in a confusing way.

2. Replace the current cart open/close behavior with click-away dismissal.
   - Use frontend behavior that closes the cart when focus or pointer activity
     moves outside the cart wrapper.
   - Preserve the existing cart count, preview items, and Shopping List link.

3. Repair the sidebar arrow styling/rendering.
   - Target the Streamlit sidebar toggle markup carefully so the intended
     collapse/expand icon is visible.
   - Remove or override any custom CSS that causes the arrow control to render
     as a text-box-like artifact.

4. Tighten the top spacing above the search bar.
   - Audit the Search page header, floating cart mount point, sticky toolbar,
     and any Streamlit block containers for excess top spacing.
   - Remove redundant spacers and reduce padding until the search bar appears
     promptly below the app header.
