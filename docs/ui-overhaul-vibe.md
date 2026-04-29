# UI Overhaul Notes - Search Page Compact Toolbar

Implemented changes:

- Removed the large `Today's Menu / Search Recipes` hero block from
  `app/pages/1_Search.py`.
- Replaced it with a compact sticky search toolbar at the top of the page.
- Kept the query input, photo upload, and search button in one dense row so
  recipe results get more vertical space.
- Made the toolbar sticky so it stays available while scrolling through
  results.
- Switched the toolbar to fixed positioning after Streamlit's nested layout
  prevented reliable sticky behavior.
- Formalized the toolbar/results separation with a flush top header, bottom
  divider, and spacer shadow instead of a floating-card look.
- Reordered the compact controls to search input, search button, then photo
  upload.
- Moved display controls into the same fixed toolbar so sort/view/tag controls
  remain available while scrolling.
- Added an in-toolbar photo placeholder, compact image preview, and remove
  action so uploads no longer create a large preview block in the results area.
- Rendered search results in a denser two-column card grid on desktop.
- Preserved the `Not relevant` relevance-feedback action as a full-width card
  control so it remains visible in the denser grid.
- Replaced the large Streamlit match metric with a compact two-decimal match
  badge that fits inside the card header.
- Reduced recent searches to compact chips under the toolbar controls.
- Constrained the file uploader inside the toolbar and hid bulky dropzone
  instructional text to avoid label overflow.
- Added `.streamlit/config.toml` with `server.maxUploadSize = 50` to cap uploads
  at 50 MB.

Feedback/relevance controls, recipe cards, filters, and shopping-list behavior
remain unchanged by this UI pass.
