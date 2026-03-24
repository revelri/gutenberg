# Design System — Gutenborg

## Product Context
- **What this is:** A semantic citation retrieval tool that ingests academic corpora and returns exact quotes with page numbers — "NVivo for the LLM era"
- **Who it's for:** Humanities and social science PhD students/researchers working with dense theoretical corpora (philosophy, critical theory, sociology)
- **Space/industry:** Academic research tools (peers: NVivo, Zotero, Elicit, Semantic Scholar)
- **Project type:** Web app (chat interface + corpus dashboard + analytics)

## Aesthetic Direction
- **Direction:** Editorial/Magazine — strong typographic hierarchy, the feeling of a beautifully typeset academic text
- **Decoration level:** Intentional — thin rule lines between sections (like chapter dividers), cream paper-toned surfaces. Not flat-white sterile, not ornate.
- **Mood:** A scholarly instrument built by someone who reads. Warm, authoritative, precise. The interface should feel like opening a well-made book, not opening a chatbot. Humanities researchers should feel "this was built for me."
- **Reference sites:** Elicit (warm tones, verification badges), Connected Papers (clean minimalism), Semantic Scholar (data-forward) — Gutenborg deliberately breaks from all of these with serif typography and a monochrome palette where they use sans-serif + blue.

## Typography
- **Display/Hero:** Instrument Serif — A modern serif with warmth and character. Says "scholarship" without saying "law firm." The biggest departure from every competitor in the space, which all use sans-serif. Humanities = books = serifs.
- **Body:** Source Sans 3 — Readable, professional, has true italics for titles and foreign terms (critical for humanities writing where *Anti-Oedipus*, *deterritorialization*, and *Aufhebung* are everyday vocabulary). Excellent at small sizes.
- **UI/Labels:** Source Sans 3 (same as body, weight 600 for labels)
- **Data/Tables:** Geist Mono (tabular-nums) — Clean and precise for page numbers, chunk counts, confidence scores. Monospace ensures columnar alignment in data displays.
- **Code:** JetBrains Mono — For query prefixes (`trace:`, `cite:`), technical output, and any code-like content.
- **Loading:** Google Fonts CDN — `https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Source+Sans+3:ital,wght@0,300;0,400;0,600;0,700;1,300;1,400;1,600&family=Geist+Mono:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap`
- **Scale:**
  - Hero: 56px / 3.5rem
  - H1: 42px / 2.625rem
  - H2: 36px / 2.25rem
  - H3: 24px / 1.5rem
  - H4: 20px / 1.25rem
  - Body: 16px / 1rem
  - Small/Labels: 14px / 0.875rem
  - Caption/Metadata: 13px / 0.8125rem
  - Micro (data labels): 11px / 0.6875rem

## Color
- **Approach:** Monochrome + one accent — deliberately restrained. Ink on cream, with saddle brown as the only color. Like marginalia in sepia on an aged page. The words ARE the design.
- **Primary:** `#8B4513` (Saddle Brown) — the color of aged leather, book spines, mahogany reading rooms. Used for: accent text, links, primary buttons, active states, citation badges. Warm, authoritative, unmistakably "library."
- **Primary Light:** `#A0522D` — hover state for primary elements
- **Primary Faint:** `rgba(139, 69, 19, 0.08)` — subtle background for citation badges, code blocks
- **Neutrals (warm stone grays):**
  - Cream (background): `#FAF7F2`
  - Vellum (surface-deep): `#F5F0E8`
  - Parchment (rules/dividers): `#D6D3CD`
  - Ash (tertiary text): `#A8A29E`
  - Stone (secondary text): `#57534E`
  - Ink (primary text): `#1C1917`
  - Surface (cards): `#FFFFFF`
- **Semantic:** success `#3D7A4A`, warning `#C17D10`, error `#B93B3B`, info `#4A6FA5`
- **Dark mode strategy:** Invert surfaces (cream → near-black), warm the primary to `#C4875A`, reduce saturation 10-20% on semantics. The warm stone grays invert naturally. Keep the same monochrome discipline.

## Spacing
- **Base unit:** 8px
- **Density:** Comfortable — academics need room to read. Generous whitespace between sections.
- **Scale:** 2xs(2px) xs(4px) sm(8px) md(16px) lg(24px) xl(32px) 2xl(48px) 3xl(64px)

## Layout
- **Approach:** Hybrid — grid-disciplined for the chat/results interface, editorial touches for the corpus dashboard and landing page
- **Grid:** Single-column for chat (max 960px), 4-column for dashboard stats, 12-column base for complex layouts
- **Max content width:** 960px
- **Border radius:** Hierarchical — sm: 3px (inputs, badges), md: 6px (cards, alerts), lg: 10px (modals, panels). No fully-rounded pill shapes. Subtle, not bubbly.

## Motion
- **Approach:** Minimal-functional — only transitions that aid comprehension. No bouncing, no choreography. This is a precision instrument.
- **Easing:** enter(ease-out) exit(ease-in) move(ease-in-out)
- **Duration:** micro(50-100ms) for hover states, short(150-250ms) for reveals, medium(250-400ms) for panel transitions. No long animations.

## Component Patterns
- **Citations:** Rendered as inline badges in monospace: `[Source: title, p. 47]` with primary-faint background and primary text. Verification status appended: `✓ verified` (success color) or `⚠ approximate` (warning color).
- **Blockquotes:** Left border 2px solid primary, italic Instrument Serif, indented. Used for direct quotations from the corpus.
- **Rule lines:** 1px solid parchment (`#D6D3CD`) between major sections. Horizontal rules as chapter dividers, not just whitespace.
- **Section labels:** Source Sans 3, 11px, weight 600, uppercase, letter-spacing 0.1em, primary color. Used above section headings.
- **Data display:** Geist Mono with tabular-nums for all numeric data. Right-aligned in tables.
- **Alerts:** Left-border accent (3px) with tinted background. No icons — the color communicates severity.
- **Buttons:** Primary (brown fill, cream text), Secondary (outlined, ink text), Ghost (underlined, primary text). No gradients. No shadows.
- **Cards:** White surface, 1px parchment border, md radius. No shadows. Elevation through border only.

## Anti-patterns (never do these)
- Blue accent color (every competitor does this — we deliberately don't)
- Purple/violet gradients
- Rounded pill buttons or fully-rounded UI elements
- Sans-serif display headings (our serif display IS the brand)
- Stock photo hero sections
- Generic "Built for researchers" marketing copy
- Centered-everything layouts
- Uniform bubbly border-radius
- Drop shadows on cards (use borders instead)
- Color-circle icons in feature grids

## Decisions Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-21 | Initial design system created | Created by /design-consultation based on competitive research (Elicit, Semantic Scholar, Zotero, Connected Papers) and product positioning as a humanities-first research tool |
| 2026-03-21 | Monochrome + one accent palette | User chose bold risk: drop all secondary colors, pure black-cream-brown. Maximum typographic focus — the words are the design |
| 2026-03-21 | Instrument Serif for display | Deliberate departure from sans-serif research tool convention. Humanities researchers live in a world of serifs — this signals "built for people who read" |
| 2026-03-21 | No blue anywhere | Every competitor uses blue. Brown + cream says "library" and "archive," not "tech startup." Instant differentiation |
