---
version: alpha
name: Beyond Surveillance
description: A spatial operations console for inspecting video estimates and crowd-flow scenarios.
colors:
  primary: "#2456d6"
  ink: "#182944"
  background: "#f3f6fa"
  surface: "#ffffff"
  muted: "#627189"
  border: "#dce3ed"
  success: "#167552"
  warning: "#a55b10"
  danger: "#bd3449"
  accent: "#df7637"
typography:
  display:
    fontFamily: "Bahnschrift, 'Arial Narrow', sans-serif"
  sans:
    fontFamily: "'Segoe UI', Arial, sans-serif"
  mono:
    fontFamily: "Consolas, monospace"
rounded:
  DEFAULT: "10px"
  sm: "6px"
  lg: "16px"
spacing:
  page: "28px"
  gap: "18px"
components:
  button: {}
  panel: {}
  dialog: {}
---

# Beyond Surveillance design system

## Overview

The visual reference is an architectural circulation plan on a daylight operations desk. This is an English-language research product for an operator reviewing recorded crowd footage and comparing conditional gate scenarios. The map, rather than a marketing headline, anchors the page. Blue printed connections, precisely labelled gate markers, and count-scaled zone fills are the signature. Avoid security-theatre red alerts, fake live dots, ornamental gradients, and invented measurements.

Token ownership: this file generates `src/tokens.css` through `scripts/design-tokens.py`. Components consume variables; generated tokens are not edited directly. Font roles use installed system stacks to eliminate remote-font dependency and layout shift.

## Colors

White panels sit on a cool drafting-paper background. Ink carries hierarchy; blue is selection and actions. Orange highlights a changed gate, green an open gate, and red a closed gate or error. Every state also has a text label. Occupancy shading is a relative display of count/capacity assumptions, never an emergency score. Two scenario maps use the same scale.

## Typography

Bahnschrift is reserved for compact titles and prominent numbers; Segoe UI handles controls and prose. Consolas marks timestamps, zone IDs, and metric units. Tabular numerals prevent shifting values. Body line height is 1.5; short control labels remain sentence case.

## Layout

A 216px navigation rail surrounds a fluid workspace. The overview has a wide map and a 320px scenario inspector. At 1100px the rail becomes compact; below 850px content stacks and navigation becomes horizontal. Natural document scrolling owns all screens; long histories and JSON editors have local bounded overflow. Minimum controls are 36px high, with 44px touch targets on narrow screens.

## Elevation & Depth

Panels use quiet borders, shallow shadows only where useful, and 16px corners. The map has a fine dot grid and geometric building outline. Modal dialogs alone use a stronger elevation and backdrop. No ambient animations compete with data.

## Shapes

Panels use the large radius, fields/buttons the default radius, badges the small radius. SVG gate markers are circles to distinguish portals from polygonal zones. Selected zones receive a clear outline.

## Components

Shared Button, Field, Select, Dialog, Status, and Panel live in `src/ui.tsx`. Native selects intentionally retain platform-owned popups. HTML dialog owns modality, Escape, and focus isolation with explicit focus restoration. All feedback uses one status region plus persistent inline errors.

The generator maps `colors.*` to `--color-*`, `typography.*.fontFamily` to `--font-*`, `rounded.*` to `--radius-*`, and `spacing.*` to `--space-*`. Global scrollbar thumb/track/hover states derive from border/muted/background tokens. Focus uses a blue outer ring. Disabled controls remain legible and explain unavailable actions.

## Do's and Don'ts

Do distinguish video estimates, temporal forecasts, and conditional simulations. Keep unavailable data as a dash. Do not turn a checkpoint metric into a freshly measured result. Do not call a schematic map a surveyed venue. Respect reduced motion; keyboard users can select every gate and zone through parallel native controls.
