# Product workflow contract

## Business-context sources

`IMPLEMENTATION_PLAN.md` and the user's current instruction define the video-to-map and gate-scenario workflow. `backend/schemas.py` and `backend/simulator.py` enforce state/units/topology/conservation. The user supplied `best_balanced_ddpfnet.pth.zip` as the authoritative DDPF weights; incompatible older features cannot silently replace that model. Local-only storage is retained until explicitly deleted by the operator. No external messaging, billing, gate actuation, or emergency advice exists in this product.

## Canonical UI Map

| Capability     | Canonical owner                        | Source of truth             | Allowed variants               | Verification            |
| -------------- | -------------------------------------- | --------------------------- | ------------------------------ | ----------------------- |
| Select/Listbox | `src/ui.tsx` Select                    | This contract               | native, OS-owned popup         | browser keyboard/select |
| Form           | `src/ui.tsx` Field and backend schemas | API                         | edit, upload                   | validation tests        |
| Scrollbar      | `src/style.css` global rules           | DESIGN.md                   | bounded editor/history         | narrow viewport         |
| Toast          | `src/ui.tsx` Status                    | This contract               | info, success, error           | live region             |
| CRUD           | `src/api.ts`, App state                | API immutable run snapshots | create, versioned edit, reopen | browser/API             |
| Dialog         | `src/ui.tsx` Dialog                    | This contract               | upload, confirm                | keyboard/focus          |

## Navigation and flow ledger

Overview, Scenario lab, Venue setup, Run library, and Model evidence share a single shell. The active view is stored in the URL. Selected run is persisted locally for reopening; no video bytes or model weights go into browser storage. Upload → validate → select venue → create persistent job → overview. Gate changes → explicit run comparison → scenario lab. Save venue → stay on setup and announce version. Export → download immutable provenance/results.

## Async and resilience

Prevent duplicate submits; retain form values after failures. Upload has byte progress and abort. Worker jobs have persisted progress, cancel, retry, and recovery after restart. Polling ignores stale responses through effect cleanup. Snapshot selection cancels superseded reads. Scenario edits mark existing comparisons stale; old results retain their input labels. Offline failures show a persistent retry affordance; prior successful data remains readable.

## Validation and accessibility

Forms use noValidate, explicit labels and inline errors. Native selects are deliberate. Native dialog is opened with showModal, closes on Escape, and restores focus. The map's pointer actions have keyboard-selectable zone/gate lists. Focus rings, reduced motion, text state labels and responsive stacking apply on every screen. Number formatting uses en-IN and video times are elapsed times, not wall-clock dates.

## Verification

Run `npm run typecheck`, `npm run build`, Python API/simulator tests, browser workflow checks, design token generation/drift check, and the premium strict audit. Compare the overview and scenario lab for shared map scales and controls. Full application training/causal validation remains distinct from software verification.
