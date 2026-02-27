# VisionTalk — Demo Design
## How the Real Demo Works

---

## THE REAL BLIND DEMO (Not a UI overlay — a real human experience)

### Setup for judges

```
Laptop screen (projector-facing):
┌─────────────────────────────────────────────────────┐
│  Live camera feed with bounding boxes overlaid      │
│  Real-time labels: "Chair [nearby, left] 89%"       │
│  Direction zone lines drawn on screen               │
│  Guidance banner: "Chair nearby, to your left"      │
│  FPS counter proving it's live                      │
└─────────────────────────────────────────────────────┘

User (the demo actor, "the blind user"):
  - Physically closes eyes
  - Wears earphones (hears laptop TTS)
  - Only interacts by voice or phone
  - Has NO idea what's on screen
```

### The Demo Flow (3 minutes, scripted)

**Minute 1 — NAVIGATE mode (judges see bounding boxes LIVE):**
```
1. Demo actor closes eyes, says nothing
2. Walks toward a chair
3. Judges on screen see: YOLOv8 bounding box tracking chair
4. System speaks aloud: "Chair nearby, to your left"
5. Actor adjusts → system updates: "Chair very close, directly ahead"
6. Actor stops
7. System says: "Path clear ahead"
   → Proves: system IS seeing real-time, not hardcoded
```

**Minute 2 — ASK mode (voice conversation, judges see context):**
```
1. Actor says: "Guide me to the door"
2. System (if door detected): "Door on your right, about 3 steps"
3. Actor says: "What color is the chair?"
4. System: "The chair appears to be dark brown" (instant — no LLM)
5. Actor says: "What is in front of me?"
6. System (Llama-3): "I can see a table directly ahead with a bottle on it"
   → Chat panel shows the conversation history
   → Detective panel shows live bounding boxes + coords
```

**Minute 3 — READ mode + Snapshot:**
```
1. Actor holds up printed paper
2. System: "Reading: Paracetamol 500mg. Take 2 tablets daily."
3. Actor says: "Remember this" → snapshot taken
4. Actor moves to different position
5. Actor says: "What changed?" 
6. System: "New objects: person. Gone: bottle from table."
   → Proves memory is real, not fake
```

---

## WHY THIS DEMO WINS

| Judge concern | How we address it |
|---|---|
| "Is the AI hardcoded?" | Bounding boxes visible live with changing coords |
| "Does it really work?" | Actor closes eyes, makes mistakes, system corrects them |
| "Is it useful?" | Real problem — actor simulates daily visually impaired life |
| "Is it offline?" | Pull WiFi cable mid-demo — system keeps working |
| "What's unique?" | Local Llama-3, approaching alerts, color sense — live |

---

## SCREEN LAYOUT DURING DEMO (Full Screen Presentation Mode)

When demo starts, press "🎬 Demo Mode" button → UI changes to:

```
┌──────────────────────────────────────────────────────────────────┐
│  VisionTalk                              🟢 LIVE  FPS: 9.8      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌────────────────────────────────────┐                         │
│   │  📷 LIVE CAMERA + BOUNDING BOXES  │  🔍 WHAT AI SEES:       │
│   │                                   │                         │
│   │  [Chair] ─── 89% ─── nearby,left  │  › chair (left, 89%)   │
│   │  ████████                         │  › bottle (ahead, 72%)  │
│   │                                   │  › table (ahead, 91%)  │
│   │                                   │                         │
│   │  [Table] ─── 91% ─── ahead       │  BBox: [142,241]→[318,479]│
│   │  ████████████████                 │                         │
│   └────────────────────────────────────┘  Mode: NAVIGATE        │
│                                                                  │
│  📢 SPEAKING NOW:                                                │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  "Chair nearby, to your left"                              │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  💬 CONVERSATION:  You: "What is in front?"                     │
│                    🧠 "Table directly ahead with a bottle"      │
└──────────────────────────────────────────────────────────────────┘
```

---

## VOICE COMMANDS THE DEMO ACTOR USES

| Actor says | System does |
|---|---|
| "Guide me to the door" | Detects door, speaks direction |
| "What is in front of me?" | Llama-3 answers from scene |
| "What color is this?" | Instant HSV color detection |
| "Remember this" | Snapshot saved |
| "What changed?" | Scene diff spoken |
| "Navigate" | Switches to NAVIGATE mode |
| "Read" | Switches to READ mode |
| (holds up paper) | OCR reads it aloud |

All voice commands go through `voice.js` → `SpeechRecognition` → WebSocket → backend.

---

## THE "IT'S NOT HARDCODED" PROOF

Judges always wonder: "Is this fake?" Here's how we prove it's real:

1. **Bounding boxes update with every frame** — if it were hardcoded, boxes wouldn't track moving objects
2. **Coordinates change in Detective Panel** — live pixel values like [142,241]→[318,479] changing every 100ms
3. **FPS counter** — shows 8-10 FPS of real inference
4. **Ask a question mid-demo that wasn't scripted** — Llama-3 answers it
5. **Pull WiFi cable** — system keeps running (local only)
6. **Walk in a random pattern** — system tracks correctly

---

## DEMO PREPARATION CHECKLIST

```
Night before:
[ ] ollama pull llama3:8b       (downloads ~5GB)
[ ] python scripts/warmup.py   (pre-loads all models)
[ ] Test all 8 voice commands

Day of demo:
[ ] Phone with IP Webcam app running on same WiFi
[ ] Phone IP entered in Settings panel
[ ] Camera feed showing in browser
[ ] All bounding boxes visible
[ ] TTS voice audible on speaker
[ ] Overlay permanently ON for demo

Demo hardware layout:
  Laptop → projector (judges see screen with bounding boxes)
  Phone camera → strapped to demo actor's chest or held in hand
  Speaker → faces audience
  Earphone → demo actor (optional, if on stage)
```
