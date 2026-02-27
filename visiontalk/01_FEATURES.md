# VisionTalk — Feature Design
## The 3 Unique, Real, Demo-able Winning Features

---

## Why Our Features Win

Most hackathon assistive tech projects do one of these generic things:
- Object detection → "Chair detected" (not useful)
- OCR → reads everything (noisy, confusing)
- Cloud AI Q&A → needs internet

**VisionTalk is different because:**
1. 3D spatial language — not "chair detected" but "Chair 1 meter to your left"
2. Local LLM reasoning — answers questions WITHOUT cloud (Ollama + Llama-3)
3. The Color Sense sub-feature — No existing assistive app does this simply
4. Smart narration — dedup, severity ranking, path-clear announcements
5. Everything runs offline on a $300 laptop + a smartphone

---

## Feature 1: 🧭 NAVIGATE — "3D Spatial Narrator"

### What it does
Always-on companion that describes nearby objects with:
- **Direction** — "left" / "ahead" / "right" based on camera thirds
- **Distance** — "very close" / "nearby" / "ahead" — from YOLO bbox size + MiDaS depth
- **Zone** — "ground level" / "overhead" — important for different hazards
- Stair/drop detection via MiDaS depth gradient analysis

### What it says (examples)
- "Chair nearby, to your left"
- "Person approaching from ahead"
- "Warning — possible step down ahead"
- "Path clear ahead"

### Why competitors can't match it
- Most use YOLO only (no depth) → just says "chair" not where/how far
- MiDaS gives monocular depth from a single phone camera — no special hardware
- Smart scene memory prevents it from saying "chair" every 100ms (only on changes)

### Tech
- YOLOv8n ONNX — 80 COCO classes, runs at 10fps on CPU
- MiDaS v2.1-small — monocular depth, every 3rd frame
- SpatialResult dataclass — encodes direction, distance, zone
- SceneMemory — TTL dedup, re-announces only when distance level changes

### Demo moment (30 seconds)
Walk toward a chair. User hears: "Chair ahead, a few steps" → closer: "Chair nearby, directly ahead" → very close: "Stop. Chair directly ahead."

---

## Feature 2: 🧠 ASK — "Ask Your Environment"

### What it does
User presses mic, asks a question. VisionTalk:
1. Snaps the current frame
2. Runs YOLO + OCR in one shot to build scene context
3. Sends context + question to **local Llama-3** via Ollama
4. Speaks the answer back via TTS

### Questions it can answer
| Question | How it answers |
|---|---|
| "What is in front of me?" | YOLO object list → LLM narrates |
| "What color is the shirt?" | Color sub-feature → HSV histogram → "Dark blue" |
| "Is the door open?" | YOLO + spatial → "Door on your left, no blocking detected" |
| "What does this label say?" | OCR text → passed to LLM → quoted back |
| "Is this medicine safe?" | OCR reads label → LLM repeats name + dosage |
| "Is there anyone in the room?" | Person detection → "Yes, 1 person to your right" |

### COLOR SENSE sub-feature (built into ASK)
When question contains "color" or "colour":
- Detect the center object's bounding box
- Extract that region from frame
- Compute HSV histogram of the region
- Map dominant hue range → color name
- e.g., "That object is dark blue"

No extra model needed. Pure OpenCV math. Works instantly.

### Why this is the WOW moment for judges
- NO team uses a local offline LLM with scene context at a hackathon
- Works without internet — demo in a basement, hotel, anywhere
- The contextual reasoning is genuinely impressive live

### Tech
- Ollama CLI (`ollama run llama3:8b`) — subprocess call
- Fallback: rule-based description if Ollama not installed
- OCR (PaddleOCR) provides text in context
- YOLO provides object list
- Scene context formatted as structured prompt

### Demo moment (40 seconds)
Point at a table with a medicine bottle. Ask: "What is this and is it safe to take?" → "I can see a bottle labeled Paracetamol 500mg on the table in front of you."

---

## Feature 3: 📖 READ — "Smart Text Reading"

### What it does
Focused text reading session:
- Continuously runs PaddleOCR on every 5th frame
- Prioritizes center-of-frame text (most important to user)
- Reads in natural sentence order (top to bottom)
- Speaks confidently — only text with >60% confidence
- Stops when you tap Navigate

### What makes it better than generic OCR
- Center-priority: reads what you're pointing at, not random background text
- Natural pacing: groups related text, speaks as sentences not word-by-word
- Smart dedup: doesn't re-read the same text on next frame

### Use cases that win judge hearts
- Reading a medicine label and dosage
- Reading a restaurant menu they're handed
- Reading a bus number or sign in a new place
- Reading a form at a government office

### Demo moment (20 seconds)
Hold up a printed page. Tap READ. Hear the relevant center text read clearly.

---

## Hackathon Narrative (say this to judges)

> "Today, 360 million people with visual impairment rely on other humans to describe their environment. VisionTalk is a local AI companion that fits in your pocket. It describes your world in 3D space, answers your questions about what it sees, and reads the text around you — all completely offline.
> 
> No internet. No cloud. No subscription. Just a smartphone camera and a laptop."

---

## Why We Will Win

| Judging Criteria | Our Advantage |
|---|---|
| Innovation | Local LLM (Ollama) — almost no hackathon team does this |
| Technical depth | YOLOv8 + MiDaS + PaddleOCR + Llama-3 local pipeline |
| Real problem | Disability access — universal empathy from judges |
| Demo quality | 3 distinct wow moments in 3 minutes |
| Accessibility | Works on any phone + any laptop. No extra hardware |
| Offline first | "Works in a hospital ward with no WiFi" |
