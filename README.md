# Attendify — Smart Attendance System

Face recognition (DeepFace / Facenet512) + a simulated Wi-Fi/MAC-address
check, backed by SQLite. Full flow: register students, mark attendance
via webcam, review/export logs from a dashboard.

## How it works

1. **Registration** — capture a student's face via the browser webcam,
   extract a 512-d Facenet embedding, and store it (plus name, roll
   number, and a device MAC address) in SQLite.
2. **Marking attendance** — capture a frame, compute its embedding, and
   compare (cosine similarity) against every registered student. The
   closest match above a threshold (0.68) is a candidate.
3. **Wi-Fi/MAC cross-check** — `wifi_simulator.py` mocks scanning the
   classroom access point for connected devices. If the matched
   student's MAC shows up, they're marked **PRESENT**; if their face
   matched but their device didn't, they're **FLAGGED** (possible proxy
   attendance — someone else's photo, or a classmate scanning on their
   behalf). Swap `simulate_connected_macs()` for a real router/AP query
   to make this live.
4. **Dashboard** — filter attendance by student/date/status, export to
   CSV.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**. First run will download the Facenet512
weights (~95 MB) automatically — needs internet access once.

Allow camera access in your browser when prompted (register/mark pages).

## Project structure

```
app.py                 Flask routes
models.py               SQLAlchemy models (Student, AttendanceLog)
face_utils.py           DeepFace embedding extraction + matching
wifi_simulator.py       Simulated Wi-Fi/MAC presence check
templates/              Jinja2 pages (Bootstrap 5)
static/                 CSS + webcam.js
data/                   SQLite DB + stored student photos (created at runtime)
```

## Notes / things to swap for a real deployment

- **Wi-Fi verification** is currently simulated (`wifi_simulator.py`).
  Replace it with a real query against your access point (SNMP, the
  router's admin API, or `arp-scan`) to check which MACs are actually
  connected.
- **Match threshold** (`face_utils.MATCH_THRESHOLD = 0.68`) — tune this
  against your own test photos; stricter lighting/camera setups can
  usually go higher (fewer false positives), noisier ones may need it
  a bit lower.
- **Secret key / debug mode** in `app.py` should be changed before any
  real deployment (`app.run(debug=True)` → `debug=False`, and set a real
  `SECRET_KEY`).
- Currently single-process/single-camera; for a classroom with many
  simultaneous check-ins, you'd want a queue or multiple capture
  stations feeding the same `/mark` endpoint.
