# Face Lock System

A local-network face recognition door lock.
- **Backend** runs on your laptop (Python + FastAPI + FaceNet)
- **Frontend CLI** runs on the Raspberry Pi

---

## Project structure

```
face-lock/
├── backend/
│   ├── server.py          # FastAPI app
│   ├── face_engine.py     # FaceNet wrapper
│   ├── requirements.txt
│   └── embeddings/        # auto-created, stores trained faces
│
└── client/
    ├── client.py          # CLI for the Pi
    └── requirements.txt
```

---

## Backend setup (Laptop)

### 1. Python 3.10+

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> First run downloads the pretrained VGGFace2 model (~90 MB).

### 2. Find your laptop's local IP

```bash
# macOS / Linux
ifconfig | grep "inet " | grep -v 127

# Windows
ipconfig
```

Note the IP — looks like `192.168.x.x`.

### 3. Run the server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

The server is now reachable at `http://192.168.x.x:8000` on your LAN.

---

## Client setup (Raspberry Pi)

### 1. Install dependencies

```bash
cd client
pip install -r requirements.txt
```

`picamera2` is pre-installed on Raspberry Pi OS (Bullseye / Bookworm).
If missing: `sudo apt install -y python3-picamera2`

### 2. Set backend URL

Edit the top of `client.py`:

```python
BACKEND_URL = "http://192.168.x.x:8000"   # ← your laptop's IP
```

Or export as an environment variable:

```bash
export FACE_LOCK_BACKEND=http://192.168.1.42:8000
python client.py
```

### 3. Run

```bash
python client.py
```

---

## GPIO wiring (optional)

To physically trigger a solenoid lock on verified access:

| Pi pin       | Connect to        |
|--------------|-------------------|
| BCM 17 (pin 11) | Relay IN        |
| 5 V (pin 2)  | Relay VCC         |
| GND (pin 6)  | Relay GND         |

The relay's NO (normally open) terminals connect to your solenoid/lock power circuit.

The code pulses pin 17 HIGH for 3 seconds on access granted.
Change `LOCK_PIN` and the sleep duration in `trigger_lock_open()` to suit your hardware.

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/` | Health check, list trained users |
| POST | `/verify` | Verify a face photo |
| POST | `/train` | Train a new user (2–5 photos recommended) |
| GET  | `/users` | List all trained users |
| DELETE | `/users/{user_id}` | Delete a trained user |

### Example: verify via curl

```bash
curl -X POST http://192.168.1.42:8000/verify \
     -F "photo=@/path/to/face.jpg"
```

Response:
```json
{
  "status": "open",
  "user": "alice",
  "confidence": 0.9132,
  "message": "Access granted for 'alice'."
}
```

### Example: train via curl

```bash
curl -X POST http://192.168.1.42:8000/train \
     -F "user_id=alice" \
     -F "photos=@photo1.jpg" \
     -F "photos=@photo2.jpg" \
     -F "photos=@photo3.jpg"
```

---

## Tuning the threshold

In `backend/face_engine.py`:

```python
SIMILARITY_THRESHOLD = 0.75   # raise for stricter, lower for more lenient
```

| Value | Behaviour |
|-------|-----------|
| 0.80+ | Very strict — may reject the correct user in poor lighting |
| 0.75  | Recommended default |
| 0.65  | Lenient — higher false-positive risk |

---

## Tips

- **Training photos:** Use varied angles and lighting. 3–5 photos give the best accuracy.
- **Lighting:** Consistent, front-facing light improves verification accuracy.
- **Camera distance:** 40–80 cm from the camera gives the best face detection results.
- **Retrain:** Just call `/train` again with the same `user_id` to overwrite old embeddings.
