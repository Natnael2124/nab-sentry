# Real footage and Telegram: step by step

Two tests need you and a real camera:

* **Part A: record and label 1 hour of real footage** (task 5.4). This checks the motion gate, the rules and the alert latency on real footage, not just the generated scene.
* **Part B: the live Telegram test** (task 16.5). A real chat receives a snapshot and a clip, and the bot commands work.

Every command runs in **Windows PowerShell** from the project folder:

```powershell
cd C:\Users\User\dev\nab-sentry
```

Plug in the charger first; on battery everything runs 2-3 times slower.

## Files you create

| File | From | Holds | In git? |
|---|---|---|---|
| `config\cameras.yaml` | `config\cameras.example.yaml` | camera source (password!), ROI, zones, lines | **no** (git-ignored) |
| `config\rules.yaml` | `config\rules.example.yaml` | the watch rules | yes, no secrets |
| `data\real\<folder>\ground_truth.yaml` | made for you by `prepare_footage.py` | your labels | no (`data\` is ignored) |

The Telegram token and chat ID are environment variables, never stored in a file.

---

## Before you start: models up to date

The embedder changed in Step 4 (see the README, "Embedder"). Run this once after pulling the update:

```powershell
venv\Scripts\python.exe scripts\fetch_models.py
venv\Scripts\python.exe scripts\reembed.py
```

`fetch_models.py` downloads the new weights. `reembed.py` converts footage that is already stored, or says "nothing to do".

If you skip this, search and ingest stop with a message that names `reembed.py`.

## Step 0: one camera in `config\cameras.yaml`

```powershell
Copy-Item config\cameras.example.yaml config\cameras.yaml
notepad config\cameras.yaml
```

Keep one camera and delete the other entry.

**IP camera (RTSP).** Use the camera's sub-stream URL. It is often `.../stream2` or `.../Streaming/Channels/102`; see the camera's manual.

```yaml
cameras:
  - camera_id: CAM01
    label: Front gate
    source: rtsp://admin:YOURPASSWORD@192.168.1.20:554/stream2
    segment_s: 10
    encoder: auto
```

**Phone used as a webcam (e.g. Windows "Phone Link" camera, DroidCam) or the laptop webcam.** First find the device name:

```powershell
$ff = & venv\Scripts\python.exe -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
& $ff -hide_banner -list_devices true -f dshow -i dummy
```

Copy the name shown in quotes before `(video)` into the config:

```yaml
cameras:
  - camera_id: CAM01
    label: Front gate (phone)
    source: "dshow:EXACT DEVICE NAME"
    segment_s: 10
    encoder: auto
```

Mount the camera or phone so it does not move: a fixed view for the whole hour.

### Draw the ROI, zone and line

```powershell
venv\Scripts\python.exe scripts\snapshot.py CAM01
start data\live\CAM01_grid.jpg
```

The image has a grid every 0.1 of the width and height, with (0,0) at the top-left. Add to the camera in `config\cameras.yaml`:

```yaml
    roi: [0.0, 0.3, 1.0, 1.0]           # x1, y1, x2, y2: where people's BODIES move; leave out trees, sky, a road
    zones:
      yard: [[0.10, 0.55], [0.90, 0.55], [0.90, 1.00], [0.10, 1.00]]   # polygon, 3+ corners
    lines:
      gate-line: [[0.30, 0.60], [0.70, 0.60]]                          # from point -> to point
```

A person counts as **in a zone when their feet are in it**: the bottom centre of their box, not their head. A line is **crossed** when the feet go over it. `in` means crossing to the **left of the arrow** from the first point to the second; `out` means crossing to the right.

Run `snapshot.py` again: the ROI shows in yellow, zones in green and lines in magenta. Check they sit where you meant.

## Step 1: rules in `config\rules.yaml`

```powershell
Copy-Item config\rules.example.yaml config\rules.yaml
notepad config\rules.yaml
```

For the test, use **2-3 rules without a schedule**: one `enter`, one `dwell`, one `cross`. For example:

```yaml
rules:
  - id: yard-enter
    name: Person in the yard
    camera: CAM01
    zone: yard
    classes: [person]
    trigger: {type: enter}
    cooldown_s: 60

  - id: yard-linger
    name: Someone lingering in the yard
    camera: CAM01
    zone: yard
    classes: [person]
    trigger: {type: dwell, seconds: 30}
    cooldown_s: 120

  - id: gate-in
    name: Came in through the gate
    camera: CAM01
    line: gate-line
    direction: in
    classes: [person, car, motorcycle]
    trigger: {type: cross}
    cooldown_s: 30
```

Check both files without starting anything. A mistake is reported with the camera or rule and the field:

```powershell
venv\Scripts\python.exe -c "from nab_sentry.live.camera_config import load_cameras; from nab_sentry.rules.config import load_rules; c = load_cameras('config/cameras.yaml'); print(len(c), 'camera(s),', len(load_rules('config/rules.yaml', c)), 'rule(s): OK')"
```

---

## Part A: record and label one hour (task 5.4)

### A1: record

Record for **at least 60 minutes**, ideally including dusk (a lighting change) and some people or cars passing. You can walk through yourself a few times: enter the zone, stay over 30 s once, cross the line.

```powershell
venv\Scripts\python.exe scripts\record.py CAM01
```

Leave it running and stop it with **Ctrl+C** after the hour. The segments are `data\videos\CAM01_YYYYMMDDTHHMMSS.mkv`, 10 s each.

Only the recorder runs here, so the hour is plain recording with no CPU load. Note roughly when you started, e.g. 18:00.

### A2: prepare the labelling folder

```powershell
venv\Scripts\python.exe scripts\prepare_footage.py CAM01 --from 2026-10-26T18:00 --to 2026-10-26T19:00
```

Leave out `--from`/`--to` to take every CAM01 segment. This prints the folder it made, e.g. `data\real\CAM01_20261026T180002\`, containing:

* `videos\`: copies of the segments; the originals are not touched.
* `contact\sheet_01.jpg` ...: one thumbnail per second, each stamped with the **clock time**. 120 per sheet makes about 30 sheets for an hour.
* `cameras.yaml`: your camera without its password.
* `rules.yaml`: your camera's rules.
* `ground_truth.yaml`: the template you fill in.

```powershell
start data\real\CAM01_20261026T180002\contact
```

### A3: label

Flip through the sheets in the Photos app. For anything that moves, open the segment of that minute to get the exact second; the file name is its start time:

```powershell
start data\real\CAM01_20261026T180002\videos\CAM01_20261026T180410.mkv
```

Open `ground_truth.yaml` in Notepad and fill in two lists.

**`objects`**: every person or vehicle that appears, also the ones no rule is about:

* `class` is one of `person, bicycle, car, motorcycle, bus, truck`;
* `visible` is the first and last second it is on screen (inside the ROI);
* leave out cars that stay parked the whole hour.

**`expected_alerts`**: when each rule should fire:

* `enter`: the second the feet reach the zone.
* `cross`: the second the feet go over the line, in the rule's direction only.
* `dwell` (seconds: 30): entry time plus 30 s, if they stay that long.
* `count` (min: 2): the second the 2nd person is in the zone.
* **Cooldown**: a second event of the same rule within `cooldown_s` of the previous alert gets **no** alert, so don't list it.

Times can be clock times `'HH:MM:SS'`, with the quotes, or seconds from `start`. Being 1-2 s off is fine: an alert counts as matched within ±5 s.

### Example `ground_truth.yaml`

```yaml
camera: CAM01
start: '2026-10-26T18:00:02'      # filled in by prepare_footage.py - do not change
duration_s: 3598                  # filled in - do not change
objects:
  - name: me, walking in through the gate and standing in the yard
    class: person
    visible: ['18:04:10', '18:05:02']
  - name: neighbour's white car drives past on the street
    class: car
    visible: ['18:12:40', '18:12:47']
  - name: delivery man in a blue jacket, crosses the yard to the door
    class: person
    visible: ['18:31:05', '18:31:29']
  - name: cyclist on the pavement (outside the yard)
    class: bicycle
    visible: ['18:47:13', '18:47:19']
expected_alerts:
  # me: through the gate at 18:04:18, feet in the yard at 18:04:20, still there 30 s later
  - {rule: gate-in,     t: '18:04:18'}
  - {rule: yard-enter,  t: '18:04:20'}
  - {rule: yard-linger, t: '18:04:50'}
  # delivery man: in through the gate and into the yard (cooldowns have long run out)
  - {rule: gate-in,     t: '18:31:09'}
  - {rule: yard-enter,  t: '18:31:11'}
  # the car and the cyclist never enter the yard or cross the gate line: no alerts
```

A mistake in the file (wrong class, unknown rule id, end before start) is reported by name when you run A4.

### A4: run the measurements

Replace the folder name with yours.

```powershell
$f = "data\real\CAM01_20261026T180002"

# 1) motion gate: misses and idle cost for several thresholds (~10-20 min)
venv\Scripts\python.exe scripts\gate_sweep.py $f --methods diff --thresholds 0.01 0.02 0.05

# 2) full chain as fast as possible: alerts vs your labels, tracks, gate, speed, memory (~10-15 min)
venv\Scripts\python.exe scripts\acceptance.py $f batch

# 3) real-time replay of a busy stretch through the watcher and notifier: alert latency.
#    Pick a 5-10 minute window with events, in seconds from start (18:04:00 = 238 s here):
venv\Scripts\python.exe scripts\acceptance.py $f realtime --from 200 --to 700
```

Results:
* `$f\gate_sweep_<folder>.md`
* `$f\run_batch\report.yaml`
* `$f\run_realtime\report.yaml`

Each run is a fresh copy and never touches your live `data\nab_sentry.db`.

**Send me those three files**, plus `ground_truth.yaml`. I'll read them against the targets: no missed alerts, few false ones, latency under 30 s, gate idle cost near the keyframe floor (0.1). The videos and contact sheets stay on your laptop.

---

## Part B: the live Telegram test (task 16.5)

### B1: create the bot

1. In Telegram, open **@BotFather**, send `/newbot`, and pick a name and a username ending in `bot`.
2. It replies with a **token** like `7123456789:AAH...`. Keep it secret.
3. Open your new bot and press **Start**.

### B2: find your chat ID

In PowerShell; the variables last only for this window:

```powershell
$env:NAB_TELEGRAM_TOKEN = "7123456789:AAH...your token..."
$env:NAB_TELEGRAM_CHATS = "1"
venv\Scripts\python.exe scripts\notify.py
```

Send the bot any message, e.g. `hi`. The console prints:

```
... ignoring Telegram chat 123456789 (not in NAB_TELEGRAM_CHATS)
```

Press Ctrl+C. That number is your chat ID:

```powershell
$env:NAB_TELEGRAM_CHATS = "123456789"
```

For several people, separate the IDs with commas. Group chat IDs start with `-`.

### B3: start everything

Do this in the same PowerShell window, so the two variables are set:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_live.ps1
```

It starts four services: the recorder, the watcher (ingest, tracking, rules), the notifier (Telegram) and the search console at http://127.0.0.1:8765. It restarts any of them that crashes. Logs go to `data\logs\` (`run_live.log`, `recorder.*.log`, `watcher.*.log`, `notifier.*.log`).

To also use the local VLM (`/describe`, and rules with `severity: verify`), start Ollama first and add `-Vlm`:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_live.ps1 -Vlm
```

### B4: test checklist

1. **Alert:** walk into the zone.
   * Within about 30 s the chat should get a **photo** captioned `🚨 Person in the yard`, with ✅ Real / ❌ False alarm buttons.
   * A few seconds later an **8 s video clip** should arrive as a reply to the photo.
   * Press one of the buttons; the caption updates.
2. **`/status`**: shows the camera as `ok` (or `OFFLINE`), how long ago its last footage was, and how many alerts are waiting to be sent.
3. **`/rules`** and **`/alerts`**: list the rules and today's alerts.
4. **`/snapshot`**: a current frame.
5. **`/mute yard-enter 10`**, walk in again: **no** alert. Then **`/unmute yard-enter`**, wait out the 60 s cooldown, walk in: an alert again.
6. **`/today`**: today's activity log, e.g. "person in a red top in yard for 40 s". **`/ask red jacket`** finds the best match.
7. **Camera-offline alert:** unplug or cover-and-disconnect the camera. After about 30 s (3 × segment_s) you should get `⚠️ Camera offline`, and `✅ Camera back online` when it returns.
8. Stop everything with **Ctrl+C** in the run_live window.

If something does not arrive, send me:
* `data\logs\run_live.log`
* `data\logs\notifier.err.log`
* `data\logs\watcher.err.log`
* the output of `/status`

The token never appears in the logs.

### Notes

* **Network:** the notifier is the only part that goes online, and only to `api.telegram.org`. Footage leaves the laptop only as the alert photos and clips you receive.
* **Retention:** segments are deleted after 7 days and alert media after 30. The copies made by `prepare_footage.py` under `data\real\` are not.
* **Keeping the token:** the variables are gone when the window closes. To keep them for your user account:

  ```powershell
  [Environment]::SetEnvironmentVariable("NAB_TELEGRAM_TOKEN", "7123...", "User")
  [Environment]::SetEnvironmentVariable("NAB_TELEGRAM_CHATS", "123456789", "User")
  ```

  Then open a new window.
