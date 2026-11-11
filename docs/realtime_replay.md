# All-fly real-time replay

Run from the repository root, using the analysis Python environment (only NumPy
and OpenCV are required):

```bash
python scripts/replay_realtime.py /path/to/recording.avi
```

The left panel shows the full camera frame, yellow tracking ellipses for every
available fly, and white reward circles during each experimental fly's training
intervals. The right panel shows the reconstructed live heatmap. The header gives
source video time and frame number. No analysis or tracking interpolation is run.
The tool reads the video's `.data` and `.trx` sidecars; both `recording.data`
and `recording.avi.data` naming conventions are supported (likewise for `.trx`).
Use trusted experiment files: these sidecars use Python pickle serialization.

Playback defaults to real time. Space pauses/resumes; `r` starts/stops MP4 recording;
`q` exits and finalizes the recording. The preview displays a red REC indicator
while recording; this indicator is excluded from the MP4. Pauses add no duplicate
frames. Repeated recording sessions create separate files, with `_002`, `_003`,
etc. appended after the first clip. Existing files are never overwritten.

Use `--output` to choose the recording path, `--record` to start recording
immediately, and `--speed` to set both preview and recorded playback speed:

```bash
# Preview and capture at twice real time
python scripts/replay_realtime.py recording.avi --speed 2 --record

# Capture at half speed without opening a preview window
python scripts/replay_realtime.py recording.avi --speed 0.5 --headless

# Preview at real time, but capture a clip that plays back at 4x
python scripts/replay_realtime.py recording.avi --record-speed 4
```

Both speed options accept positive, finite multipliers; the default is 1x.
`--record-speed` overrides the recorded speed independently of `--speed`.
Previously `--speed` affected only the preview; to retain that behavior, add
`--record-speed 1`.

Every source frame is processed and retained in the recording. MP4 frame rate is
source frame rate multiplied by recording speed, so a 60-second source interval
plays for 30 seconds at 2x or 120 seconds at 0.5x. Heatmap windows and the timestamp
header still use source video time. If rendering is slower than the requested
preview speed, preview slows down; the saved MP4 retains the requested speed.
MP4s use OpenCV's `mp4v` codec, with even image dimensions and no audio.

Additional keys:

| Key | Action |
| --- | --- |
| `m` | Switch between running and training/protocol heatmaps |
| `,` / `.` | Step backward/forward one frame and pause |
| Digits then `g` | Seek to a zero-based frame and pause |
| `HH:MM:SS` then `g` | Seek to a source video time and pause |

Stepping or seeking closes an active recording so each clip spans a continuous
source interval. Heatmap history is reconstructed at the destination; seeking
does not clear the accumulated history.

For a reproducible supplementary clip without a GUI:

```bash
python scripts/replay_realtime.py /path/to/recording.avi \
  --start 00:15:00 --end 00:16:00 \
  --headless --output exports/supplementary_replay.mp4
```

`--start` is inclusive and `--end` exclusive; either accepts an integer frame
number or `HH:MM:SS[.fraction]`. Default range: the full video, limited to the
available tracking frames. Headless mode automatically records and runs as fast
as processing allows. Exported duration equals the source interval duration
divided by the recording speed.

For large chambers, specify `--large-layout original` (244-pixel template floor)
or `--large-layout new` (295-pixel template floor). This is required because the
saved `ct='large'` metadata does not identify which separator geometry was used.
Regular and high-throughput chambers are selected from metadata automatically.
Saved template-match coordinates (`protocol.tm`) are required.

## Heatmap reconstruction

The implementation follows `Heatmap` in `tmp/rt-trx.py`: two-pixel bins initialized
to one, one count per available fly position per source frame, log transform,
independent min/max normalization in each chamber, and the JET color map. Binning
uses the tracker's Python 2 rounding convention. Missing tracking samples are
omitted. Chamber separators follow the tracker templates and saved template match.

The default running map covers the last five minutes (`--heatmap-minutes` changes
this). `--heatmap protocol` selects the training map instead. As in the tracker,
the first experimental fly's training events reset and stop the protocol map;
the completed map remains visible until the next training begins. Before the
first training it shows the empty baseline. Circle overlays follow each fly's
own saved training intervals and historical circle positions/radii, rather than
the final circle state in the metadata. Yoked chambers receive ellipses but no
extra reward circles, matching `rt-trx` annotation behavior.

This reconstructs the display from **saved video-frame tracking samples**. The
original live heatmap could receive additional camera frames before the tracker
writer dropped or repeated frames, so it cannot be guaranteed pixel-identical to
the original live display. LED-dependent circle thickness is not reconstructed;
reward circle outlines have a fixed one-pixel width. Camera controls and optional
tracker debugging overlays are not replayed.
