import { useRef, useState } from "react";
import { api, idempotency, post, upload, number } from "./api";
import type { Job, Venue } from "./types";
import { Button, Dialog, Field, Icon, Select, Status } from "./ui";
export function UploadDialog({
  open,
  onClose,
  venues,
  onRun,
}: {
  open: boolean;
  onClose: () => void;
  venues: Venue[];
  onRun: (run: Job) => void;
}) {
  const [file, setFile] = useState<File | null>(null),
    [venue, setVenue] = useState(""),
    [fps, setFps] = useState("1"),
    [camera, setCamera] = useState("Camera 01"),
    [progress, setProgress] = useState(0),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const abort = useRef<AbortController | null>(null),
    picker = useRef<HTMLInputElement>(null);
  function choose(f?: File) {
    if (!f) return;
    setError("");
    if (!/\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(f.name)) {
      setError("Choose an MP4, MOV, AVI, MKV, M4V or WebM video.");
      return;
    }
    if (f.size > 1024 ** 3) {
      setError("Video must be under 1 GB.");
      return;
    }
    setFile(f);
  }
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) {
      setError("Choose a video before starting analysis.");
      picker.current?.focus();
      return;
    }
    if (!camera.trim()) {
      setError("Enter a camera name.");
      return;
    }
    setBusy(true);
    setError("");
    setProgress(0);
    const controller = new AbortController();
    abort.current = controller;
    try {
      const video = await upload(file, setProgress, controller.signal);
      const run = await post<Job>("/runs", {
        video_id: video.id,
        venue_id: venue || venues[0]?.id,
        camera_name: camera.trim(),
        sample_fps: Number(fps),
        idempotency_key: idempotency(),
      });
      onRun({ ...run, video });
      onClose();
      setFile(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      abort.current = null;
    }
  }
  return (
    <Dialog
      open={open}
      title="Analyze a video"
      onClose={() => {
        abort.current?.abort();
        onClose();
      }}
    >
      <form noValidate onSubmit={submit} className="upload-form">
        <p className="muted">
          Upload recorded footage and map its crowd estimates to a venue.
          Processing runs locally and continues if you close this dialog.
        </p>
        <div
          className={`dropzone ${file ? "has-file" : ""}`}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            if (!busy) choose(e.dataTransfer.files[0]);
          }}
        >
          <span className="upload-symbol">
            <Icon name={file ? "file" : "upload"} size={30} />
          </span>
          <strong>{file ? file.name : "Drop your video here"}</strong>
          <p>
            {file
              ? `${number(file.size / 1024 / 1024, 1)} MB · ready to upload`
              : "MP4, MOV, AVI, MKV or WebM · up to 1 GB / 60 min"}
          </p>
          <Button
            type="button"
            onClick={() => picker.current?.click()}
            disabled={busy}
          >
            {file ? "Choose another video" : "Browse files"}
          </Button>
          <input
            ref={picker}
            className="sr-only"
            type="file"
            aria-label="Video file"
            accept=".mp4,.mov,.avi,.mkv,.webm,.m4v"
            onChange={(e) => choose(e.target.files?.[0])}
          />
        </div>
        <Field label="Venue mapping" id="upload-venue">
          <Select
            id="upload-venue"
            value={venue || venues[0]?.id || ""}
            disabled={busy}
            onChange={(e) => setVenue(e.target.value)}
          >
            {venues.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name}
              </option>
            ))}
          </Select>
        </Field>
        <div className="form-grid">
          <Field label="Camera name" id="camera">
            <input
              id="camera"
              value={camera}
              maxLength={80}
              disabled={busy}
              onChange={(e) => setCamera(e.target.value)}
            />
          </Field>
          <Field label="Analysis sampling" id="sample">
            <Select
              id="sample"
              value={fps}
              disabled={busy}
              onChange={(e) => setFps(e.target.value)}
            >
              <option value="0.5">1 frame / 2 seconds</option>
              <option value="1">1 frame / second</option>
              <option value="2">2 frames / second</option>
            </Select>
          </Field>
        </div>
        {busy && (
          <div className="upload-progress">
            <div className="progress-track">
              <i style={{ width: `${progress * 100}%` }} />
            </div>
            <span>
              {progress < 1
                ? `Uploading · ${Math.round(progress * 100)}%`
                : "Validating video and starting analysis…"}
            </span>
          </div>
        )}
        {error && <Status message={error} tone="error" />}
        <div className="dialog-actions">
          <Button
            type="button"
            onClick={() => {
              abort.current?.abort();
              if (!busy) onClose();
            }}
          >
            {busy ? "Cancel upload" : "Cancel"}
          </Button>
          <Button
            variant="primary"
            type="submit"
            busy={busy}
            disabled={!venues.length}
          >
            <Icon name="pulse" size={17} />
            Start analysis
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
