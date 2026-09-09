export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status}). Please retry.`;
    try {
      const data = await response.json();
      message =
        typeof data.detail === "string"
          ? data.detail
          : Array.isArray(data.detail)
            ? data.detail
                .map(
                  (x: { msg: string; loc: string[] }) =>
                    `${x.loc.slice(1).join(".")}: ${x.msg}`,
                )
                .join("; ")
            : message;
    } catch {
      /* preserve HTTP status */
    }
    throw new Error(message);
  }
  return response.json();
}
export const post = <T>(path: string, value?: unknown) =>
  api<T>(path, {
    method: "POST",
    body: value === undefined ? undefined : JSON.stringify(value),
  });
export const idempotency = () => crypto.randomUUID();
export const number = (n: number | null | undefined, digits = 0) =>
  n == null
    ? "—"
    : new Intl.NumberFormat("en-IN", { maximumFractionDigits: digits }).format(
        n,
      );
export const clock = (seconds: number) =>
  `${Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0")}:${Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0")}`;
export function upload(
  file: File,
  progress: (value: number) => void,
  signal: AbortSignal,
): Promise<import("./types").Video> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/videos");
    const abort = () => xhr.abort();
    signal.addEventListener("abort", abort, { once: true });
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) progress(e.loaded / e.total);
    };
    xhr.onload = () => {
      signal.removeEventListener("abort", abort);
      try {
        const data = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) resolve(data);
        else reject(new Error(data.detail || "Video upload failed"));
      } catch {
        reject(new Error("Invalid server response. Please retry."));
      }
    };
    xhr.onerror = () =>
      reject(
        new Error(
          "Connection lost. Your file is still selected; retry the upload.",
        ),
      );
    xhr.onabort = () => reject(new Error("Upload cancelled"));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}
