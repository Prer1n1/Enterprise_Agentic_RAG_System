"""Google Drive connector.

Deliberately NOT a parallel ingestion path. Its only job is: sync a
Drive folder into a local cache directory, then hand off to the EXISTING
ingest_directory() pipeline unchanged. That means every already-tested
piece — recursive traversal, the incremental tracker, root-scoped
delete-detection, chunking, storage — just works with zero new code
paths to trust. Same reasoning as why /documents/upload reuses
ingest_directory() instead of its own pipeline.

Google-native files (Docs/Sheets) have no raw bytes to download — they
must be EXPORTED via the Drive API to a real format (Docs -> .docx,
Sheets -> .csv, first sheet only — a known limitation, not fixed here).
Regular uploaded files (already .pdf/.docx/.csv/.html) are downloaded
directly.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
_FOLDER_MIME = "application/vnd.google-apps.folder"

# Google-native types have no raw bytes — must be exported to a real
# format. (export_mime_type, local_extension)
_EXPORT_MIME_MAP = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
    "application/vnd.google-apps.spreadsheet": ("text/csv", ".csv"),
}

# Regular (non-exported) mime types our loaders already understand.
_DIRECT_EXTENSIONS = {
    "application/pdf": ".pdf",
    "text/csv": ".csv",
    "text/html": ".html",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}

_STATE_FILENAME = ".drive_sync_state.json"


@dataclass
class DriveFile:
    file_id: str
    name: str
    mime_type: str
    modified_time: str


class GoogleDriveConnector:
    def __init__(self, credentials_path: str, folder_id: str):
        creds = service_account.Credentials.from_service_account_file(credentials_path, scopes=_SCOPES)
        self.service = build("drive", "v3", credentials=creds)
        self.folder_id = folder_id

    def list_files(self, folder_id: Optional[str] = None) -> List[DriveFile]:
        """Recursively lists files under the folder, descending into
        subfolders — same reasoning as local ingestion using rglob()
        instead of iterdir(): real content lives in subfolders."""
        folder_id = folder_id or self.folder_id
        files: List[DriveFile] = []
        page_token = None
        while True:
            response = (
                self.service.files()
                .list(
                    q=f"'{folder_id}' in parents and trashed=false",
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                    pageToken=page_token,
                )
                .execute()
            )
            for f in response.get("files", []):
                if f["mimeType"] == _FOLDER_MIME:
                    files.extend(self.list_files(folder_id=f["id"]))
                else:
                    files.append(DriveFile(f["id"], f["name"], f["mimeType"], f["modifiedTime"]))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return files

    def _local_extension(self, drive_file: DriveFile) -> str:
        if drive_file.mime_type in _EXPORT_MIME_MAP:
            return _EXPORT_MIME_MAP[drive_file.mime_type][1]
        return _DIRECT_EXTENSIONS.get(drive_file.mime_type, "")

    def _download_bytes(self, drive_file: DriveFile) -> bytes:
        if drive_file.mime_type in _EXPORT_MIME_MAP:
            export_mime, _ = _EXPORT_MIME_MAP[drive_file.mime_type]
            request = self.service.files().export_media(fileId=drive_file.file_id, mimeType=export_mime)
        else:
            request = self.service.files().get_media(fileId=drive_file.file_id)

        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buffer.getvalue()

    def _local_name(self, drive_file: DriveFile, ext: str) -> str:
        # file_id prefix keeps names unique across Drive subfolders and
        # makes the identity STABLE even if the file is renamed in Drive —
        # canonical_source() of this path becomes the citation source, so
        # it needs to survive renames, unlike a raw filename would.
        stem = drive_file.name[: -len(ext)] if drive_file.name.lower().endswith(ext) else drive_file.name
        return f"{drive_file.file_id}__{stem}{ext}"

    def sync_to_local(self, cache_dir: Path) -> None:
        """Mirrors the Drive folder into cache_dir:
        - Downloads new/changed files. A file whose Drive `modifiedTime`
          hasn't changed since the last sync is skipped without
          downloading — Drive's own metadata is a free pre-filter before
          ever touching content.
        - Removes local copies of files no longer present in Drive, so
          ingest_directory()'s own delete-detection picks up Drive-side
          deletions with no connector-specific logic needed.
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        state_path = cache_dir / _STATE_FILENAME
        state = json.loads(state_path.read_text()) if state_path.exists() else {}

        drive_files = self.list_files()

        for drive_file in drive_files:
            ext = self._local_extension(drive_file)
            if not ext:
                continue  # unsupported type (Slides, images, ...) — skip, don't crash

            local_path = cache_dir / self._local_name(drive_file, ext)
            if state.get(drive_file.file_id) == drive_file.modified_time and local_path.exists():
                continue

            content = self._download_bytes(drive_file)
            local_path.write_bytes(content)
            state[drive_file.file_id] = drive_file.modified_time

        tracked_ids = {f.file_id for f in drive_files}
        for existing in cache_dir.iterdir():
            if existing.name == _STATE_FILENAME:
                continue
            file_id = existing.name.split("__", 1)[0]
            if file_id not in tracked_ids:
                existing.unlink()
                state.pop(file_id, None)

        state_path.write_text(json.dumps(state))
