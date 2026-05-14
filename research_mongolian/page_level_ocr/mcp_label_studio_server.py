#!/usr/bin/env python3
"""MCP and CLI helpers for the page-level OCR Label Studio workflow."""

from __future__ import annotations

import argparse
import csv
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP
from PIL import Image


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv_labelstudio"
LABEL_STUDIO = VENV / "bin" / "label-studio"
PYTHON = VENV / "bin" / "python"
DATA_DIR = ROOT / "label_studio_data"
MEDIA_DIR = ROOT / "label_studio_media"
VAL_PNG_DIR = MEDIA_DIR / "val_png"
IMPORT_DIR = ROOT / "label_studio_import"
EXPORT_DIR = ROOT / "label_studio_export"
ENV_PATH = ROOT / "label_studio_local.env"
MANIFEST = ROOT / "page_split_manifest.csv"
BOOTSTRAP = ROOT / "page_level_annotations.bootstrap.json"
PROJECT_JSON = ROOT / "page_level_annotations.json"
VALIDATE_SCRIPT = ROOT / "validate_annotations.py"
PROJECT_TITLE = "Traditional Mongolian Page-Level OCR - val"
DEFAULT_USERNAME = "ocr-labeler@example.local"
DEFAULT_PORT = 8080

ORIENTATIONS = ["correct", "rotated_90_ccw", "rotated_90_cw", "rotated_180", "ambiguous"]
DEGRADATION_TAGS = [
    "severe_fade",
    "bleed_through",
    "red_seal",
    "fold",
    "stain",
    "broken_spine",
    "dense_background",
    "marginalia",
    "overlap",
    "other",
]


mcp = FastMCP("label_studio_mcp")


LABEL_CONFIG = """
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="TextColumn" background="#1f77b4"/>
    <Label value="Ignore" background="#d62728"/>
  </RectangleLabels>
  <Header value="Reading order"/>
  <TextArea name="reading_order" toName="image" perRegion="true" editable="true" rows="1"/>
  <Header value="Orientation"/>
  <Choices name="orientation" toName="image" perRegion="true" choice="single">
    <Choice value="correct"/>
    <Choice value="rotated_90_ccw"/>
    <Choice value="rotated_90_cw"/>
    <Choice value="rotated_180"/>
    <Choice value="ambiguous"/>
  </Choices>
  <Header value="Degradation tags"/>
  <Choices name="degradation_tags" toName="image" perRegion="true" choice="multiple">
    <Choice value="severe_fade"/>
    <Choice value="bleed_through"/>
    <Choice value="red_seal"/>
    <Choice value="fold"/>
    <Choice value="stain"/>
    <Choice value="broken_spine"/>
    <Choice value="dense_background"/>
    <Choice value="marginalia"/>
    <Choice value="overlap"/>
    <Choice value="other"/>
  </Choices>
</View>
""".strip()


def ensure_dirs() -> None:
    for path in (DATA_DIR, MEDIA_DIR, VAL_PNG_DIR, IMPORT_DIR, EXPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def read_env() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    values: dict[str, str] = {}
    with ENV_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value
    return values


def write_env(values: dict[str, str]) -> None:
    lines = [f"{key}={values[key]}\n" for key in sorted(values)]
    ENV_PATH.write_text("".join(lines), encoding="utf-8")
    ENV_PATH.chmod(0o600)


def ensure_env(port: int | None = None) -> dict[str, str]:
    ensure_dirs()
    values = read_env()
    values.setdefault("LABEL_STUDIO_USERNAME", DEFAULT_USERNAME)
    values.setdefault("LABEL_STUDIO_PASSWORD", secrets.token_urlsafe(18))
    values.setdefault("LABEL_STUDIO_USER_TOKEN", secrets.token_hex(24))
    values.setdefault("SECRET_KEY", secrets.token_urlsafe(48))
    if port is not None:
        values["LABEL_STUDIO_PORT"] = str(port)
    else:
        values.setdefault("LABEL_STUDIO_PORT", str(DEFAULT_PORT))
    values["LABEL_STUDIO_DATA_DIR"] = str(DATA_DIR)
    values["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    values["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(MEDIA_DIR)
    values["LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN"] = "true"
    write_env(values)
    return values


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def choose_port(preferred: int = DEFAULT_PORT) -> int:
    if is_port_open(preferred):
        return 8081
    return preferred


def service_url(port: int | None = None) -> str:
    env = read_env()
    if port is None:
        port = int(env.get("LABEL_STUDIO_PORT", DEFAULT_PORT))
    return f"http://localhost:{port}"


def api_headers() -> dict[str, str]:
    token = ensure_env().get("LABEL_STUDIO_USER_TOKEN", "")
    return {"Authorization": f"Token {token}", "Content-Type": "application/json"}


def wait_for_server(port: int, timeout: int = 90) -> bool:
    base = service_url(port)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(f"{base}/api/projects", headers=api_headers(), timeout=5)
            if response.status_code in {200, 401, 403}:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def start_label_studio_process(port: int | None = None, open_browser: bool = False) -> dict[str, Any]:
    selected_port = port or DEFAULT_PORT
    if is_port_open(selected_port):
        ensure_env(selected_port)
        try:
            response = requests.get(f"{service_url(selected_port)}/api/projects", headers=api_headers(), timeout=5)
            if response.status_code == 200:
                env_values = read_env()
                return {
                    "status": "already_running",
                    "url": service_url(selected_port),
                    "port": selected_port,
                    "username": env_values.get("LABEL_STUDIO_USERNAME", DEFAULT_USERNAME),
                    "env_file": str(ENV_PATH),
                }
        except requests.RequestException:
            pass
        fallback_port = 8081
        if fallback_port != selected_port and not is_port_open(fallback_port):
            selected_port = fallback_port
        else:
            raise RuntimeError(f"Port {selected_port} is occupied and port {fallback_port} is not available as a fallback.")
    env_values = ensure_env(selected_port)
    env = os.environ.copy()
    env.update(env_values)
    env["LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK"] = "true"
    env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(MEDIA_DIR)
    env["LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN"] = "true"

    command = [
        str(LABEL_STUDIO),
        "start",
        "--no-browser",
        "--data-dir",
        str(DATA_DIR),
        "--host",
        service_url(selected_port),
        "--port",
        str(selected_port),
        "--enable-legacy-api-token",
    ]
    log_path = DATA_DIR / "label_studio.log"
    log_file = log_path.open("ab")
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )
    ready = wait_for_server(selected_port)
    if open_browser:
        webbrowser.open(service_url(selected_port))
    return {
        "status": "started" if ready else "starting",
        "url": service_url(selected_port),
        "port": selected_port,
        "pid": process.pid,
        "username": env_values["LABEL_STUDIO_USERNAME"],
        "env_file": str(ENV_PATH),
        "log_file": str(log_path),
    }


def read_manifest(split: str = "val") -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row["split"] == split]


def read_bootstrap() -> dict[str, dict[str, Any]]:
    with BOOTSTRAP.open(encoding="utf-8") as f:
        payload = json.load(f)
    return {page["page_id"]: page for page in payload.get("pages", [])}


def convert_tif_to_png(source: Path, dest: Path) -> tuple[int, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.load()
        if image.mode not in {"RGB", "RGBA", "L"}:
            image = image.convert("RGB")
        image.save(dest)
        return image.size


def bbox_to_percent(bbox: list[float], width: int, height: int) -> dict[str, float]:
    x_min, y_min, x_max, y_max = [float(v) for v in bbox]
    return {
        "x": x_min / width * 100.0,
        "y": y_min / height * 100.0,
        "width": (x_max - x_min) / width * 100.0,
        "height": (y_max - y_min) / height * 100.0,
        "rotation": 0,
        "rectanglelabels": ["Ignore" if False else "TextColumn"],
    }


def make_prediction_result(column: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    region_id = column.get("column_id") or secrets.token_hex(8)
    result = [
        {
            "id": region_id,
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": bbox_to_percent(column["bbox"], width, height),
            "origin": "prediction",
        }
    ]
    result.append(
        {
            "id": region_id,
            "from_name": "reading_order",
            "to_name": "image",
            "type": "textarea",
            "value": {"text": [str(column.get("reading_order", ""))]},
            "origin": "prediction",
        }
    )
    result.append(
        {
            "id": region_id,
            "from_name": "orientation",
            "to_name": "image",
            "type": "choices",
            "value": {"choices": [column.get("orientation", "correct")]},
            "origin": "prediction",
        }
    )
    tags = column.get("degradation_tags", [])
    if tags:
        result.append(
            {
                "id": region_id,
                "from_name": "degradation_tags",
                "to_name": "image",
                "type": "choices",
                "value": {"choices": tags},
                "origin": "prediction",
            }
        )
    return result


def prepare_tasks(split: str = "val") -> dict[str, Any]:
    ensure_dirs()
    rows = read_manifest(split)
    bootstrap = read_bootstrap()
    tasks: list[dict[str, Any]] = []
    for row in rows:
        page_id = row["page_id"]
        source = Path(row["image_path"])
        png_path = VAL_PNG_DIR / f"{page_id}.png"
        width, height = convert_tif_to_png(source, png_path)
        page = bootstrap.get(page_id, {"columns": []})
        predictions: list[dict[str, Any]] = []
        pred_result: list[dict[str, Any]] = []
        for column in page.get("columns", []):
            pred_result.extend(make_prediction_result(column, width, height))
        if pred_result:
            predictions.append({"model_version": "bootstrap-proposed", "score": 0.5, "result": pred_result})
        tasks.append(
            {
                "data": {
                    "image": f"/data/local-files/?d=val_png/{png_path.name}",
                    "page_id": page_id,
                    "split": row["split"],
                    "original_image_path": row["image_path"],
                    "png_path": str(png_path),
                    "width": width,
                    "height": height,
                },
                "predictions": predictions,
            }
        )
    output = IMPORT_DIR / f"{split}_tasks.json"
    output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "tasks": len(tasks),
        "png_dir": str(VAL_PNG_DIR),
        "task_file": str(output),
        "split": split,
    }


def get_projects() -> list[dict[str, Any]]:
    response = requests.get(f"{service_url()}/api/projects", headers=api_headers(), timeout=20)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    if isinstance(payload, list):
        return payload
    return []


def find_project(title: str = PROJECT_TITLE) -> dict[str, Any] | None:
    for project in get_projects():
        if project.get("title") == title:
            return project
    return None


def create_or_open_project_impl(title: str = PROJECT_TITLE) -> dict[str, Any]:
    existing = find_project(title)
    if existing:
        project_id = existing["id"]
        requests.patch(
            f"{service_url()}/api/projects/{project_id}",
            headers=api_headers(),
            data=json.dumps({"label_config": LABEL_CONFIG}),
            timeout=20,
        ).raise_for_status()
        return {"status": "existing", "id": project_id, "title": title, "url": f"{service_url()}/projects/{project_id}"}
    response = requests.post(
        f"{service_url()}/api/projects",
        headers=api_headers(),
        data=json.dumps({"title": title, "label_config": LABEL_CONFIG}),
        timeout=20,
    )
    response.raise_for_status()
    project = response.json()
    return {"status": "created", "id": project["id"], "title": title, "url": f"{service_url()}/projects/{project['id']}"}


def import_tasks_impl(project_id: int | None = None, split: str = "val") -> dict[str, Any]:
    if project_id is None:
        project = create_or_open_project_impl()
        project_id = int(project["id"])
    task_file = IMPORT_DIR / f"{split}_tasks.json"
    if not task_file.exists():
        prepare_tasks(split)
    tasks = json.loads(task_file.read_text(encoding="utf-8"))
    response = requests.post(
        f"{service_url()}/api/projects/{project_id}/import",
        headers=api_headers(),
        data=json.dumps(tasks),
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return {"project_id": project_id, "task_count": len(tasks), "response": payload}


def open_project_impl(project_id: int | None = None) -> dict[str, Any]:
    if project_id is None:
        project = find_project(PROJECT_TITLE) or create_or_open_project_impl()
        project_id = int(project["id"])
    url = f"{service_url()}/projects/{project_id}/data"
    webbrowser.open(url)
    return {"opened": url, "project_id": project_id}


def export_annotations_impl(project_id: int | None = None) -> dict[str, Any]:
    if project_id is None:
        project = find_project(PROJECT_TITLE)
        if not project:
            raise RuntimeError(f"Project not found: {PROJECT_TITLE}")
        project_id = int(project["id"])
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    export_path = EXPORT_DIR / f"project_{project_id}_export.json"
    response = requests.get(
        f"{service_url()}/api/projects/{project_id}/tasks",
        headers=api_headers(),
        params={"fields": "all", "page_size": 1000},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    tasks = payload.get("results", payload) if isinstance(payload, dict) else payload
    export_path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"project_id": project_id, "export_file": str(export_path), "tasks": len(tasks)}


def percent_to_bbox(value: dict[str, Any], width: int, height: int) -> list[int]:
    x_min = float(value["x"]) / 100.0 * width
    y_min = float(value["y"]) / 100.0 * height
    x_max = x_min + float(value["width"]) / 100.0 * width
    y_max = y_min + float(value["height"]) / 100.0 * height
    return [round(x_min), round(y_min), round(x_max), round(y_max)]


def values_by_region(results: list[dict[str, Any]], region_id: str, from_name: str) -> list[Any]:
    values = []
    for result in results:
        if result.get("id") != region_id or result.get("from_name") != from_name:
            continue
        value = result.get("value", {})
        if "text" in value:
            values.extend(value.get("text") or [])
        if "choices" in value:
            values.extend(value.get("choices") or [])
    return values


def convert_export_impl(export_file: str | None = None, output: str | None = None) -> dict[str, Any]:
    if export_file is None:
        files = sorted(EXPORT_DIR.glob("project_*_export.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise RuntimeError("No Label Studio export JSON found. Run export_annotations first.")
        export_path = files[0]
    else:
        export_path = resolve_existing_path(export_file)
    output_path = Path(output) if output else PROJECT_JSON
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    pages = []
    for task in exported:
        data = task.get("data", {})
        page_id = data.get("page_id")
        width = int(data.get("width") or 0)
        height = int(data.get("height") or 0)
        annotations = task.get("annotations") or []
        results: list[dict[str, Any]] = []
        if annotations:
            results = annotations[-1].get("result", [])
        elif task.get("predictions"):
            results = task["predictions"][-1].get("result", [])

        columns = []
        for result in results:
            if result.get("from_name") != "label" or result.get("type") != "rectanglelabels":
                continue
            labels = result.get("value", {}).get("rectanglelabels", [])
            region_id = result.get("id") or f"{page_id}_col_{len(columns):03d}"
            order_values = values_by_region(results, region_id, "reading_order")
            orientation_values = values_by_region(results, region_id, "orientation")
            tag_values = values_by_region(results, region_id, "degradation_tags")
            try:
                reading_order = int(str(order_values[0]).strip()) if order_values else len(columns)
            except ValueError:
                reading_order = len(columns)
            orientation = orientation_values[0] if orientation_values and orientation_values[0] in ORIENTATIONS else "correct"
            degradation_tags = [tag for tag in tag_values if tag in DEGRADATION_TAGS]
            ignore = "Ignore" in labels
            columns.append(
                {
                    "column_id": f"{page_id}_col_{len(columns):03d}",
                    "bbox": percent_to_bbox(result.get("value", {}), width, height),
                    "reading_order": reading_order,
                    "orientation": orientation,
                    "transcript": "",
                    "degradation_tags": degradation_tags,
                    "ignore": ignore,
                    "notes": "converted from Label Studio export",
                }
            )
        columns.sort(key=lambda item: item["reading_order"])
        active_idx = 0
        ignored_idx = 0
        for column in columns:
            if column.get("ignore", False):
                column["reading_order"] = len(columns) + ignored_idx
                column["column_id"] = f"{page_id}_ignore_{ignored_idx:03d}"
                ignored_idx += 1
            else:
                column["reading_order"] = active_idx
                column["column_id"] = f"{page_id}_col_{active_idx:03d}"
                active_idx += 1
        pages.append(
            {
                "page_id": page_id,
                "image_path": data.get("original_image_path", ""),
                "split": data.get("split", "val"),
                "width": width,
                "height": height,
                "columns": columns,
            }
        )
    payload = {
        "dataset": {
            "name": "Traditional Mongolian Page-Level OCR",
            "version": "label-studio-val-v1",
            "split_manifest": str(MANIFEST),
            "notes": f"Converted from Label Studio export: {export_path}",
        },
        "pages": pages,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(output_path), "pages": len(pages), "source_export": str(export_path)}


def validate_impl(annotations: str | None = None) -> dict[str, Any]:
    annotation_path = resolve_existing_path(annotations) if annotations else PROJECT_JSON
    command = [str(PYTHON), str(VALIDATE_SCRIPT), "--annotations", str(annotation_path), "--manifest", str(MANIFEST)]
    completed = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True)
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "annotations": str(annotation_path),
    }


def json_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def resolve_existing_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute() or path.exists():
        return path
    for base in (ROOT, ROOT.parent):
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


@mcp.tool(name="label_studio_start_label_studio")
def start_label_studio(port: int = DEFAULT_PORT, open_browser: bool = False) -> str:
    """Start the local Label Studio server for OCR page annotation."""
    return json_response(start_label_studio_process(port=port, open_browser=open_browser))


@mcp.tool(name="label_studio_prepare_val_tasks")
def prepare_val_tasks() -> str:
    """Convert validation TIFF pages to PNG and create Label Studio import tasks."""
    return json_response(prepare_tasks("val"))


@mcp.tool(name="label_studio_create_or_open_project")
def create_or_open_project() -> str:
    """Create or reuse the OCR validation project with the fixed labeling interface."""
    return json_response(create_or_open_project_impl())


@mcp.tool(name="label_studio_import_val_tasks")
def import_val_tasks(project_id: int | None = None) -> str:
    """Import the prepared validation tasks and bootstrap predictions."""
    return json_response(import_tasks_impl(project_id=project_id, split="val"))


@mcp.tool(name="label_studio_open_project")
def open_label_studio_project(project_id: int | None = None) -> str:
    """Open the OCR validation project in the default browser."""
    return json_response(open_project_impl(project_id=project_id))


@mcp.tool(name="label_studio_export_annotations")
def export_annotations(project_id: int | None = None) -> str:
    """Export Label Studio annotations for the OCR validation project."""
    return json_response(export_annotations_impl(project_id=project_id))


@mcp.tool(name="label_studio_convert_export_to_project_json")
def convert_export_to_project_json(export_file: str | None = None, output: str | None = None) -> str:
    """Convert Label Studio export JSON into page_level_annotations.json format."""
    return json_response(convert_export_impl(export_file=export_file, output=output))


@mcp.tool(name="label_studio_validate_project_annotations")
def validate_project_annotations(annotations: str | None = None) -> str:
    """Validate converted page-level annotation JSON against the project manifest."""
    return json_response(validate_impl(annotations=annotations))


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true", help="Start Label Studio.")
    parser.add_argument("--prepare", action="store_true", help="Prepare validation PNGs and import task JSON.")
    parser.add_argument("--create-project", action="store_true", help="Create or reuse the Label Studio project.")
    parser.add_argument("--import-tasks", action="store_true", help="Import prepared validation tasks.")
    parser.add_argument("--open", action="store_true", help="Open the project in the browser.")
    parser.add_argument("--export", action="store_true", help="Export project annotations.")
    parser.add_argument("--convert", action="store_true", help="Convert latest export to page_level_annotations.json.")
    parser.add_argument("--validate", action="store_true", help="Validate page_level_annotations.json.")
    parser.add_argument("--setup-all", action="store_true", help="Start server, prepare data, create project, import tasks, and open browser.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    outputs = []
    if args.setup_all:
        outputs.append(start_label_studio_process(port=args.port, open_browser=False))
        outputs.append(prepare_tasks("val"))
        project = create_or_open_project_impl()
        outputs.append(project)
        outputs.append(import_tasks_impl(project_id=int(project["id"]), split="val"))
        outputs.append(open_project_impl(project_id=int(project["id"])))
    else:
        if args.start:
            outputs.append(start_label_studio_process(port=args.port, open_browser=False))
        if args.prepare:
            outputs.append(prepare_tasks("val"))
        if args.create_project:
            outputs.append(create_or_open_project_impl())
        if args.import_tasks:
            outputs.append(import_tasks_impl(split="val"))
        if args.open:
            outputs.append(open_project_impl())
        if args.export:
            outputs.append(export_annotations_impl())
        if args.convert:
            outputs.append(convert_export_impl())
        if args.validate:
            outputs.append(validate_impl())
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli()
    else:
        mcp.run()
