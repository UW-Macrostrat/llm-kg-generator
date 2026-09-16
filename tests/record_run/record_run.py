#!/usr/bin/env python3
"""Send sample_run.json to /record_run once; uses only Python's standard library."""
import argparse
import http.client
import json
import logging
import math
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

LOG = logging.getLogger("test_upload")
DEFAULT_URL = "https://macrostrat-xdd.dev.svc.macrostrat.org/record_run"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")
    if "REPLACE_WITH_" in json.dumps(payload):
        raise ValueError("Replace every REPLACE_WITH_ value in sample_run.json before uploading.")
    for key in ("run_id", "extraction_pipeline_id", "results"):
        if key not in payload:
            raise ValueError(f"Missing {key}.")
    if not isinstance(payload["results"], list) or not payload["results"]:
        raise ValueError("results must be a nonempty list.")
    matches = 0
    for result in payload["results"]:
        paragraph = result["text"]["paragraph_text"]
        for entity in result.get("just_entities", []):
            start, end = entity["start_idx"], entity["end_idx"]
            if not (type(start) is int and type(end) is int
                    and 0 <= start < end <= len(paragraph)
                    and paragraph[start:end] == entity["entity"]):
                raise ValueError(f"Invalid span for entity {entity.get('entity')!r}.")
            term_id = entity.get("macrostrat_terms_id")
            if term_id is not None:
                if type(term_id) is not int or term_id <= 0:
                    raise ValueError("macrostrat_terms_id must be a positive integer or null.")
                matches += 1
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, default=Path(__file__).with_name("sample_run.json"))
    parser.add_argument("--url", default=os.getenv("RECORD_RUN_URL", DEFAULT_URL))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--response-file", type=Path, default=Path("upload_response.json"))
    parser.add_argument("--check-only", action="store_true", help="Validate locally without sending.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("--timeout must be a finite positive number.")
        raw = args.payload.read_bytes()
        payload = json.loads(raw)
        matches = validate_payload(payload)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        LOG.error("Invalid payload or configuration: %s", exc)
        return 2
    LOG.info("run_id=%s results=%d supplied_matches=%d", payload["run_id"], len(payload["results"]), matches)
    if args.check_only:
        LOG.info("Local validation passed; database IDs and stored matches were not checked.")
        return 0

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = os.getenv("RECORD_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    LOG.info("POST %s (one attempt)", args.url)
    try:
        request = urllib.request.Request(args.url, data=raw, headers=headers, method="POST")
        opener = urllib.request.build_opener(NoRedirect)
        try:
            with opener.open(request, timeout=args.timeout) as response:
                status, body = response.status, response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            status, body = exc.code, exc.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
        LOG.error("Request failed: %s. Outcome may be unknown; no retry was made.", exc)
        return 1

    print(body)
    try:
        decoded = json.loads(body)
    except ValueError:
        decoded = body
    try:
        args.response_file.write_text(json.dumps({"status_code": status, "body": decoded}, indent=2) + "\n")
    except OSError as exc:
        LOG.error("HTTP %s received, but response could not be saved: %s", status, exc)
        return 1
    if not 200 <= status < 300:
        LOG.error("HTTP %s. The server may have committed part of the run; inspect the response before retrying.", status)
        return 1
    if not isinstance(decoded, dict) or not decoded.get("success"):
        LOG.error("HTTP %s without the expected success response; inspect %s.", status, args.response_file)
        return 1
    LOG.info("Server accepted the run (HTTP %s). Response saved to %s; stored matches still need verification.", status, args.response_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
