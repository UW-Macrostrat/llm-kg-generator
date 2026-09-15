# Macrostrat queued extraction pipeline

This replaces the attached single-paragraph script with a complete asynchronous
pipeline. It fetches `source_text` in pages of **100 rows**, prepares prompts,
keeps a configurable number of vLLM requests outstanding, validates results,
saves them to disk, and uploads each successful source through `/record_run`.

Here, **one source means one row of `source_text`, normally one paragraph**.
It does not mean every paragraph belonging to one paper. This matches the
input record and upload contract in the original script.

## Run it

Use Python 3.11 or newer. In the extracted `macrostrat_pipeline` directory:

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env
```

If using your existing `.env`, merge the example's relevant values into it.
Use `SOURCE_TEXT_URL=https://dev.macrostrat.org/api/pg/source_text` without the
old `?id=eq.19744` filter. Existing environment variables take precedence over
`.env`; select a different file with `--env-file /path/to/.env`.

Start your existing vLLM server in a separate terminal, or use the included
launcher in the environment where vLLM is installed:

```bash
bash start_vllm.sh
```

The client package needs only HTTPX and python-dotenv. It does not load another
model or import torch/vLLM into the pipeline process.

Try the original source without uploading:

```bash
python3 -m kg_pipeline --test --dry-run --source-id 19744 --output runs/one-source
```

Then process all source text:

```bash
# All sources, JSON output only
python3 -m kg_pipeline --dry-run --output runs/all-local

# All sources, saving locally and uploading through the upload workers
python3 -m kg_pipeline --output runs/all-uploaded
```

| Flags | Source rows processed | `/record_run` uploads |
|---|---|---|
| Neither | All matching rows | One attempt per successful source |
| `--dry-run` | All matching rows | None |
| `--test` | One row | One attempt if successful |
| `--test --dry-run` | One row | None |

`--test` requests `limit=1`; it does not fetch a 100-row page and discard 99 rows.
Without `--source-id`, it selects the first available row by ascending ID.
A long source can require several LLM calls, but it still counts as one source
and produces at most one upload. `--source-id` also works without `--test`.

`--dry-run` still reads source/type APIs and calls the local tokenizer and LLM.
It never calls `/record_run`. Normal uploads begin when each source's local
result is saved; they overlap with inference instead of waiting for the full scan.

## Structured stages

| Stage | Module | Responsibility |
|---|---|---|
| Fetch | `kg_pipeline/fetch.py` | Read current type definitions and source-text pages using asynchronous HTTP and an ID cursor. |
| Preprocess | `kg_pipeline/preprocess.py` | Validate source fields, count prompt tokens, split long text, and enqueue ready LLM jobs. |
| Inference | `kg_pipeline/llm.py` | Send concurrent requests and pass response bytes downstream immediately. |
| Postprocess | `kg_pipeline/postprocess.py` | Parse/validate JSON, ground mentions in source text, map offsets, deduplicate overlaps, and build one source payload. |
| Local save | `kg_pipeline/storage.py` | Atomically save each source before it can be uploaded. |
| Upload | `kg_pipeline/upload.py` | POST saved payloads independently of inference; preserve uncertain outcomes. |
| Upload receipts | `pipeline.py` + `storage.py` | Persist upload responses through a separate queue/writer. |
| Run-step recording | `kg_pipeline/events.py` | Batch stage events from a dedicated queue into `run_steps.jsonl`. |

`pipeline.py` wires these stages together using bounded `asyncio.Queue` objects
and worker pools. `models.py` defines queue messages, `prompts.py` owns the
extraction contract, `http.py` owns retry behavior, and `config.py`/`cli.py`
own configuration and the command-line interface.

Each pool supervisor closes the next queue only after every worker in its own
pool has finished. This prevents an early-finishing worker from shutting down
downstream workers while another worker is still producing results. Normal
completion waits for all source, inference, saving, upload, and event work.

Run steps are recorded **locally**. The supplied script only defines a
`/record_run` upload contract, so this implementation does not invent a separate
remote run-step endpoint or add unsupported fields to the upload body.

## Keeping the LLM busy

The defaults are appropriate starting settings for your 4 GB RTX A500 with the
server configured for `--max-num-seqs 1`:

```bash
python3 -m kg_pipeline \
    --llm-workers 2 \
    --preprocess-workers 2 \
    --postprocess-workers 2 \
    --upload-workers 2 \
    --queue-size 200 \
    --inflight-sources 200
```

Two client workers can keep another request waiting while the server handles
one active sequence. Client concurrency does not increase the server's GPU
memory capacity or override `--max-num-seqs`. On a server with more available
capacity, increase `--llm-workers` gradually and compare throughput.

There is no barrier between source pages: the producer fetches another page
while earlier sources are being processed. Inference workers do no JSON
validation, source aggregation, filesystem writes, or uploads. Source JSON
decoding, result processing, and disk work run through background threads.
Persistent HTTP clients reuse connections, with separate connection pools for
source reads, vLLM, and uploads. HTTPX documents the importance of reusing async
clients for connection pooling in its [async guide](https://www.python-httpx.org/async/).

Queues and the in-flight source limit bound memory growth. If downstream saving
or uploading stays slower than inference long enough to fill its buffers,
backpressure intentionally slows upstream work. No finite-buffer pipeline can
guarantee 100% GPU utilization during sustained downstream failure. Increase
upload workers or resolve the slow stage when the progress counters show this.

Progress prints every 10 seconds by default. `--progress-seconds 0` disables it.
The summary reports elapsed time, generated tokens, completed jobs, saved
sources, uploads, and failures. There are no hardware throughput claims from
the mocked test suite.

## Context limits and extraction behavior

The client discovers the served model and its `max_model_len` from `/v1/models`.
If the server omits that value, it defaults to 2048; override it with
`--context-length` to match the server. This option does not reconfigure vLLM.

Preprocessing counts the actual chat prompt through vLLM's `/tokenize` endpoint,
including the system message and generation prefix. Every submitted job satisfies:

```text
prompt_tokens + max_tokens + token_margin <= context_length
```

Defaults reserve 1024 output tokens and a 32-token margin. If the type
descriptions alone exceed the remaining prompt budget, startup fails with an
actionable error before fetching sources. You can reduce `--max-tokens`, shorten
the prompt/type descriptions in `prompts.py`, or increase the actual server
context if memory permits. The tokenizer endpoint must be exposed by your vLLM
deployment; the client does not silently substitute a token estimate.

Long paragraphs are split into fitting spans without dropping source characters.
Chunk overlap defaults to 100 characters, capped at a quarter of each chunk.
Whitespace boundaries are preferred where possible. Offsets are mapped back
to the original `paragraph_text` and are zero-based Python character indices,
with an exclusive end: `paragraph_text[start_idx:end_idx]`.

Chunking can miss relationships whose evidence spans distant chunks. More
overlap or a larger context can help. It does not guarantee the same extraction
as seeing an entire long paragraph in one request.

The extraction schema retains the original six-relationship and twelve-entity
limits **per chunk**, as well as the original sampling parameters and disabled
Qwen thinking. Source-level merged results may exceed those limits. Requests
use the current `structured_outputs.json` field described in
[vLLM's structured-output documentation](https://docs.vllm.ai/en/latest/features/structured_outputs/).
There is no hard-coded example containing potentially obsolete database labels.

Validation retains the original exact-match/case-insensitive fallback behavior
and locates the first matching occurrence within each chunk. Unsupported
mentions/types are excluded and retained in the chunk's `rejected` array for
inspection. Malformed JSON, invalid response shapes, truncation, or a failed
chunk marks the source failed and prevents a partial upload. Successful empty
extractions still produce a valid source payload.

## Pagination scope

The full scan first reads the highest currently visible source ID. Subsequent
requests use `order=id.asc`, `limit=100`, and an ID range above the last processed
page and at or below that initial high-water mark. The range uses the comparison
operators documented by [PostgREST](https://docs.postgrest.org/en/stable/references/api/tables_views.html).

This avoids large offset scans and prevents a run from chasing newly appended
rows indefinitely. It assumes unique, increasing integer IDs. It tolerates
gaps and smaller server-enforced page sizes, continues past short pages, and
rejects non-advancing IDs. This is an ID-bounded scan, not a transactional
snapshot: rows updated or deleted during the scan can change what is returned.

`--after-id 12345` restricts the scan to higher IDs. It is not automatic resume:
out-of-order completions and previously failed sources make the largest saved
ID an unsafe resume cursor. A new run generates new run IDs and can upload
previously processed sources again.

## Files and failure handling

`--output` is now a **new directory**, rather than the single output filename
accepted by the old script. If omitted, a unique directory under `runs/` is
created. An existing output directory is rejected to avoid overwriting a run.

| File | Contents |
|---|---|
| `run.json` | Run ID, configuration without token values, exact vocabulary snapshot, prompt, schema, and model. |
| `records/source_<id>.json` | Original source, per-chunk raw responses, cleaned extraction, errors, timings, and upload-ready payload. Written before any POST. |
| `receipts/source_<id>.json` | Upload state and server response, separate from the immutable extraction record. |
| `run_steps.jsonl` | One JSON object per stage event; written continuously by the event queue. |
| `summary.json` | Final counters, elapsed time, status, and any fatal/fetch error. |
| `results.json` | One standard JSON object containing `run`, `results`, and `summary`. Upload receipts are folded into the results. |

The final JSON is assembled one source at a time, so the full dataset is never
loaded into RAM. Result ordering is unspecified. The individual record's
initial upload status can remain `pending`; its receipt and final `results.json`
provide the later upload status. Completed records remain available if the
process stops before final assembly.

Source reads, tokenizer calls, and inference requests retry temporary network
errors and HTTP 408/429/500/502/503/504 with bounded backoff. Deterministic 4xx
errors and invalid/truncated model output are not repeatedly retried.

Uploads make **one application-level POST attempt per successful source in a
run**. There is no automatic POST retry: the supplied endpoint's idempotency
contract is unknown, and retrying a response lost after commit could duplicate
data. Timeouts/transport failures and HTTP 5xx are recorded as `unknown`.
A `started` receipt is saved before the POST so a crash during submission is
also recognizable as uncertain. Reconcile those run IDs with the server before
replaying them. This does not claim exactly-once delivery across crashes.

Per-source failures are saved and other sources continue. A fetch failure stops
new fetching but drains already-fetched work. An unexpected worker error or a
disk failure cancels the other stages. Ctrl-C preserves completed writes and
attempts to finalize the JSON/summary; a hard kill can leave only the incremental
records and receipts. There is no automatic restart/replay mode.

Exit codes: `0` complete success, `2` completed with source/upload errors,
`1` startup/fetch/fatal failure, and `130` Ctrl-C.

## Validation

Run the network-free integration suite:

```bash
python3 -m unittest discover -s tests -v
```

Tests cover 100-row paging over 205 sources, all flag combinations, single-row
selection, short pages and gaps, concurrent inference during blocked uploads,
context-aware splitting with global offsets, retries, partial-source failure,
malformed output, pagination failure, uncertain uploads, credential scoping,
disk failure, and cancellation. They use mocked vLLM and Macrostrat services.
Live GPU throughput and the remote `/record_run` endpoint were not exercised
while building this project; the payload contract follows the supplied script.
