test-dry:
	python3 -m kg_pipeline   --test

test-upload:
	python3 -m kg_pipeline  --source-id 23578

test-gpt:
	python3 -m kg_pipeline  --model "openai/gpt-oss-20b" --source-id 23578 --max-tokens 2048

test-qwen:
	python3 -m kg_pipeline  --model "Qwen/Qwen3-32B-AWQ" --source-id 23578 --max-tokens 2048

dry-run:
	python3 -m kg_pipeline  --dry-run --max 10

qwen:
	python3 -m kg_pipeline --model "Qwen/Qwen3-32B-AWQ" --source-id 23578 23579 23580 23581 23582 23583 23584 23585 23586 23587 