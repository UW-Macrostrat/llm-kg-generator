test-gpt:
	python -m pipeline --model "openai/gpt-oss-20b" --constraints constraints.json --source-id 23578 --max-tokens 4096 --upload

test-qwen:
	python -m pipeline --constraints constraints.json --source-id 23578 --upload

dry-run:
	python -m pipeline  --dry-run max 1

stack:
	python3 -m kg_pipeline --model "Qwen/Qwen3-32B-AWQ" --source-id 23578 23579 23580 23581 23582 23583 23584 23585 23586 23587 