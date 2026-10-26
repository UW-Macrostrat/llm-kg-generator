MODELS := qwen3 gpt-oss glm-5 kimi minimax-m2 deepseek-v4-flash gemma
CONSTRAINTS_FILE := constraints.json

VENV_PYTHON := /home/david/anaconda3/envs/llm-kg/bin/python

SHORT_ID := 32749
LONG_ID := 25215


.PHONY: short long stack $(addsuffix -short,$(MODELS)) $(addsuffix -long,$(MODELS))

short: $(addsuffix -short,$(MODELS))

long: $(addsuffix -long,$(MODELS))


define MODEL_TARGETS

$(1)-short:
	$(VENV_PYTHON) -m pipeline \
		--model $(1) \
		--source-id $(SHORT_ID) \
		--upload \
		--constraints $(CONSTRAINTS_FILE) \
		--max-tokens 5000

$(1)-long:
	$(VENV_PYTHON) -m pipeline \
		--model $(1) \
		--source-id $(LONG_ID) \
		--upload \
		--constraints $(CONSTRAINTS_FILE) \
		--max-tokens 5000

endef


$(foreach model,$(MODELS),$(eval $(call MODEL_TARGETS,$(model))))


stack:
	$(VENV_PYTHON) -m pipeline \
		--source-id 23578 23579 23580 23581 23582 23583 23584 23585 23586 23587 \
		--upload \
		--constraints $(CONSTRAINTS_FILE)