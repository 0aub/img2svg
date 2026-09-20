IMAGE ?= img2svg:local

.PHONY: build test shell example clean

build:
	docker build -t $(IMAGE) .

test: build
	docker run --rm -v "$(PWD):/src" -w /src --entrypoint sh $(IMAGE) \
	  -c "pip install --no-cache-dir -q pytest && python -m pytest -q"

shell: build
	docker run --rm -it -v "$(PWD):/work" --entrypoint bash $(IMAGE)

example: build
	docker run --rm -u $$(id -u):$$(id -g) -v "$(PWD)/examples:/work" $(IMAGE) \
	  honey-heart.png -o out/honey-heart.svg --regularize container --mono --mark --report

clean:
	rm -rf examples/out .pytest_cache **/__pycache__
