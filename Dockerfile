FROM python:3.12-slim

LABEL org.opencontainers.image.title="img2svg" \
      org.opencontainers.image.description="Redraw flat and cel-shaded raster art as clean, verified SVG." \
      org.opencontainers.image.source="https://github.com/0aub/img2svg" \
      org.opencontainers.image.licenses="Apache-2.0"

# libcairo2 is the only system dependency: it backs the verification renderer.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libcairo2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml README.md ./
COPY img2svg ./img2svg
RUN pip install --no-cache-dir ".[render]"

# Everything the user passes is relative to this; mount your files here.
WORKDIR /work
ENTRYPOINT ["img2svg"]
CMD ["--help"]
