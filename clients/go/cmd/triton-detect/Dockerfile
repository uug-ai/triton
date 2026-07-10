# Minimal, CVE-minimal image for a Go service that talks to Triton.
#
# Because inference runs in the Triton server, this service needs NO Python, ONNX
# runtime or OpenCV — just a static Go binary. It is built with CGO disabled
# (pure Go, netgo) and shipped on distroless/static:nonroot, which has no shell,
# no package manager and effectively no OS CVE surface (compare an alpine or a
# python:* base). `scratch` works too if you add ca-certificates yourself.
#
# Build from the module root (clients/go):
#   docker build -f cmd/triton-detect/Dockerfile -t triton-detect .

FROM golang:1.25-bookworm AS builder
WORKDIR /src
COPY . .
# Fully static, stripped, reproducible binary; timetzdata bakes in the tz database
# so the runtime image needs no OS files at all.
RUN CGO_ENABLED=0 go build -trimpath -tags timetzdata \
    -ldflags "-s -w" -o /out/triton-detect ./cmd/triton-detect

FROM gcr.io/distroless/static:nonroot
LABEL org.opencontainers.image.source=https://github.com/uug-ai/triton
LABEL AUTHOR=uug-ai
COPY --from=builder /out/triton-detect /triton-detect
USER nonroot:nonroot
ENTRYPOINT ["/triton-detect"]
