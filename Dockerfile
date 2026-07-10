# Root image for the triton repo, built by CI (pr-build / security-scan /
# release). The Triton SERVER itself uses the upstream nvcr.io/nvidia/tritonserver
# image (see deploy/); this image packages the repo's one buildable artifact —
# the Go client tool in clients/go (see also clients/go/cmd/triton-detect).
#
# Inference is offloaded to Triton, so the client needs no Python/ONNX/OpenCV:
# it is a fully static (CGO_ENABLED=0, netgo) Go binary shipped on
# distroless/static:nonroot — no shell, no package manager, non-root, and
# effectively no OS CVE surface (which keeps the security scan clean).

# Declared so the build-args the reusable CI passes (project/github_username/
# github_token) are consumed without a warning. They are unused here because the
# Go client has no private module dependencies to authenticate for.
ARG project
ARG github_username
ARG github_token

FROM golang:1.25-bookworm AS builder
WORKDIR /src
# The Go module lives in clients/go; copy just that and build the client tool.
COPY clients/go/ ./
RUN CGO_ENABLED=0 go build -trimpath -tags timetzdata \
    -ldflags "-s -w" -o /out/triton-detect ./cmd/triton-detect

FROM gcr.io/distroless/static:nonroot
LABEL org.opencontainers.image.source=https://github.com/uug-ai/triton
LABEL AUTHOR=uug-ai
COPY --from=builder /out/triton-detect /triton-detect
USER nonroot:nonroot
ENTRYPOINT ["/triton-detect"]
