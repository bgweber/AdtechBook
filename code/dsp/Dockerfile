# The reference bidder: a small static binary on a minimal base image.
FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /bidder ./cmd/bidder
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /loadgen ./cmd/loadgen

FROM alpine:3.20
COPY --from=build /bidder /usr/local/bin/bidder
# The load test (Chapter 17): docker compose run --rm loadgen
COPY --from=build /loadgen /usr/local/bin/loadgen
# Runs as root so it can write the ledger to the ./out folder mounted from
# your machine. A production image would run as an unprivileged user.
EXPOSE 8080 8081
ENTRYPOINT ["bidder"]
