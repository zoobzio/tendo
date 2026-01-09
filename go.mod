module github.com/zoobzio/tendo

go 1.25.0

require (
	github.com/zoobzio/capitan v0.1.0
	github.com/zoobzio/pipz v0.1.3
	github.com/zoobzio/tendo/cpu v0.0.0-20260109175738-116066c8fcf8
	github.com/zoobzio/tendo/cuda v0.0.0-20260109175738-116066c8fcf8
)

require (
	github.com/google/uuid v1.6.0 // indirect
	github.com/zoobzio/clockz v0.0.2 // indirect
	gonum.org/v1/gonum v0.16.0 // indirect
)

replace (
	github.com/zoobzio/tendo/cpu => ./cpu
	github.com/zoobzio/tendo/cuda => ./cuda
	github.com/zoobzio/tendo/models => ./models
	github.com/zoobzio/tendo/nn => ./nn
	github.com/zoobzio/tendo/testing => ./testing
)
