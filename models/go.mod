module github.com/zoobzio/tendo/models

go 1.25.0

require (
	github.com/daulet/tokenizers v1.24.0
	github.com/zoobzio/tendo v0.0.0
	github.com/zoobzio/tendo/nn v0.0.0
)

require (
	github.com/google/uuid v1.6.0 // indirect
	github.com/zoobzio/capitan v0.1.0 // indirect
	github.com/zoobzio/clockz v0.0.2 // indirect
	github.com/zoobzio/pipz v0.1.3 // indirect
)

replace (
	github.com/zoobzio/tendo => ../
	github.com/zoobzio/tendo/cpu => ../cpu
	github.com/zoobzio/tendo/cuda => ../cuda
	github.com/zoobzio/tendo/nn => ../nn
)
