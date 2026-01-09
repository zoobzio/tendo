module github.com/zoobzio/tendo/testing

go 1.25.0

require (
	github.com/zoobzio/pipz v0.1.3
	github.com/zoobzio/tendo v0.0.0
)

require (
	github.com/google/uuid v1.6.0 // indirect
	github.com/zoobzio/capitan v0.1.0 // indirect
	github.com/zoobzio/clockz v0.0.2 // indirect
	github.com/zoobzio/tendo/cpu v0.0.0-20260106225412-f59220386cbd // indirect
	github.com/zoobzio/tendo/cuda v0.0.0-20260106225412-f59220386cbd // indirect
)

replace (
	github.com/zoobzio/tendo => ../
	github.com/zoobzio/tendo/cpu => ../cpu
	github.com/zoobzio/tendo/cuda => ../cuda
)
