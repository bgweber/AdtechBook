package console

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/bgweber/adtechbook/code/dsp/internal/model"
)

// Trainer starts training runs on the training service (code/training,
// `python -m adtrain serve`) and, when asked, deploys the result.
//
// Training is Python and runs in its own process or container, as in a
// production DSP, where training jobs never share a machine with the
// bidder. The console only starts runs and shows their results.
type Trainer struct {
	URL    string // such as http://trainer:8090; empty = no training service
	Models *model.Registry

	mu     sync.Mutex
	job    TrainJob
	parity json.RawMessage
}

// TrainRequest is what the console sends to start a run.
type TrainRequest struct {
	Deploy bool `json:"deploy"` // deploy the model if its checks pass
	// Optional: which simulated days to train and evaluate on. By
	// default, the trainer uses every day but the last for training and
	// the last for evaluation.
	TrainDays []int `json:"train_days,omitempty"`
	EvalDay   *int  `json:"eval_day,omitempty"`
	// Constant publishes a model that predicts this click rate for every
	// impression: the broken model of the Chapter 12 exercise.
	Constant *float64 `json:"constant,omitempty"`
}

// TrainJob is the state of the latest run.
type TrainJob struct {
	State    string              `json:"state"` // idle, running, done, failed
	Started  string              `json:"started,omitempty"`
	Finished string              `json:"finished,omitempty"`
	Seconds  float64             `json:"seconds,omitempty"`
	Version  string              `json:"version,omitempty"`
	Passed   bool                `json:"passed"`
	Message  string              `json:"message,omitempty"`
	Deploy   *model.DeployResult `json:"deploy,omitempty"`
	Log      []string            `json:"log,omitempty"`
}

// Status returns the latest run.
func (t *Trainer) Status() TrainJob {
	t.mu.Lock()
	defer t.mu.Unlock()
	j := t.job
	if j.State == "" {
		j.State = "idle"
	}
	j.Log = append([]string(nil), t.job.Log...)
	return j
}

// Start begins a run unless one is already going.
func (t *Trainer) Start(req TrainRequest) error {
	if t.URL == "" {
		return fmt.Errorf("no training service: start the bidder with -trainer-url, " +
			"or run `docker compose up`, which starts one")
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.job.State == "running" {
		return fmt.Errorf("a training run is already going")
	}
	t.job = TrainJob{State: "running", Started: time.Now().UTC().Format(time.RFC3339)}
	go t.run(req)
	return nil
}

func (t *Trainer) run(req TrainRequest) {
	start := time.Now()
	body, _ := json.Marshal(req)
	client := &http.Client{Timeout: 30 * time.Minute}
	var out struct {
		Version string   `json:"version"`
		Passed  bool     `json:"passed"`
		Error   string   `json:"error"`
		Log     []string `json:"log"`
	}
	resp, err := client.Post(t.URL+"/train", "application/json", bytes.NewReader(body))
	if err == nil {
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if jerr := json.Unmarshal(raw, &out); jerr != nil {
			err = fmt.Errorf("HTTP %d from the trainer: %s", resp.StatusCode, bytes.TrimSpace(raw))
		} else if out.Error != "" {
			err = fmt.Errorf("%s", out.Error)
		}
	}
	var dep *model.DeployResult
	if err == nil && req.Deploy {
		if out.Passed {
			r := t.Models.Deploy(out.Version, false)
			dep = &r
		} else {
			dep = &model.DeployResult{Version: out.Version,
				Message: "not deployed: the model failed its checks. Review it on the Models tab."}
		}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.job.Finished = time.Now().UTC().Format(time.RFC3339)
	t.job.Seconds = time.Since(start).Seconds()
	t.job.Log = out.Log
	t.job.Deploy = dep
	if err != nil {
		t.job.State, t.job.Message = "failed", err.Error()
		return
	}
	t.job.State, t.job.Version, t.job.Passed = "done", out.Version, out.Passed
	t.job.Message = "trained " + out.Version
}

// Parity runs the feature parity check on the training service (Chapter
// 8) and keeps the latest report for the console.
func (t *Trainer) Parity() (json.RawMessage, error) {
	if t.URL == "" {
		return nil, fmt.Errorf("no training service: start the bidder with -trainer-url")
	}
	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Post(t.URL+"/parity", "application/json", nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d from the trainer: %s", resp.StatusCode, bytes.TrimSpace(raw))
	}
	t.mu.Lock()
	t.parity = raw
	t.mu.Unlock()
	return raw, nil
}

// LastParity is the latest parity report, or nil.
func (t *Trainer) LastParity() json.RawMessage {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.parity
}
