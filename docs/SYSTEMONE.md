# BioJev System One compatibility layer

This exposes existing BioJev NLI checkpoints through a Jev/Ollama-shaped
`POST /v1/systemone` API. It does not retrain the model.

## Semantics

Each candidate is converted into an NLI pair:

- premise = serialized `state`
- hypothesis = `Decision question: <instructions>\nCandidate answer: <candidate>`

Candidate support:

`entailment_logit - contradiction_logit`

Probabilities are a softmax over candidate supports. Choice/score confidence uses
normalized entropy `1 - H(p)/ln(N)`.

This is an engineering compatibility bridge, not a native System One checkpoint.
It requires separate evaluation before making accuracy claims.

## Install HTTP dependencies

```bash
pip install fastapi uvicorn
```

## Serve BioJev-4B

```bash
python scripts/serve_systemone.py \
  --checkpoint outputs/biojev/4b_full/stage-02-biomedical/final \
  --model-name biojev-4b \
  --load-in-4bit \
  --host 0.0.0.0 \
  --port 8000
```

## Request

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "biojev-4b",
    "state": "The patient has fever, productive cough, and a new lobar infiltrate.",
    "questions": {
      "diagnosis": {
        "type": "choice",
        "instructions": "Which diagnosis is best supported?",
        "criteria": {
          "pneumonia": "Community-acquired pneumonia",
          "asthma": "Acute asthma exacerbation",
          "migraine": "Migraine"
        }
      },
      "infection": {
        "type": "noul",
        "instructions": "Is an infectious pulmonary process supported?"
      },
      "severity": {
        "type": "score",
        "instructions": "How severe is the presentation?",
        "criteria": ["Low", "Moderate", "High"]
      }
    }
  }'
```

## Nano

```bash
python scripts/serve_systemone.py \
  --checkpoint outputs/biojev/sprint6_nano/full/stage-02-biomedical/final \
  --model-name biojev-nano \
  --load-in-4bit \
  --port 8000
```
