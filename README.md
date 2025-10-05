pip install -r requirements.txt
uvicorn app.server:app --reload --port 8080
curl -s -X POST localhost:8080/agent -H 'content-type: application/json' \
  -d '{"query":"What is the refund period and API rate limit?"}' | jq