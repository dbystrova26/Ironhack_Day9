# News Summarizer

## What The Project Does

Multi-provider news summarizer that fetches articles from NewsAPI, summarizes them
with OpenAI, analyzes sentiment with Anthropic, tracks estimated token costs, and
supports synchronous or threaded async article processing.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create or update `.env` in this folder:

```env
OPENAI_API_KEY="your-openai-key"
ANTHROPIC_API_KEY="your-anthropic-key"
NEWS_API_KEY="your-newsapi-key"
ENVIRONMENT="development"
MAX_RETRIES="3"
REQUEST_TIMEOUT="30"
DAILY_BUDGET="5.00"
OPENAI_MODEL="gpt-4o-mini"
ANTHROPIC_MODEL="claude-haiku-4-5-20251001"
```

`NEWS_API_KEY` must be a valid key from NewsAPI. The current app uses
`NEWS_API_KEY`, not `NEWSAPI_KEY`.

## How To Run

Run the full interactive app:

```bash
python main.py
```

Run individual checkpoints:

```bash
python config.py
python news_api.py
python llm_providers.py
python summarizer.py
pytest test_summarizer.py -v
```

To test async processing, uncomment this line at the bottom of `summarizer.py`:

```python
# asyncio.run(test_async())
```

## Example Output

```text
================================================================================
NEWS SUMMARIZER - Multi-Provider Edition
================================================================================

Enter news category (technology/business/health/general): technology
How many articles to process? (1-10): 2
Use async processing? (y/n): n

Fetching 2 articles from category: technology
Processing 2 articles...

NEWS SUMMARY REPORT
================================================================================

1. Example technology headline
   Source: Example Source | Published: 2026-04-30T12:00:00Z
   URL: https://example.com/article

   SUMMARY:
   The article explains the main development in two or three concise sentences.

   SENTIMENT:
   Overall sentiment is neutral with moderate confidence.

================================================================================
COST SUMMARY
================================================================================
Total requests: 4
Total cost: $0.0004
Total tokens: 850
  Input: 650
  Output: 200
Average cost per request: $0.000100
================================================================================
```

If NewsAPI returns `401 Unauthorized`, replace `NEWS_API_KEY` with a valid key.

## Cost Analysis

Costs are estimated locally in `CostTracker` using per-million-token pricing in
`llm_providers.py`. The estimate is useful for learning and budgeting, but the
provider dashboard remains the source of truth for billing.

The default models are:

- OpenAI: `gpt-4o-mini`
- Anthropic: `claude-haiku-4-5-20251001`

The app checks `DAILY_BUDGET` after each LLM request and raises an exception if
the estimated total reaches or exceeds that budget.

## File Map

```text
news-summarizer/
+-- .env                  # Local API keys and runtime settings; ignored by git
+-- .gitignore            # Ignores secrets, caches, venvs, and generated results
+-- README.md             # Setup, run instructions, examples, and file map
+-- requirements.txt      # Python dependencies
+-- config.py             # Loads and validates environment configuration
+-- news_api.py           # NewsAPI integration and rate-limited headline fetcher
+-- llm_providers.py      # OpenAI/Anthropic clients, fallback, token counting, cost tracking
+-- summarizer.py         # Article summarization, sentiment analysis, reports, async variant
+-- test_summarizer.py    # Unit tests with mocked API calls
+-- main.py               # Interactive command-line application
+-- apitest.png           # Screenshot or visual artifact from API testing
+-- __pycache__/          # Generated Python bytecode cache
+-- pytest-cache-files-*/ # Generated pytest cache artifact
```
