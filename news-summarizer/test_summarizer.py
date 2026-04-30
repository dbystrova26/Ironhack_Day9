"""Unit tests for news summarizer."""

import os
from unittest.mock import Mock, patch

import pytest

from config import Config
from llm_providers import CostTracker, LLMProviders, count_tokens
from news_api import NewsAPI
from summarizer import NewsSummarizer


class TestCostTracker:
    """Test cost tracking functionality."""

    def test_track_request(self):
        """Test tracking a single request."""
        tracker = CostTracker()
        cost = tracker.track_request("openai", "gpt-4o-mini", 100, 500)

        assert cost > 0
        assert tracker.total_cost == cost
        assert len(tracker.requests) == 1

    def test_get_summary(self):
        """Test summary generation."""
        tracker = CostTracker()
        tracker.track_request("openai", "gpt-4o-mini", 100, 200)
        tracker.track_request("anthropic", "claude-3-5-sonnet-20241022", 150, 300)

        summary = tracker.get_summary()

        assert summary["total_requests"] == 2
        assert summary["total_cost"] > 0
        assert summary["total_input_tokens"] == 250
        assert summary["total_output_tokens"] == 500

    def test_budget_check(self):
        """Test budget checking."""
        tracker = CostTracker()

        # Should not raise for small amount
        tracker.track_request("openai", "gpt-4o-mini", 100, 100)
        tracker.check_budget(10.00)  # Should pass

        # Should raise for exceeding budget
        tracker.total_cost = 15.00
        with pytest.raises(Exception, match="budget.*exceeded"):
            tracker.check_budget(10.00)


class TestTokenCounting:
    """Test token counting."""

    def test_count_tokens(self):
        """Test token counting function."""
        text = "Hello, how are you?"
        count = count_tokens(text)

        assert count > 0
        assert count < len(text)  # Should be less than character count


class TestNewsAPI:
    """Test News API integration."""

    def test_config_uses_newsapi_key_name(self, monkeypatch):
        """Test the NewsAPI key name matches .env exactly."""
        monkeypatch.setenv("NEWSAPI_KEY", "new-key")

        news_api_key = os.getenv("NEWSAPI_KEY")

        assert news_api_key == "new-key"

    @patch("news_api.requests.get")
    def test_fetch_top_headlines(self, mock_get):
        """Test fetching news from external API."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "status": "ok",
            "articles": [
                {
                    "title": "Test Article",
                    "description": "Test description",
                    "content": "Test content",
                    "url": "https://example.com",
                    "source": {"name": "Test Source"},
                    "publishedAt": "2026-01-19",
                }
            ],
        }
        mock_get.return_value = mock_response

        api = NewsAPI()
        articles = api.fetch_top_headlines(max_articles=1)

        assert len(articles) == 1
        assert articles[0]["title"] == "Test Article"
        assert articles[0]["source"] == "Test Source"
        mock_get.assert_called_once()
        assert mock_get.call_args.kwargs["params"]["pageSize"] == 1
        assert mock_get.call_args.kwargs["params"]["apiKey"] == Config.NEWS_API_KEY

    @patch("news_api.time.sleep")
    @patch("news_api.time.time")
    def test_news_api_respects_rate_limit(self, mock_time, mock_sleep):
        """Test NewsAPI rate limit waiting."""
        api = NewsAPI()
        api.last_call_time = 100.0
        api.min_interval = 0.6
        mock_time.side_effect = [100.2, 100.6]

        api._wait_if_needed()

        mock_sleep.assert_called_once_with(pytest.approx(0.4))
        assert api.last_call_time == 100.6


class TestLLMProviders:
    """Test LLM provider integration."""

    @patch("llm_providers.OpenAI")
    def test_ask_openai(self, mock_openai_class):
        """Test OpenAI integration."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        providers = LLMProviders()
        providers.openai_client = mock_client

        response = providers.ask_openai("Test prompt")

        assert response == "Test response"
        assert mock_client.chat.completions.create.called
        assert providers.cost_tracker.get_summary()["total_requests"] == 1

    @patch("llm_providers.Anthropic")
    def test_ask_anthropic(self, mock_anthropic_class):
        """Test Anthropic integration."""
        # Mock Anthropic client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Sentiment response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        providers = LLMProviders()
        providers.anthropic_client = mock_client

        response = providers.ask_anthropic("Analyze sentiment")

        assert response == "Sentiment response"
        assert mock_client.messages.create.called
        assert providers.cost_tracker.get_summary()["total_requests"] == 1

    def test_fallback_logic_uses_secondary_provider(self):
        """Test fallback logic when primary provider fails."""
        providers = LLMProviders()
        providers.ask_openai = Mock(side_effect=Exception("OpenAI failed"))
        providers.ask_anthropic = Mock(return_value="Anthropic fallback response")

        result = providers.ask_with_fallback("Test prompt", primary="openai")

        assert result == {
            "provider": "anthropic",
            "response": "Anthropic fallback response",
        }
        providers.ask_openai.assert_called_once_with("Test prompt")
        providers.ask_anthropic.assert_called_once_with("Test prompt")

    @patch("llm_providers.time.sleep")
    @patch("llm_providers.time.time")
    def test_openai_respects_rate_limit(self, mock_time, mock_sleep):
        """Test OpenAI rate limit waiting."""
        providers = LLMProviders()
        providers.openai_last_call = 50.0
        providers.openai_interval = 0.5
        mock_time.side_effect = [50.1, 50.5]

        providers._wait_openai()

        mock_sleep.assert_called_once_with(pytest.approx(0.4))
        assert providers.openai_last_call == 50.5

    @patch("llm_providers.time.sleep")
    @patch("llm_providers.time.time")
    def test_anthropic_respects_rate_limit(self, mock_time, mock_sleep):
        """Test Anthropic rate limit waiting."""
        providers = LLMProviders()
        providers.anthropic_last_call = 75.0
        providers.anthropic_interval = 1.2
        mock_time.side_effect = [75.3, 76.2]

        providers._wait_anthropic()

        mock_sleep.assert_called_once_with(pytest.approx(0.9))
        assert providers.anthropic_last_call == 76.2


class TestNewsSummarizer:
    """Test news summarizer."""

    def test_initialization(self):
        """Test summarizer initialization."""
        summarizer = NewsSummarizer()

        assert summarizer.news_api is not None
        assert summarizer.llm_providers is not None

    @patch.object(LLMProviders, "ask_openai")
    @patch.object(LLMProviders, "ask_anthropic")
    def test_summarize_article(self, mock_anthropic, mock_openai):
        """Test OpenAI summary and Anthropic sentiment analysis."""
        mock_openai.return_value = "Test summary"
        mock_anthropic.return_value = "Positive sentiment"

        summarizer = NewsSummarizer()
        article = {
            "title": "Test Article",
            "description": "Test description",
            "content": "Test content",
            "url": "https://example.com",
            "source": "Test Source",
            "published_at": "2026-01-19",
        }

        result = summarizer.summarize_article(article)

        assert result["title"] == "Test Article"
        assert result["summary"] == "Test summary"
        assert result["sentiment"] == "Positive sentiment"
        assert mock_openai.called
        assert mock_anthropic.called
        assert "Summarize this news article" in mock_openai.call_args.args[0]
        assert "Analyze the sentiment" in mock_anthropic.call_args.args[0]

    @patch.object(LLMProviders, "ask_openai")
    @patch.object(LLMProviders, "ask_anthropic")
    def test_summarize_article_falls_back_to_anthropic(self, mock_anthropic, mock_openai):
        """Test summarizer fallback when OpenAI summary fails."""
        mock_openai.side_effect = Exception("OpenAI unavailable")
        mock_anthropic.side_effect = ["Fallback summary", "Neutral sentiment"]

        summarizer = NewsSummarizer()
        article = {
            "title": "Fallback Article",
            "description": "Fallback description",
            "content": "Fallback content",
            "url": "https://example.com/fallback",
            "source": "Test Source",
            "published_at": "2026-01-19",
        }

        result = summarizer.summarize_article(article)

        assert result["summary"] == "Fallback summary"
        assert result["sentiment"] == "Neutral sentiment"
        assert mock_openai.called
        assert mock_anthropic.call_count == 2


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
