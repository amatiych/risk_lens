"""Process guardrails for rate limiting, cost tracking, and audit logging.

This module provides operational safeguards including request rate limiting,
API cost tracking, comprehensive audit logging, and a guarded client wrapper.
"""

import os
import json
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import anthropic


@dataclass
class GuardResult:
    """Result of a guardrail check (imported to avoid circular imports)."""
    passed: bool
    guard_name: str
    message: str
    severity: str
    details: Optional[dict] = None


@dataclass
class UsageRecord:
    """Record of a single API usage event."""
    timestamp: datetime
    user_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    request_type: str
    duration_ms: float


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 20
    requests_per_hour: int = 200
    tokens_per_minute: int = 100000
    tokens_per_hour: int = 1000000
    max_concurrent: int = 5


class RateLimiter:
    """Token bucket rate limiter with multiple time windows.

    Implements rate limiting for both request counts and token usage
    across minute and hour windows.
    """

    def __init__(self, config: RateLimitConfig = None):
        """Initialize rate limiter with configuration."""
        self.config = config or RateLimitConfig()
        self._lock = threading.Lock()

        # Track requests per user
        self._request_times: Dict[str, List[datetime]] = defaultdict(list)
        self._token_usage: Dict[str, List[tuple]] = defaultdict(list)  # (timestamp, tokens)
        self._concurrent: Dict[str, int] = defaultdict(int)

    def check_rate_limit(self, user_id: str, estimated_tokens: int = 0) -> GuardResult:
        """Check if request is within rate limits.

        Args:
            user_id: Identifier for the user/session.
            estimated_tokens: Estimated tokens for the request.

        Returns:
            GuardResult indicating if request is allowed.
        """
        with self._lock:
            now = datetime.now()

            # Clean old entries
            self._cleanup(user_id, now)

            # Check concurrent requests
            if self._concurrent[user_id] >= self.config.max_concurrent:
                return GuardResult(
                    passed=False,
                    guard_name="rate_limit",
                    message=f"Too many concurrent requests (max {self.config.max_concurrent})",
                    severity="block",
                    details={"concurrent": self._concurrent[user_id]}
                )

            # Check requests per minute
            minute_ago = now - timedelta(minutes=1)
            recent_requests = [t for t in self._request_times[user_id] if t > minute_ago]
            if len(recent_requests) >= self.config.requests_per_minute:
                return GuardResult(
                    passed=False,
                    guard_name="rate_limit",
                    message=f"Rate limit exceeded ({self.config.requests_per_minute}/min)",
                    severity="block",
                    details={"requests_last_minute": len(recent_requests)}
                )

            # Check requests per hour
            hour_ago = now - timedelta(hours=1)
            hourly_requests = [t for t in self._request_times[user_id] if t > hour_ago]
            if len(hourly_requests) >= self.config.requests_per_hour:
                return GuardResult(
                    passed=False,
                    guard_name="rate_limit",
                    message=f"Hourly rate limit exceeded ({self.config.requests_per_hour}/hr)",
                    severity="block",
                    details={"requests_last_hour": len(hourly_requests)}
                )

            # Check token usage per minute
            minute_tokens = sum(
                tokens for ts, tokens in self._token_usage[user_id]
                if ts > minute_ago
            )
            if minute_tokens + estimated_tokens > self.config.tokens_per_minute:
                return GuardResult(
                    passed=False,
                    guard_name="rate_limit",
                    message="Token rate limit exceeded",
                    severity="block",
                    details={"tokens_last_minute": minute_tokens}
                )

            return GuardResult(
                passed=True,
                guard_name="rate_limit",
                message="Within rate limits",
                severity="info",
                details={
                    "requests_last_minute": len(recent_requests),
                    "requests_last_hour": len(hourly_requests),
                    "tokens_last_minute": minute_tokens
                }
            )

    def record_request(self, user_id: str, tokens: int = 0) -> None:
        """Record a completed request."""
        with self._lock:
            now = datetime.now()
            self._request_times[user_id].append(now)
            if tokens > 0:
                self._token_usage[user_id].append((now, tokens))

    def start_request(self, user_id: str) -> None:
        """Mark start of a request (for concurrent tracking)."""
        with self._lock:
            self._concurrent[user_id] += 1

    def end_request(self, user_id: str) -> None:
        """Mark end of a request."""
        with self._lock:
            self._concurrent[user_id] = max(0, self._concurrent[user_id] - 1)

    def _cleanup(self, user_id: str, now: datetime) -> None:
        """Remove old entries."""
        hour_ago = now - timedelta(hours=1)
        self._request_times[user_id] = [
            t for t in self._request_times[user_id] if t > hour_ago
        ]
        self._token_usage[user_id] = [
            (ts, tokens) for ts, tokens in self._token_usage[user_id]
            if ts > hour_ago
        ]


class CostTracker:
    """Tracks API costs and provides usage statistics.

    Maintains running totals of API usage and costs with configurable
    budget alerts.
    """

    # Pricing per million tokens (as of 2024)
    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
    }

    def __init__(
        self,
        daily_budget: float = 10.0,
        session_budget: float = 5.0
    ):
        """Initialize cost tracker with budgets."""
        self.daily_budget = daily_budget
        self.session_budget = session_budget
        self._lock = threading.Lock()

        self._daily_cost: Dict[str, float] = defaultdict(float)
        self._session_cost: float = 0.0
        self._total_cost: float = 0.0
        self._records: List[UsageRecord] = []
        self._session_start = datetime.now()

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for a request."""
        pricing = self.PRICING.get(model, {"input": 3.0, "output": 15.0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def record_usage(
        self,
        user_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_type: str = "chat",
        duration_ms: float = 0
    ) -> UsageRecord:
        """Record API usage and calculate costs."""
        cost = self.calculate_cost(model, input_tokens, output_tokens)

        record = UsageRecord(
            timestamp=datetime.now(),
            user_id=user_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            request_type=request_type,
            duration_ms=duration_ms
        )

        with self._lock:
            today = datetime.now().strftime("%Y-%m-%d")
            self._daily_cost[today] += cost
            self._session_cost += cost
            self._total_cost += cost
            self._records.append(record)

        return record

    def check_budget(self, user_id: str) -> GuardResult:
        """Check if within budget limits."""
        with self._lock:
            today = datetime.now().strftime("%Y-%m-%d")
            daily_used = self._daily_cost[today]
            session_used = self._session_cost

        warnings = []
        if daily_used > self.daily_budget * 0.8:
            warnings.append(f"Daily spend at {daily_used/self.daily_budget:.0%} of budget")

        if session_used > self.session_budget * 0.8:
            warnings.append(f"Session spend at {session_used/self.session_budget:.0%} of budget")

        if daily_used >= self.daily_budget:
            return GuardResult(
                passed=False,
                guard_name="budget_check",
                message="Daily budget exceeded",
                severity="block",
                details={"daily_used": daily_used, "budget": self.daily_budget}
            )

        if session_used >= self.session_budget:
            return GuardResult(
                passed=False,
                guard_name="budget_check",
                message="Session budget exceeded",
                severity="block",
                details={"session_used": session_used, "budget": self.session_budget}
            )

        if warnings:
            return GuardResult(
                passed=True,
                guard_name="budget_check",
                message="; ".join(warnings),
                severity="warning",
                details={
                    "daily_used": daily_used,
                    "session_used": session_used,
                    "daily_budget": self.daily_budget,
                    "session_budget": self.session_budget
                }
            )

        return GuardResult(
            passed=True,
            guard_name="budget_check",
            message="Within budget",
            severity="info"
        )

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        with self._lock:
            today = datetime.now().strftime("%Y-%m-%d")
            return {
                "daily_cost": self._daily_cost[today],
                "session_cost": self._session_cost,
                "total_cost": self._total_cost,
                "request_count": len(self._records),
                "session_duration": str(datetime.now() - self._session_start)
            }


class AuditLogger:
    """Comprehensive logging for all LLM interactions.

    Logs all requests and responses for compliance, debugging,
    and analysis purposes.
    """

    def __init__(self, log_dir: Optional[str] = None):
        """Initialize audit logger."""
        self.log_dir = Path(log_dir) if log_dir else Path("/tmp/risk_lens_audit")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"

    def log_interaction(
        self,
        request_id: str,
        user_id: str,
        request_type: str,
        prompt: str,
        response: str,
        model: str,
        tokens: Dict[str, int],
        duration_ms: float,
        guardrails_report: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Log an LLM interaction."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "user_id": user_id,
            "request_type": request_type,
            "model": model,
            "prompt_preview": prompt[:500] if len(prompt) > 500 else prompt,
            "prompt_length": len(prompt),
            "response_preview": response[:500] if len(response) > 500 else response,
            "response_length": len(response),
            "tokens": tokens,
            "duration_ms": duration_ms,
            "guardrails": guardrails_report,
            "metadata": metadata
        }

        with self._lock:
            with open(self._log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def get_recent_logs(self, count: int = 100) -> List[Dict]:
        """Get recent log entries."""
        entries = []
        with self._lock:
            if self._log_file.exists():
                with open(self._log_file, "r") as f:
                    for line in f:
                        entries.append(json.loads(line))
        return entries[-count:]


class ProcessGuardrails:
    """Combined process guardrails with rate limiting, cost tracking, and logging."""

    def __init__(
        self,
        rate_config: Optional[RateLimitConfig] = None,
        daily_budget: float = 10.0,
        session_budget: float = 5.0,
        log_dir: Optional[str] = None
    ):
        """Initialize process guardrails."""
        self.rate_limiter = RateLimiter(rate_config)
        self.cost_tracker = CostTracker(daily_budget, session_budget)
        self.audit_logger = AuditLogger(log_dir)

    def pre_request_checks(
        self,
        user_id: str,
        estimated_tokens: int = 0
    ) -> List[GuardResult]:
        """Run all pre-request checks."""
        results = []
        results.append(self.rate_limiter.check_rate_limit(user_id, estimated_tokens))
        results.append(self.cost_tracker.check_budget(user_id))
        return results

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics."""
        return self.cost_tracker.get_usage_stats()


class GuardedClient:
    """Wrapper around Anthropic client with integrated guardrails.

    Provides automatic rate limiting, cost tracking, and logging
    for all API calls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        guardrails: Optional[ProcessGuardrails] = None,
        user_id: str = "default"
    ):
        """Initialize guarded client."""
        self.api_key = api_key or os.environ.get("CLAUDE_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.guardrails = guardrails or ProcessGuardrails()
        self.user_id = user_id
        self._request_counter = 0

    def create_message(
        self,
        model: str,
        max_tokens: int,
        messages: List[Dict],
        system: Optional[str] = None,
        tools: Optional[List] = None,
        **kwargs
    ) -> tuple:
        """Create a message with guardrails.

        Returns:
            Tuple of (response, GuardResult list, usage_record).
        """
        import uuid
        request_id = str(uuid.uuid4())[:8]
        self._request_counter += 1

        # Estimate tokens
        prompt_text = system or ""
        for msg in messages:
            if isinstance(msg.get("content"), str):
                prompt_text += msg["content"]
        estimated_tokens = len(prompt_text) // 4

        # Pre-request checks
        pre_results = self.guardrails.pre_request_checks(self.user_id, estimated_tokens)

        # Check for blocks
        for result in pre_results:
            if result.severity == "block":
                return None, pre_results, None

        # Make the request
        self.guardrails.rate_limiter.start_request(self.user_id)
        start_time = time.time()

        try:
            api_kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages
            }
            if system:
                api_kwargs["system"] = system
            if tools:
                api_kwargs["tools"] = tools
            api_kwargs.update(kwargs)

            response = self.client.messages.create(**api_kwargs)

            duration_ms = (time.time() - start_time) * 1000

            # Record usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            total_tokens = input_tokens + output_tokens

            self.guardrails.rate_limiter.record_request(self.user_id, total_tokens)
            usage_record = self.guardrails.cost_tracker.record_usage(
                user_id=self.user_id,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_type="chat",
                duration_ms=duration_ms
            )

            # Extract response text
            response_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    response_text += block.text

            # Log interaction
            self.guardrails.audit_logger.log_interaction(
                request_id=request_id,
                user_id=self.user_id,
                request_type="chat",
                prompt=prompt_text,
                response=response_text,
                model=model,
                tokens={"input": input_tokens, "output": output_tokens},
                duration_ms=duration_ms,
                guardrails_report={"pre_checks": [r.guard_name for r in pre_results]}
            )

            return response, pre_results, usage_record

        finally:
            self.guardrails.rate_limiter.end_request(self.user_id)
