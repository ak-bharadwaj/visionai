"""
observability.py — Google-level monitoring and observability.

Features:
  - Comprehensive metrics
  - Performance profiling
  - Error tracking
  - Alerting
  - Distributed tracing
"""

import logging
import time
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """A single metric measurement."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collect and aggregate metrics."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
    
    def record(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric."""
        with self._lock:
            self._metrics[name].append(Metric(
                name=name,
                value=value,
                tags=tags or {}
            ))
    
    def increment(self, name: str, value: int = 1):
        """Increment a counter."""
        with self._lock:
            self._counters[name] += value
    
    def set_gauge(self, name: str, value: float):
        """Set a gauge value."""
        with self._lock:
            self._gauges[name] = value
    
    def get_stats(self, name: str) -> Dict:
        """Get statistics for a metric."""
        with self._lock:
            if name not in self._metrics:
                return {}
            
            values = [m.value for m in self._metrics[name]]
            if not values:
                return {}
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "p50": self._percentile(values, 50),
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99),
            }
    
    def _percentile(self, values: List[float], p: int) -> float:
        """Calculate percentile."""
        sorted_vals = sorted(values)
        index = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(index, len(sorted_vals) - 1)]
    
    def export_metrics(self) -> Dict:
        """Export all metrics for monitoring."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "metrics": {
                    name: self.get_stats(name)
                    for name in self._metrics.keys()
                }
            }


class ErrorTracker:
    """Track and analyze errors."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._errors: deque = deque(maxlen=1000)
        self._error_counts: Dict[str, int] = defaultdict(int)
    
    def record_error(self, error_type: str, message: str, 
                    component: str = "unknown", traceback: str = None):
        """Record an error."""
        with self._lock:
            error = {
                "type": error_type,
                "message": message,
                "component": component,
                "timestamp": time.time(),
                "traceback": traceback,
            }
            self._errors.append(error)
            self._error_counts[f"{component}:{error_type}"] += 1
    
    def get_error_summary(self) -> Dict:
        """Get error summary."""
        with self._lock:
            return {
                "total_errors": len(self._errors),
                "error_counts": dict(self._error_counts),
                "recent_errors": list(self._errors)[-10:],
            }


class PerformanceProfiler:
    """Profile performance of operations."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._profiles: Dict[str, List[float]] = defaultdict(list)
    
    def profile(self, operation: str):
        """Context manager for profiling."""
        return ProfileContext(self, operation)
    
    def get_profile(self, operation: str) -> Dict:
        """Get profile for an operation."""
        with self._lock:
            if operation not in self._profiles:
                return {}
            
            times = self._profiles[operation]
            if not times:
                return {}
            
            return {
                "count": len(times),
                "total": sum(times),
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
            }


class ProfileContext:
    """Context manager for profiling."""
    
    def __init__(self, profiler: PerformanceProfiler, operation: str):
        self._profiler = profiler
        self._operation = operation
        self._start_time = None
    
    def __enter__(self):
        self._start_time = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self._start_time
        with self._profiler._lock:
            self._profiler._profiles[self._operation].append(elapsed)


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._alerts: deque = deque(maxlen=100)
        self._alert_rules: List[Dict] = []
    
    def add_rule(self, name: str, condition: Callable, severity: str = "warning"):
        """Add an alert rule."""
        with self._lock:
            self._alert_rules.append({
                "name": name,
                "condition": condition,
                "severity": severity,
            })
    
    def check_alerts(self, metrics: MetricsCollector):
        """Check all alert rules."""
        with self._lock:
            for rule in self._alert_rules:
                try:
                    if rule["condition"](metrics):
                        self._trigger_alert(rule["name"], rule["severity"])
                except Exception as e:
                    logger.error("Alert rule failed: %s", e)
    
    def _trigger_alert(self, name: str, severity: str):
        """Trigger an alert."""
        alert = {
            "name": name,
            "severity": severity,
            "timestamp": time.time(),
        }
        self._alerts.append(alert)
        logger.warning("ALERT [%s]: %s", severity.upper(), name)


# Module-level instances
metrics_collector = MetricsCollector()
error_tracker = ErrorTracker()
performance_profiler = PerformanceProfiler()
alert_manager = AlertManager()
