"""
reliability_system.py — Google-level reliability and fault tolerance.

Features:
  - Circuit breakers
  - Automatic fallbacks
  - Health monitoring
  - Auto-recovery
  - Graceful degradation
"""

import logging
import time
import threading
from enum import Enum
from typing import Callable, Optional, Dict
from collections import deque

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 30.0):
        self._failure_threshold = failure_threshold
        self._timeout = timeout
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self._state == "open":
                if time.time() - self._last_failure_time > self._timeout:
                    self._state = "half_open"
                    logger.info("Circuit breaker: transitioning to half-open")
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Reset on successful call."""
        with self._lock:
            if self._state == "half_open":
                self._state = "closed"
                logger.info("Circuit breaker: closed (recovered)")
            self._failure_count = 0
    
    def _on_failure(self):
        """Increment failure count."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self._failure_threshold:
                self._state = "open"
                logger.warning(
                    "Circuit breaker: OPEN (failures: %d)",
                    self._failure_count
                )


class HealthMonitor:
    """Monitor system health and trigger recovery actions."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._metrics: Dict[str, deque] = {
            "fps": deque(maxlen=60),
            "error_rate": deque(maxlen=60),
            "latency": deque(maxlen=60),
        }
        self._status = HealthStatus.HEALTHY
        self._last_check = time.time()
    
    def record_metric(self, metric: str, value: float):
        """Record a health metric."""
        with self._lock:
            if metric in self._metrics:
                self._metrics[metric].append(value)
    
    def check_health(self) -> HealthStatus:
        """Check current health status."""
        with self._lock:
            now = time.time()
            
            # Check FPS
            if self._metrics["fps"]:
                avg_fps = sum(self._metrics["fps"]) / len(self._metrics["fps"])
                if avg_fps < 3.0:
                    self._status = HealthStatus.CRITICAL
                elif avg_fps < 5.0:
                    self._status = HealthStatus.UNHEALTHY
                elif avg_fps < 8.0:
                    self._status = HealthStatus.DEGRADED
                else:
                    self._status = HealthStatus.HEALTHY
            
            # Check error rate
            if self._metrics["error_rate"]:
                avg_error = sum(self._metrics["error_rate"]) / len(self._metrics["error_rate"])
                if avg_error > 0.5:
                    self._status = HealthStatus.CRITICAL
                elif avg_error > 0.2:
                    self._status = HealthStatus.UNHEALTHY
                elif avg_error > 0.1:
                    self._status = HealthStatus.DEGRADED
            
            self._last_check = now
            return self._status
    
    def get_status(self) -> HealthStatus:
        """Get current health status."""
        with self._lock:
            return self._status


class AutoRecovery:
    """Automatic recovery from failures."""
    
    def __init__(self):
        self._recovery_actions: Dict[str, Callable] = {}
        self._lock = threading.Lock()
    
    def register_recovery(self, component: str, action: Callable):
        """Register a recovery action for a component."""
        with self._lock:
            self._recovery_actions[component] = action
    
    def attempt_recovery(self, component: str) -> bool:
        """Attempt to recover a component."""
        with self._lock:
            action = self._recovery_actions.get(component)
            if action:
                try:
                    logger.info("Auto-recovery: attempting recovery for %s", component)
                    action()
                    logger.info("Auto-recovery: %s recovered successfully", component)
                    return True
                except Exception as e:
                    logger.error("Auto-recovery: %s failed: %s", component, e)
                    return False
        return False


class GracefulDegradation:
    """Gracefully degrade features when system is under stress."""
    
    def __init__(self):
        self._degradation_level = 0  # 0 = full, 1 = reduced, 2 = minimal
        self._lock = threading.Lock()
    
    def get_detection_cadence(self) -> int:
        """Get detection cadence based on degradation level."""
        with self._lock:
            if self._degradation_level == 0:
                return 2  # Full quality
            elif self._degradation_level == 1:
                return 4  # Reduced quality
            else:
                return 8  # Minimal quality
    
    def get_ocr_frequency(self) -> int:
        """Get OCR frequency based on degradation level."""
        with self._lock:
            if self._degradation_level == 0:
                return 5  # Every 5th frame
            elif self._degradation_level == 1:
                return 10  # Every 10th frame
            else:
                return 20  # Every 20th frame
    
    def set_degradation_level(self, level: int):
        """Set degradation level (0-2)."""
        with self._lock:
            self._degradation_level = max(0, min(2, level))
            logger.info("Graceful degradation: level set to %d", self._degradation_level)


# Module-level instances
circuit_breaker = CircuitBreaker()
health_monitor = HealthMonitor()
auto_recovery = AutoRecovery()
graceful_degradation = GracefulDegradation()
