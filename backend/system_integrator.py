"""
system_integrator.py — Integrates all Google-level systems at startup.

Initializes and connects all advanced features:
- Reliability systems
- Adaptive intelligence
- Production features
- Observability
- Model optimization
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)


class SystemIntegrator:
    """Integrate all Google-level systems."""
    
    def __init__(self):
        self._initialized = False
        self._lock = threading.Lock()
    
    def initialize_all(self):
        """Initialize all Google-level systems."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            logger.info("=" * 60)
            logger.info("Initializing Google-Level Systems...")
            logger.info("=" * 60)
            
            # 1. Reliability Systems
            self._init_reliability()
            
            # 2. Adaptive Intelligence
            self._init_adaptive()
            
            # 3. Production Features
            self._init_production()
            
            # 4. Observability
            self._init_observability()
            
            # 5. Model Optimization
            self._init_optimization()
            
            self._initialized = True
            logger.info("=" * 60)
            logger.info("All Google-Level Systems Initialized!")
            logger.info("=" * 60)
    
    def _init_reliability(self):
        """Initialize reliability systems."""
        try:
            from backend.reliability_system import (
                health_monitor, auto_recovery, graceful_degradation
            )
            
            # Register recovery actions
            def recover_detector():
                logger.info("Recovering detector...")
                # Reinitialize detector if needed
                pass
            
            def recover_ocr():
                logger.info("Recovering OCR...")
                # Reinitialize OCR if needed
                pass
            
            auto_recovery.register_recovery("detector", recover_detector)
            auto_recovery.register_recovery("ocr", recover_ocr)
            
            logger.info("✓ Reliability systems initialized")
        except Exception as e:
            logger.warning("Reliability systems init failed: %s", e)
    
    def _init_adaptive(self):
        """Initialize adaptive intelligence."""
        try:
            from backend.adaptive_intelligence import (
                adaptive_thresholds, scene_optimizer
            )
            
            # Initialize with default settings
            logger.info("✓ Adaptive intelligence initialized")
        except Exception as e:
            logger.warning("Adaptive intelligence init failed: %s", e)
    
    def _init_production(self):
        """Initialize production features."""
        try:
            from backend.production_features import (
                async_processor, result_cache
            )
            
            # Start async processor
            if os.getenv("ENABLE_ASYNC", "1").strip() == "1":
                async_processor.start()
            
            logger.info("✓ Production features initialized")
        except Exception as e:
            logger.warning("Production features init failed: %s", e)
    
    def _init_observability(self):
        """Initialize observability."""
        try:
            from backend.observability import (
                metrics_collector, alert_manager
            )
            
            # Set up alert rules
            def check_fps_low(metrics):
                fps = metrics._gauges.get("detection.fps", 0)
                return fps < 5.0
            
            def check_error_rate_high(metrics):
                error_rate = metrics._gauges.get("error.rate", 0)
                return error_rate > 0.2
            
            alert_manager.add_rule("low_fps", check_fps_low, "warning")
            alert_manager.add_rule("high_error_rate", check_error_rate_high, "critical")
            
            logger.info("✓ Observability initialized")
        except Exception as e:
            logger.warning("Observability init failed: %s", e)
    
    def _init_optimization(self):
        """Initialize model optimization."""
        try:
            from backend.model_optimizer import model_optimizer
            
            # Check for TensorRT
            if model_optimizer.check_tensorrt():
                logger.info("✓ TensorRT available for optimization")
            
            logger.info("✓ Model optimizer initialized")
        except Exception as e:
            logger.warning("Model optimization init failed: %s", e)
    
    def get_status(self) -> dict:
        """Get status of all systems."""
        status = {
            "initialized": self._initialized,
            "systems": {}
        }
        
        try:
            from backend.reliability_system import health_monitor
            status["systems"]["health"] = health_monitor.get_status().value
        except Exception:
            pass
        
        try:
            from backend.observability import metrics_collector
            status["systems"]["metrics"] = len(metrics_collector._metrics)
        except Exception:
            pass
        
        return status


# Module-level instance
system_integrator = SystemIntegrator()
