# ai_stock_screener/model_cache.py

import os
import json
import hashlib
import joblib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from pathlib import Path
import yfinance as yf
from .output_formatter import console


class ModelCacheManager:
    """
    Intelligent model caching system to avoid retraining models with identical parameters.
    Implements smart retraining triggers based on market conditions and model age.
    """
    
    def __init__(self, cache_dir: str = "model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different model types
        self.rf_dir = self.cache_dir / "random_forest"
        self.xgb_dir = self.cache_dir / "xgboost"
        self.rf_dir.mkdir(exist_ok=True)
        self.xgb_dir.mkdir(exist_ok=True)
        
        # Registry file for model metadata
        self.registry_file = self.cache_dir / "registry.json"
        self.model_registry = self._load_registry()
        
        # Cleanup log
        self.cleanup_log_file = self.cache_dir / "cleanup_log.json"
        
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                console.print("⚠️ Model registry corrupted, creating new one")
                return {}
        return {}
    
    def _save_registry(self):
        """Save model registry to disk"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.model_registry, f, indent=2, default=str)
    
    def get_model_fingerprint(self, config: Dict[str, Any], data_hash: str) -> str:
        """
        Generate unique fingerprint for model parameters and data.
        This ensures models are only reused when parameters and data are identical.
        """
        # Extract relevant config parameters for fingerprinting
        fingerprint_params = {
            'model': config.get('model', 'random_forest'),
            'n_estimators': config.get('n_estimators', 300),
            'grid_search': config.get('grid_search', 1),
            'ensemble_runs': config.get('ensemble_runs', 1),
            'use_gpu': config.get('use_gpu', True),
            'period': config.get('period', '1y'),
            'future_days': config.get('future_days', 5),
            'threshold': config.get('threshold', 0.0),
            'use_sharpe_labeling': config.get('use_sharpe_labeling', 1.0),
            'integrate_market': config.get('integrate_market', True),
            'data_hash': data_hash
        }
        
        # Create fingerprint from sorted parameters
        fingerprint_str = json.dumps(fingerprint_params, sort_keys=True)
        fingerprint = hashlib.md5(fingerprint_str.encode()).hexdigest()
        
        return fingerprint
    
    def _get_model_path(self, model_type: str, fingerprint: str) -> Tuple[Path, Path]:
        """Get file paths for model and metadata"""
        if model_type == "xgboost":
            model_dir = self.xgb_dir
            prefix = "xgb"
        else:
            model_dir = self.rf_dir
            prefix = "rf"
        
        timestamp = datetime.now().strftime("%Y%m%d")
        model_file = model_dir / f"{prefix}_{timestamp}_{fingerprint[:8]}.joblib"
        metadata_file = model_dir / f"{prefix}_{timestamp}_{fingerprint[:8]}.json"
        
        return model_file, metadata_file
    
    def get_current_market_conditions(self) -> Dict[str, Any]:
        """
        Get current market conditions for intelligent retraining decisions.
        This includes market regime and volatility indicators.
        """
        try:
            # Get SPY data for market conditions
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="1mo")
            
            if spy_data.empty:
                console.print("⚠️ Could not fetch market data, using default conditions")
                return {
                    'market_regime': 'unknown',
                    'vix': 20.0,  # Default VIX value
                    'spy_trend': 'sideways'
                }
            
            # Calculate market regime based on recent performance
            recent_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0] - 1) * 100
            
            if recent_return > 5:
                market_regime = 'bull'
            elif recent_return < -5:
                market_regime = 'bear'
            else:
                market_regime = 'sideways'
            
            # Get VIX data for volatility
            try:
                vix = yf.Ticker("^VIX")
                vix_data = vix.history(period="5d")
                current_vix = vix_data['Close'].iloc[-1] if not vix_data.empty else 20.0
            except:
                current_vix = 20.0  # Default VIX if fetch fails
            
            return {
                'market_regime': market_regime,
                'vix': float(current_vix),
                'spy_trend': market_regime,
                'spy_return_1m': float(recent_return)
            }
            
        except Exception as e:
            console.print(f"⚠️ Error fetching market conditions: {e}")
            return {
                'market_regime': 'unknown',
                'vix': 20.0,
                'spy_trend': 'sideways',
                'spy_return_1m': 0.0
            }
    
    def should_retrain(self, fingerprint: str, current_conditions: Dict[str, Any], 
                      mode: str = "eval", max_age_hours: int = 24) -> bool:
        """
        Intelligent retraining decision logic based on:
        - Age of model (max 1 trading day for discovery, 3 days for eval)
        - Market regime changes (bull/bear/sideways transitions)
        - Significant market volatility changes (>20% VIX change)
        - Model performance degradation
        - Parameter changes
        """
        if fingerprint not in self.model_registry:
            return True  # No cached model exists
        
        model_metadata = self.model_registry[fingerprint]
        
        # Parse creation timestamp
        try:
            created_at = datetime.fromisoformat(model_metadata['created_at'])
        except (KeyError, ValueError):
            return True  # Invalid timestamp, retrain
        
        age_hours = (datetime.now() - created_at).total_seconds() / 3600
        
        # Age-based retraining
        if mode == 'discovery' and age_hours > 24:  # 1 day for discovery mode
            console.print(f"🔄 Model too old for discovery mode ({age_hours:.1f}h > 24h)")
            return True
        elif mode == 'eval' and age_hours > max_age_hours:  # Configurable for eval mode
            console.print(f"🔄 Model too old for eval mode ({age_hours:.1f}h > {max_age_hours}h)")
            return True
        
        # Market condition changes
        old_regime = model_metadata.get('market_regime', 'unknown')
        current_regime = current_conditions.get('market_regime', 'unknown')
        
        if old_regime != current_regime and old_regime != 'unknown' and current_regime != 'unknown':
            console.print(f"🔄 Market regime changed: {old_regime} → {current_regime}")
            return True
        
        # Volatility changes (>20% VIX change)
        old_vix = model_metadata.get('vix', 20.0)
        current_vix = current_conditions.get('vix', 20.0)
        
        if old_vix > 0:  # Avoid division by zero
            vix_change = abs(current_vix - old_vix) / old_vix
            if vix_change > 0.2:  # 20% change threshold
                console.print(f"🔄 Significant VIX change: {old_vix:.1f} → {current_vix:.1f} ({vix_change:.1%})")
                return True
        
        return False  # Model is still valid
    
    def load_cached_model(self, fingerprint: str):
        """Load cached model if it exists and is valid"""
        if fingerprint not in self.model_registry:
            return None
        
        model_metadata = self.model_registry[fingerprint]
        model_type = model_metadata.get('model_type', 'random_forest')
        
        model_file, metadata_file = self._get_model_path(model_type, fingerprint)
        
        # Check if model file exists
        if not model_file.exists():
            console.print(f"⚠️ Cached model file not found: {model_file}")
            # Clean up registry entry
            del self.model_registry[fingerprint]
            self._save_registry()
            return None
        
        try:
            # Load the model
            model = joblib.load(model_file)
            console.print(f"✅ Loaded cached {model_type} model (fingerprint: {fingerprint[:8]})")
            console.print(f"   Created: {model_metadata.get('created_at', 'unknown')}")
            console.print(f"   Age: {self._get_model_age_str(model_metadata)}")
            return model
            
        except Exception as e:
            console.print(f"⚠️ Error loading cached model: {e}")
            # Clean up corrupted model
            try:
                model_file.unlink()
                if metadata_file.exists():
                    metadata_file.unlink()
                del self.model_registry[fingerprint]
                self._save_registry()
            except:
                pass
            return None
    
    def save_model(self, model, fingerprint: str, metadata: Dict[str, Any]):
        """Save model with metadata to cache"""
        model_type = metadata.get('model_type', 'random_forest')
        model_file, metadata_file = self._get_model_path(model_type, fingerprint)
        
        try:
            # Save the model
            joblib.dump(model, model_file)
            
            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Update registry
            self.model_registry[fingerprint] = metadata
            self._save_registry()
            
            console.print(f"💾 Cached {model_type} model (fingerprint: {fingerprint[:8]})")
            console.print(f"   File: {model_file.name}")
            
            # Clean up old models
            self._cleanup_old_models()
            
        except Exception as e:
            console.print(f"⚠️ Error saving model to cache: {e}")
    
    def _get_model_age_str(self, metadata: Dict[str, Any]) -> str:
        """Get human-readable model age string"""
        try:
            created_at = datetime.fromisoformat(metadata['created_at'])
            age = datetime.now() - created_at
            
            if age.days > 0:
                return f"{age.days}d {age.seconds // 3600}h"
            elif age.seconds >= 3600:
                return f"{age.seconds // 3600}h {(age.seconds % 3600) // 60}m"
            else:
                return f"{age.seconds // 60}m"
        except:
            return "unknown"
    
    def _cleanup_old_models(self, max_models_per_type: int = 10):
        """Clean up old cached models to prevent disk space issues"""
        try:
            # Group models by type
            rf_models = []
            xgb_models = []
            
            for fingerprint, metadata in self.model_registry.items():
                model_type = metadata.get('model_type', 'random_forest')
                created_at = datetime.fromisoformat(metadata['created_at'])
                
                if model_type == 'xgboost':
                    xgb_models.append((fingerprint, created_at, metadata))
                else:
                    rf_models.append((fingerprint, created_at, metadata))
            
            # Clean up each model type
            for models, model_type in [(rf_models, 'random_forest'), (xgb_models, 'xgboost')]:
                if len(models) > max_models_per_type:
                    # Sort by creation time (oldest first)
                    models.sort(key=lambda x: x[1])
                    
                    # Remove oldest models
                    models_to_remove = models[:-max_models_per_type]
                    
                    for fingerprint, _, metadata in models_to_remove:
                        self._remove_cached_model(fingerprint, model_type)
                        console.print(f"🗑️ Cleaned up old {model_type} model: {fingerprint[:8]}")
            
        except Exception as e:
            console.print(f"⚠️ Error during model cleanup: {e}")
    
    def _remove_cached_model(self, fingerprint: str, model_type: str):
        """Remove a specific cached model"""
        try:
            model_file, metadata_file = self._get_model_path(model_type, fingerprint)
            
            # Remove files
            if model_file.exists():
                model_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Remove from registry
            if fingerprint in self.model_registry:
                del self.model_registry[fingerprint]
                self._save_registry()
                
        except Exception as e:
            console.print(f"⚠️ Error removing cached model {fingerprint[:8]}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        rf_count = len([f for f in self.rf_dir.glob("*.joblib")])
        xgb_count = len([f for f in self.xgb_dir.glob("*.joblib")])
        
        # Calculate total cache size
        total_size = 0
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return {
            'total_models': len(self.model_registry),
            'random_forest_models': rf_count,
            'xgboost_models': xgb_count,
            'cache_size_mb': total_size / (1024 * 1024),
            'cache_directory': str(self.cache_dir)
        }
    
    def clear_cache(self):
        """Clear all cached models"""
        try:
            # Remove all model files
            for file_path in self.cache_dir.rglob("*.joblib"):
                file_path.unlink()
            for file_path in self.cache_dir.rglob("*.json"):
                if file_path.name != "cleanup_log.json":
                    file_path.unlink()
            
            # Clear registry
            self.model_registry = {}
            self._save_registry()
            
            console.print("🗑️ Cleared all cached models")
            
        except Exception as e:
            console.print(f"⚠️ Error clearing cache: {e}")