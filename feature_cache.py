"""
Feature Caching System for Helformer
Caches feature engineering results to avoid recomputation
"""

import os
import hashlib
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from improved_training_utils import create_research_based_features
from config_helformer import config

class FeatureCache:
    """
    Caches feature engineering results to disk
    """
    
    def __init__(self, cache_dir=None):
        # Use configuration value if not provided
        self.cache_dir = Path(cache_dir or config.FEATURE_CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_data_hash(self, data):
        """Generate hash of data to detect changes"""
        # Create hash based on data shape, first/last rows, and some sample data
        hash_data = {
            'shape': data.shape,
            'columns': list(data.columns),
            'first_row': data.iloc[0].to_dict() if len(data) > 0 else {},
            'last_row': data.iloc[-1].to_dict() if len(data) > 0 else {},
            'sample_middle': data.iloc[len(data)//2].to_dict() if len(data) > 2 else {}
        }
        
        # Convert to string and hash
        hash_str = str(hash_data)
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def _get_cache_path(self, asset, data_hash):
        """Get cache file path for asset and data hash"""
        return self.cache_dir / f"{asset}_{data_hash}_features.pkl"
    
    def get_features(self, asset, data):
        """
        Get features for asset, using cache if available
        
        Args:
            asset: Asset symbol (e.g., 'BTC')
            data: Raw price data DataFrame
            
        Returns:
            DataFrame with features added
        """
        
        # Generate hash of input data
        data_hash = self._get_data_hash(data)
        cache_path = self._get_cache_path(asset, data_hash)
        
        # Try to load from cache
        if cache_path.exists():
            try:
                print(f"📁 Loading cached features for {asset}...")
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Verify cache integrity
                if (len(cached_data) == len(data) and 
                    list(cached_data.columns[:6]) == list(data.columns)):
                    print(f"✅ Cache hit for {asset} - {len(cached_data)} records loaded")
                    return cached_data
                else:
                    print(f"⚠️ Cache validation failed for {asset}, regenerating...")
                    
            except Exception as e:
                print(f"⚠️ Cache read error for {asset}: {e}, regenerating...")
        
        # Cache miss or invalid - compute features
        print(f"🔄 Computing features for {asset} (cache miss)...")
        data_with_features = create_research_based_features(data.copy())
        
        # Save to cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data_with_features, f)
            print(f"💾 Features cached for {asset} at {cache_path}")
        except Exception as e:
            print(f"⚠️ Failed to cache features for {asset}: {e}")
        
        return data_with_features
    
    def clear_cache(self, asset=None):
        """Clear cache for specific asset or all assets"""
        
        if asset:
            # Clear cache for specific asset
            pattern = f"{asset}_*_features.pkl"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
                print(f"🗑️ Cleared cache: {cache_file}")
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*_features.pkl"):
                cache_file.unlink()
                print(f"🗑️ Cleared cache: {cache_file}")
    
    def list_cache(self):
        """List all cached feature files"""
        
        cache_files = list(self.cache_dir.glob("*_features.pkl"))
        
        if not cache_files:
            print("📁 No cached features found")
            return
        
        print(f"📁 Found {len(cache_files)} cached feature files:")
        for cache_file in sorted(cache_files):
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"  {cache_file.name} ({size_mb:.1f} MB)")
    
    def get_cache_stats(self):
        """Get cache statistics"""
        
        cache_files = list(self.cache_dir.glob("*_features.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'num_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }

# Global cache instance
feature_cache = FeatureCache()

def get_cached_features(asset, data):
    """
    Convenience function to get cached features
    
    Args:
        asset: Asset symbol (e.g., 'BTC')
        data: Raw price data DataFrame
        
    Returns:
        DataFrame with features added
    """
    return feature_cache.get_features(asset, data)

def clear_feature_cache(asset=None):
    """
    Convenience function to clear feature cache
    
    Args:
        asset: Asset symbol to clear (None for all)
    """
    feature_cache.clear_cache(asset)

def list_feature_cache():
    """List all cached features"""
    feature_cache.list_cache()

if __name__ == "__main__":
    # Demo/test the cache system
    cache = FeatureCache()
    cache.list_cache()
    stats = cache.get_cache_stats()
    print(f"\nCache stats: {stats}")