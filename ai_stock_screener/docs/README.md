# AI Stock Screener Documentation

Welcome to the AI Stock Screener documentation! This folder contains detailed explanations of the system's components and calculations.

## 📚 Documentation Index

### Core Calculations
- **[REGIME Calculation](regime-calculation.md)** - Market condition adjustments and regime analysis

### Feature Implementations
- **[Feature 4: Model Persistence and Reuse System](Model_Persistence_and_Reuse_System.md)** - Intelligent model caching implementation

### Coming Soon
- **SECTOR Calculation** - Sector rotation and performance analysis
- **NEWS Intelligence** - Alpha Vantage and Yahoo Finance sentiment analysis
- **Feature Engineering** - Technical indicators and ML features
- **Model Architecture** - Random Forest and XGBoost implementation
- **API Integration** - Market data sources and configuration

## 🎯 Quick Reference

The AI Stock Screener uses a multi-layered approach:

1. **Base ML Model** - Random Forest predictions on technical indicators
2. **REGIME Adjustment** - Market condition multipliers (VIX, Fear/Greed, etc.)
3. **SECTOR Adjustment** - Sector rotation and performance weighting
4. **NEWS Adjustment** - Sentiment analysis from financial news sources

Each layer refines the predictions to account for different market dynamics.

## 🚀 Contributing to Docs

When adding new documentation:
1. Create markdown files in this directory
2. Update this README index
3. Use clear examples and code snippets
4. Include emoji for better readability 📊
5. Reference actual code locations when relevant 