# ai_screener/cli.py

import argparse
import sys
from ai_stock_screener.ai_screener import get_sp500_tickers, run_screening
from ai_stock_screener.gpu_utils import print_gpu_status, get_gpu_manager

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Stock Screener CLI")
    parser.add_argument("--mode", choices=["eval", "discovery"], default="eval",
                        help="Run mode: eval (use provided tickers) or discovery (scan S&P 500)")
    parser.add_argument("--tickers", type=str,
                        help="Comma-separated list of tickers for eval mode (e.g., AAPL,NVDA,MSFT)")
    parser.add_argument("--period", default="1y", help="Historical data period (default: 1y)")
    parser.add_argument("--future_days", type=int, default=5, help="Days ahead to evaluate returns (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.0, help="Label threshold (default: 0.0 for any growth)")
    parser.add_argument("--n_estimators", type=int, default=300, help="RandomForest trees (default: 300)")
    parser.add_argument("--use_sharpe_labeling", type=float, default=1.0, help="Enable return-volatility labeling with the given threshold (default: 1.0)")
    parser.add_argument("--model", type=str, default="random_forest", choices=["random_forest", "xgboost"],
                        help="Which model to train: 'random_forest' or 'xgboost'. Default is random_forest.")
    parser.add_argument("--grid_search", type=int, default=1, 
                        help="Enable grid search over model hyperparameters (1 = enabled, 0 = disabled)")
    parser.add_argument("--ensemble_runs", type=int, default=1, help="Number of ensemble runs (default: 1)")
    parser.add_argument("--no_integrate_market", action="store_true",
                    help="Disable integration of SPY market data into training (default: enabled)")
    parser.add_argument("--news", action="store_true",
                    help="Enable news sentiment analysis (adds processing time)")
    parser.add_argument("--no_gpu", action="store_true",
                    help="Disable GPU acceleration (force CPU-only mode)")
    parser.add_argument("--gpu_info", action="store_true",
                    help="Display GPU information and exit")
    parser.add_argument("--force_retrain", action="store_true",
                    help="Force model retraining (ignore cache)")
    parser.add_argument("--model_cache_dir", type=str, default="model_cache",
                    help="Directory for model cache (default: model_cache)")
    parser.add_argument("--max_model_age", type=int, default=72,
                    help="Maximum model age in hours before retraining (default: 72 for eval, 24 for discovery)")

    args = parser.parse_args()

    # Handle GPU info request
    if args.gpu_info:
        print("🚀 GPU Information:")
        print_gpu_status()
        gpu_manager = get_gpu_manager()
        gpu_info = gpu_manager.get_gpu_info()
        print(f"\nDetailed GPU Info:")
        for key, value in gpu_info.items():
            print(f"  {key}: {value}")
        sys.exit(0)

    config = {
        "period": args.period,
        "future_days": args.future_days,
        "threshold": args.threshold,
        "n_estimators": args.n_estimators,
        "use_sharpe_labeling": args.use_sharpe_labeling,
        "model": args.model,
        "grid_search": args.grid_search,
        "ensemble_runs": args.ensemble_runs,
        "integrate_market": not args.no_integrate_market,
        "use_gpu": not args.no_gpu,
        "force_retrain": args.force_retrain,
        "model_cache_dir": args.model_cache_dir,
        "max_model_age": args.max_model_age
    }

    if args.mode == "eval":
        if not args.tickers:
            from ai_stock_screener.output_formatter import console
            console.print("❌ Please provide --tickers for eval mode.")
            sys.exit(1)
        ticker_list = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
        run_screening(ticker_list, config, mode="eval", news_analysis=args.news)
    else:
        tickers = get_sp500_tickers()
        run_screening(tickers, config, mode="discovery", news_analysis=args.news)

if __name__ == "__main__":
    main()
