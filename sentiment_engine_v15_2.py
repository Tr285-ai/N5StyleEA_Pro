"""
Sentiment Analysis Engine v15.2

A real-time sentiment analysis module for financial markets.
Combines news, social media, and market data to generate sentiment signals.
"""

import os
import re
import json
import time
import logging
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, AsyncGenerator
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import defaultdict, deque
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderSentiment
import tweepy
from newsapi import NewsApiClient
import yfinance as yf
from bs4 import BeautifulSoup
import aiohttp
import pytz
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import gc

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sentiment_engine.log')
    ]
)
logger = logging.getLogger('sentiment_engine')

# Type aliases
DataFrame = pd.DataFrame
Array = np.ndarray

class SentimentSource(Enum):
    """Sources of sentiment data."""
    NEWS = 'news'
    TWITTER = 'twitter'
    REDDIT = 'reddit'
    YOUTUBE = 'youtube'
    FINANCIAL_NEWS = 'financial_news'
    BLOGS = 'blogs'
    FORUMS = 'forums'

class SentimentType(Enum):
    """Types of sentiment analysis."""
    VADER = 'vader'          # VADER sentiment
    TEXTBLOB = 'textblob'    # TextBlob sentiment
    CUSTOM_ML = 'custom_ml'  # Custom ML model
    ENSEMBLE = 'ensemble'    # Combined approach

@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""
    sources: List[SentimentSource] = field(default_factory=lambda: [
        SentimentSource.NEWS,
        SentimentSource.TWITTER
    ])
    sentiment_types: List[SentimentType] = field(default_factory=lambda: [
        SentimentType.VADER,
        SentimentType.TEXTBLOB
    ])
    update_interval: int = 300  # seconds
    max_retries: int = 3
    request_timeout: int = 10  # seconds
    cache_size: int = 1000     # Number of items to keep in cache
    min_confidence: float = 0.6  # Minimum confidence score to consider
    language: str = 'en'
    timezone: str = 'UTC'
    api_keys: Dict[str, str] = field(default_factory=dict)
    symbols: List[str] = field(default_factory=lambda: ['BTC-USD', 'ETH-USD'])
    keywords: List[str] = field(default_factory=list)
    model_path: str = 'models/sentiment/'
    use_ml: bool = True
    ml_confidence_threshold: float = 0.7
    save_raw_data: bool = True
    save_processed_data: bool = True

@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""
    source: SentimentSource
    sentiment_type: SentimentType
    score: float
    confidence: float
    text: str
    timestamp: float
    symbol: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['source'] = self.source.value
        result['sentiment_type'] = self.sentiment_type.value
        result['timestamp'] = datetime.fromtimestamp(
            self.timestamp, 
            tz=pytz.UTC
        ).isoformat()
        return result

class SentimentEngine:
    """
    Real-time sentiment analysis engine for financial markets.
    Aggregates and analyzes sentiment from multiple sources.
    """
    
    def __init__(
        self,
        config: Optional[Union[Dict, str, SentimentConfig]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None
    ):
        """
        Initialize the sentiment engine.
        
        Args:
            config: Configuration dictionary, path to config file, or SentimentConfig instance
            loop: Optional asyncio event loop
        """
        self.config = self._load_config(config) if config else SentimentConfig()
        self.loop = loop or asyncio.get_event_loop()
        self.cache = deque(maxlen=self.config.cache_size)
        self.sentiment_analyzers = self._initialize_analyzers()
        self.api_clients = self._initialize_api_clients()
        self._stop_event = asyncio.Event()
        self._update_task = None
        self._data_queue = asyncio.Queue()
        self._result_cache = {}
        self._ml_model = None
        self._vectorizer = None
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize ML model if enabled
        if self.config.use_ml:
            self._load_ml_model()
        
        logger.info("SentimentEngine initialized")
    
    def _load_config(self, config: Union[Dict, str, SentimentConfig]) -> SentimentConfig:
        """Load configuration from dict, file, or use default."""
        if isinstance(config, str):
            with open(config, 'r') as f:
                config_data = json.load(f)
            return SentimentConfig(**config_data)
        elif isinstance(config, dict):
            return SentimentConfig(**config)
        elif isinstance(config, SentimentConfig):
            return config
        else:
            raise ValueError("Invalid config type")
    
    def _initialize_analyzers(self) -> Dict[SentimentType, Any]:
        """Initialize sentiment analysis tools."""
        return {
            SentimentType.VADER: VaderSentiment(),
            SentimentType.TEXTBLOB: lambda x: TextBlob(x).sentiment.polarity,
            SentimentType.ENSEMBLE: self._ensemble_sentiment
        }
    
    def _initialize_api_clients(self) -> Dict[SentimentSource, Any]:
        """Initialize API clients for data sources."""
        clients = {}
        api_keys = self.config.api_keys
        
        try:
            if SentimentSource.NEWS in self.config.sources:
                news_api_key = api_keys.get('newsapi')
                if news_api_key:
                    clients[SentimentSource.NEWS] = NewsApiClient(api_key=news_api_key)
            
            if SentimentSource.TWITTER in self.config.sources:
                twitter_keys = {
                    'consumer_key': api_keys.get('twitter_consumer_key'),
                    'consumer_secret': api_keys.get('twitter_consumer_secret'),
                    'access_token': api_keys.get('twitter_access_token'),
                    'access_token_secret': api_keys.get('twitter_access_token_secret')
                }
                if all(twitter_keys.values()):
                    auth = tweepy.OAuthHandler(
                        twitter_keys['consumer_key'],
                        twitter_keys['consumer_secret']
                    )
                    auth.set_access_token(
                        twitter_keys['access_token'],
                        twitter_keys['access_token_secret']
                    )
                    clients[SentimentSource.TWITTER] = tweepy.API(
                        auth,
                        wait_on_rate_limit=True,
                        wait_on_rate_limit_notify=True
                    )
        
        except Exception as e:
            logger.error(f"Error initializing API clients: {e}")
        
        return clients
    
    def _load_ml_model(self) -> None:
        """Load pre-trained ML model and vectorizer."""
        try:
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                logger.warning(f"Model directory not found: {model_path}")
                return
                
            model_file = model_path / 'sentiment_model.joblib'
            vectorizer_file = model_path / 'vectorizer.joblib'
            
            if model_file.exists() and vectorizer_file.exists():
                self._ml_model = joblib.load(model_file)
                self._vectorizer = joblib.load(vectorizer_file)
                logger.info("Loaded ML model and vectorizer")
            else:
                logger.warning("ML model or vectorizer not found, falling back to rule-based")
                self.config.use_ml = False
        
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            self.config.use_ml = False
    
    async def start(self) -> None:
        """Start the sentiment analysis engine."""
        if self._update_task is None or self._update_task.done():
            self._stop_event.clear()
            self._update_task = asyncio.create_task(self._update_loop())
            logger.info("Sentiment engine started")
    
    async def stop(self) -> None:
        """Stop the sentiment analysis engine."""
        self._stop_event.set()
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("Sentiment engine stopped")
    
    async def _update_loop(self) -> None:
        """Main update loop for the sentiment engine."""
        while not self._stop_event.is_set():
            try:
                start_time = time.time()
                
                # Fetch data from all sources in parallel
                tasks = []
                for source in self.config.sources:
                    tasks.append(self._fetch_source_data(source))
                
                # Process results as they complete
                for task in asyncio.as_completed(tasks):
                    try:
                        source, items = await task
                        if items:
                            await self._process_items(source, items)
                    except Exception as e:
                        logger.error(f"Error processing source data: {e}")
                
                # Calculate aggregate sentiment
                await self._calculate_aggregate_sentiment()
                
                # Sleep until next update
                elapsed = time.time() - start_time
                sleep_time = max(0, self.config.update_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def _fetch_source_data(
        self,
        source: SentimentSource
    ) -> Tuple[SentimentSource, List[Dict[str, Any]]]:
        """Fetch data from a specific source."""
        try:
            if source == SentimentSource.NEWS:
                return source, await self._fetch_news()
            elif source == SentimentSource.TWITTER:
                return source, await self._fetch_tweets()
            elif source == SentimentSource.REDDIT:
                return source, await self._fetch_reddit()
            elif source == SentimentSource.YOUTUBE:
                return source, await self._fetch_youtube_comments()
            elif source == SentimentSource.FINANCIAL_NEWS:
                return source, await self._fetch_financial_news()
            elif source == SentimentSource.BLOGS:
                return source, await self._fetch_blogs()
            elif source == SentimentSource.FORUMS:
                return source, await self._fetch_forum_posts()
            else:
                logger.warning(f"Unsupported source: {source}")
                return source, []
        except Exception as e:
            logger.error(f"Error fetching data from {source}: {e}")
            return source, []
    
    async def _fetch_news(self) -> List[Dict[str, Any]]:
        """Fetch news articles."""
        if SentimentSource.NEWS not in self.api_clients:
            logger.warning("News API client not initialized")
            return []
        
        client = self.api_clients[SentimentSource.NEWS]
        results = []
        
        try:
            for symbol in self.config.symbols:
                response = client.get_everything(
                    q=symbol,
                    language=self.config.language,
                    sort_by='publishedAt',
                    page_size=100
                )
                
                for article in response.get('articles', []):
                    results.append({
                        'text': f"{article.get('title', '')}. {article.get('description', '')}",
                        'source': 'news',
                        'timestamp': article.get('publishedAt', ''),
                        'url': article.get('url', ''),
                        'symbol': symbol
                    })
        
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
        
        return results
    
    async def _fetch_tweets(self) -> List[Dict[str, Any]]:
        """Fetch tweets."""
        if SentimentSource.TWITTER not in self.api_clients:
            logger.warning("Twitter API client not initialized")
            return []
        
        client = self.api_clients[SentimentSource.TWITTER]
        results = []
        
        try:
            for symbol in self.config.symbols:
                query = f"${symbol} OR {symbol.replace('-', '')} -filter:retweets"
                
                # For demo purposes; in production, use proper rate limiting
                tweets = client.search_tweets(
                    q=query,
                    count=100,
                    tweet_mode='extended',
                    lang='en'
                )
                
                for tweet in tweets:
                    results.append({
                        'text': tweet.full_text,
                        'source': 'twitter',
                        'timestamp': tweet.created_at.timestamp(),
                        'url': f"https://twitter.com/user/status/{tweet.id}",
                        'symbol': symbol,
                        'retweets': tweet.retweet_count,
                        'likes': tweet.favorite_count
                    })
        
        except Exception as e:
            logger.error(f"Error fetching tweets: {e}")
        
        return results
    
    async def _fetch_reddit(self) -> List[Dict[str, Any]]:
        """Fetch Reddit posts and comments."""
        # Implement Reddit API integration
        # This is a placeholder - requires PRAW or similar library
        return []
    
    async def _fetch_youtube_comments(self) -> List[Dict[str, Any]]:
        """Fetch YouTube comments."""
        # Implement YouTube API integration
        # This is a placeholder - requires YouTube Data API
        return []
    
    async def _fetch_financial_news(self) -> List[Dict[str, Any]]:
        """Fetch financial news from specialized sources."""
        # Implement financial news API integration
        return []
    
    async def _fetch_blogs(self) -> List[Dict[str, Any]]:
        """Fetch blog posts."""
        # Implement blog scraping/API integration
        return []
    
    async def _fetch_forum_posts(self) -> List[Dict[str, Any]]:
        """Fetch forum posts."""
        # Implement forum scraping/API integration
        return []
    
    async def _process_items(
        self,
        source: SentimentSource,
        items: List[Dict[str, Any]]
    ) -> None:
        """Process items from a data source."""
        tasks = []
        
        for item in items:
            tasks.append(self._analyze_sentiment(source, item))
        
        # Process items in batches to avoid overwhelming the system
        for i in range(0, len(tasks), 10):
            batch = tasks[i:i+10]
            await asyncio.gather(*batch, return_exceptions=True)
    
    async def _analyze_sentiment(
        self,
        source: SentimentSource,
        item: Dict[str, Any]
    ) -> None:
        """Analyze sentiment of a single item."""
        try:
            text = item.get('text', '')
            if not text or not text.strip():
                return
                
            symbol = item.get('symbol', '')
            timestamp = item.get('timestamp', time.time())
            
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Analyze sentiment using all configured methods
            results = []
            
            for stype in self.config.sentiment_types:
                if stype == SentimentType.CUSTOM_ML and self._ml_model:
                    score, confidence = self._ml_predict_sentiment(cleaned_text)
                else:
                    score, confidence = self._get_sentiment_score(cleaned_text, stype)
                
                if confidence >= self.config.min_confidence:
                    result = SentimentResult(
                        source=source,
                        sentiment_type=stype,
                        score=score,
                        confidence=confidence,
                        text=cleaned_text,
                        timestamp=timestamp,
                        symbol=symbol,
                        metadata={
                            'original_text': text,
                            **{k: v for k, v in item.items() if k not in ['text', 'timestamp', 'symbol']}
                        }
                    )
                    results.append(result)
            
            # Store results
            if results:
                async with self._lock:
                    for result in results:
                        self.cache.append(result)
                        symbol_key = result.symbol or 'global'
                        if symbol_key not in self._result_cache:
                            self._result_cache[symbol_key] = []
                        self._result_cache[symbol_key].append(result)
                
                # Save to database or file if configured
                if self.config.save_processed_data:
                    await self._save_sentiment_results(results)
        
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis."""
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _get_sentiment_score(
        self,
        text: str,
        stype: SentimentType
    ) -> Tuple[float, float]:
        """Get sentiment score using the specified analyzer."""
        if not text:
            return 0.0, 0.0
        
        try:
            if stype == SentimentType.VADER:
                scores = self.sentiment_analyzers[SentimentType.VADER].polarity_scores(text)
                score = scores['compound']
                confidence = abs(score)  # Use absolute value as confidence
                return score, confidence
                
            elif stype == SentimentType.TEXTBLOB:
                score = self.sentiment_analyzers[SentimentType.TEXTBLOB](text)
                confidence = abs(score)  # Use absolute value as confidence
                return score, confidence
                
            elif stype == SentimentType.ENSEMBLE:
                return self.sentiment_analyzers[SentimentType.ENSEMBLE](text)
                
            else:
                logger.warning(f"Unsupported sentiment type: {stype}")
                return 0.0, 0.0
                
        except Exception as e:
            logger.error(f"Error in {stype} sentiment analysis: {e}")
            return 0.0, 0.0
    
    def _ml_predict_sentiment(self, text: str) -> Tuple[float, float]:
        """Predict sentiment using the ML model."""
        if not self._ml_model or not self._vectorizer:
            return 0.0, 0.0
        
        try:
            # Vectorize text
            X = self._vectorizer.transform([text])
            
            # Predict
            if hasattr(self._ml_model, 'predict_proba'):
                probas = self._ml_model.predict_proba(X)[0]
                pred_class = self._ml_model.classes_[np.argmax(probas)]
                confidence = np.max(probas)
            else:
                pred_class = self._ml_model.predict(X)[0]
                confidence = 1.0  # Fallback confidence
                
            # Convert to -1 to 1 range if needed
            if pred_class in [0, 1]:  # Binary classification
                score = (pred_class * 2) - 1  # Convert 0/1 to -1/1
            else:
                score = float(pred_class)
                
            return score, confidence
            
        except Exception as e:
            logger.error(f"Error in ML sentiment prediction: {e}")
            return 0.0, 0.0
    
    def _ensemble_sentiment(self, text: str) -> Tuple[float, float]:
        """Combine multiple sentiment analysis methods."""
        scores = []
        confidences = []
        
        for stype in self.config.sentiment_types:
            if stype != SentimentType.ENSEMBLE:
                score, confidence = self._get_sentiment_score(text, stype)
                if confidence >= self.config.min_confidence:
                    scores.append(score * confidence)  # Weight by confidence
                    confidences.append(confidence)
        
        if not scores:
            return 0.0, 0.0
            
        # Weighted average of scores
        total_confidence = sum(confidences)
        if total_confidence > 0:
            ensemble_score = sum(scores) / total_confidence
            ensemble_confidence = sum(confidences) / len(confidences)
        else:
            ensemble_score = 0.0
            ensemble_confidence = 0.0
            
        return ensemble_score, ensemble_confidence
    
    async def _calculate_aggregate_sentiment(self) -> None:
        """Calculate aggregate sentiment scores."""
        async with self._lock:
            for symbol, results in self._result_cache.items():
                if not results:
                    continue
                
                # Calculate average sentiment
                scores = [r.score * r.confidence for r in results]
                confidences = [r.confidence for r in results]
                
                if confidences:
                    avg_score = sum(scores) / sum(confidences)
                    avg_confidence = sum(confidences) / len(confidences)
                    
                    # Store or use the aggregate sentiment
                    logger.info(
                        f"Aggregate sentiment for {symbol or 'global'}: "
                        f"score={avg_score:.3f}, confidence={avg_confidence:.3f}, "
                        f"n={len(results)}"
                    )
    
    async def _save_sentiment_results(self, results: List[SentimentResult]) -> None:
        """Save sentiment results to storage."""
        try:
            # Create output directory if it doesn't exist
            output_dir = Path('data/sentiment_results')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get current timestamp for filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = output_dir / f"sentiment_{timestamp}.json"
            
            # Convert results to dict and save as JSON
            with open(filename, 'w') as f:
                json.dump([r.to_dict() for r in results], f, indent=2)
                
            logger.debug(f"Saved {len(results)} sentiment results to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving sentiment results: {e}")
    
    async def get_sentiment(
        self,
        symbol: Optional[str] = None,
        source: Optional[SentimentSource] = None,
        stype: Optional[SentimentType] = None,
        lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get sentiment analysis results.
        
        Args:
            symbol: Filter by symbol (e.g., 'BTC-USD')
            source: Filter by data source
            stype: Filter by sentiment analysis type
            lookback_hours: Only include results from the last N hours
            
        Returns:
            Dictionary with sentiment analysis results
        """
        now = time.time()
        lookback_sec = lookback_hours * 3600
        results = []
        
        async with self._lock:
            cache = self._result_cache.get(symbol or 'global', [])
            
            for result in cache:
                # Apply filters
                if source is not None and result.source != source:
                    continue
                if stype is not None and result.sentiment_type != stype:
                    continue
                if now - result.timestamp > lookback_sec:
                    continue
                
                results.append(result)
        
        # Calculate statistics
        if not results:
            return {
                'count': 0,
                'average_score': 0.0,
                'average_confidence': 0.0,
                'sources': {},
                'results': []
            }
        
        # Calculate statistics
        scores = [r.score for r in results]
        confidences = [r.confidence for r in results]
        sources = {}
        
        for r in results:
            src = r.source.value
            if src not in sources:
                sources[src] = 0
            sources[src] += 1
        
        return {
            'count': len(results),
            'average_score': sum(scores) / len(scores),
            'average_confidence': sum(confidences) / len(confidences),
            'sources': sources,
            'results': [r.to_dict() for r in results]
        }
    
    def train_ml_model(
        self,
        X: List[str],
        y: List[float],
        test_size: float = 0.2,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train a custom ML model for sentiment analysis.
        
        Args:
            X: List of text samples
            y: List of sentiment scores (-1 to 1)
            test_size: Fraction of data to use for testing
            save_model: Whether to save the trained model
            
        Returns:
            Dictionary with training results
        """
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, accuracy_score
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Create pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    stop_words='english'
                )),
                ('clf', RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight='balanced'
                ))
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = pipeline.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            accuracy = accuracy_score(
                [1 if y >= 0 else 0 for y in y_test],
                [1 if y >= 0 else 0 for y in y_pred]
            )
            
            # Save model if requested
            if save_model:
                model_dir = Path(self.config.model_path)
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Save model and vectorizer
                model_file = model_dir / 'sentiment_model.joblib'
                vectorizer_file = model_dir / 'vectorizer.joblib'
                
                joblib.dump(pipeline.named_steps['clf'], model_file)
                joblib.dump(pipeline.named_steps['tfidf'], vectorizer_file)
                
                # Update in-memory model
                self._ml_model = pipeline.named_steps['clf']
                self._vectorizer = pipeline.named_steps['tfidf']
                self.config.use_ml = True
            
            return {
                'success': True,
                'mse': mse,
                'accuracy': accuracy,
                'model_info': str(pipeline)
            }
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Example usage
async def main():
    # Initialize with API keys
    config = {
        'api_keys': {
            'newsapi': 'YOUR_NEWSAPI_KEY',
            'twitter_consumer_key': 'YOUR_TWITTER_API_KEY',
            'twitter_consumer_secret': 'YOUR_TWITTER_API_SECRET',
            'twitter_access_token': 'YOUR_TWITTER_ACCESS_TOKEN',
            'twitter_access_token_secret': 'YOUR_TWITTER_ACCESS_TOKEN_SECRET'
        },
        'symbols': ['BTC-USD', 'ETH-USD'],
        'update_interval': 300,  # 5 minutes
        'use_ml': True
    }
    
    # Create and start the sentiment engine
    engine = SentimentEngine(config)
    await engine.start()
    
    try:
        # Run for a while to collect data
        print("Collecting sentiment data... (press Ctrl+C to stop)")
        await asyncio.sleep(300)  # Run for 5 minutes
        
        # Get sentiment results
        results = await engine.get_sentiment(lookback_hours=1)
        print(f"Collected {results['count']} sentiment results")
        print(f"Average sentiment: {results['average_score']:.3f}")
        print(f"Average confidence: {results['average_confidence']:.3f}")
        print(f"Sources: {results['sources']}")
        
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())