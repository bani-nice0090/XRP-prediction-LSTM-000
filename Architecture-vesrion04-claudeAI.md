# Jules: Elite ML Prediction System - Implementation Brief

## Mission Statement
Build an institutional-grade prediction system that competes with top quantitative hedge funds. This is not a tutorial project - this is production software that handles real money.

## Architecture Reference
Complete system architecture is in `ARCHITECTURE.md`. Read it thoroughly. Every design decision is intentional.

## Core Philosophy: Intelligence Over Complexity

**DO NOT:**
- Create 50 files with scattered logic
- Build toy models that "work on paper"
- Use traditional ML approaches (random forests, XGBoost, etc.)
- Manually design neural networks
- Write academic code that never runs in production

**DO:**
- Write dense, intelligent code (~300 lines per module)
- Implement state-of-the-art deep learning
- Focus on the NAS engine (the competitive advantage)
- Optimize for real-world performance (speed + accuracy + risk)
- Build something you'd deploy with $10M

---

## Implementation Roadmap

### Phase 1: Foundation (Day 1)

#### 1.1 Project Structure
```bash
elite-ml-system/
├── config.yaml
├── requirements.txt
├── main.py
├── src/
│   ├── __init__.py
│   ├── data_engine.py
│   ├── nas_engine.py
│   ├── meta_controller.py
│   ├── feature_learner.py
│   ├── online_learner.py
│   ├── risk_predictor.py
│   ├── experience_memory.py
│   ├── executor.py
│   └── utils.py
└── tests/
    └── test_all.py
```

#### 1.2 Requirements (requirements.txt)
```txt
torch>=2.0.0
polars>=0.19.0
websockets>=11.0
redis>=5.0.0
faiss-cpu>=1.7.4
ray[tune]>=2.7.0
onnxruntime>=1.16.0
prometheus-client>=0.18.0
rich>=13.5.0
pyyaml>=6.0
pytest>=7.4.0
```

#### 1.3 Configuration (config.yaml)
Create a comprehensive config following the example in ARCHITECTURE.md. Make it production-ready.

---

### Phase 2: Data Engine (Day 2-3)

**File: `src/data_engine.py` (~500 lines)**

This is NOT about connecting to APIs. This is about intelligent data management.

```python
class IntelligentDataEngine:
    """
    Autonomous data collection, validation, and streaming.
    
    Key capabilities:
    1. Multi-source streaming (WebSockets + REST)
    2. Automatic quality scoring
    3. Adversarial validation (detect distribution shift)
    4. Adaptive sampling (higher frequency in volatility)
    5. Smart caching with LRU + frequency
    """
    
    def __init__(self, config):
        self.sources = self._init_sources(config['data']['sources'])
        self.cache = Redis(...)
        self.quality_model = self._train_quality_scorer()
        
    async def stream(self) -> AsyncGenerator[MarketData]:
        """
        Main data stream. Yields normalized market data.
        
        Returns:
            MarketData object with fields:
            - timestamp, symbol, price, volume
            - bid/ask, spread, order_imbalance
            - quality_score (0-1)
        """
        # Implementation details:
        # 1. Connect to all WebSocket sources
        # 2. Merge and deduplicate data
        # 3. Score data quality in real-time
        # 4. Filter low-quality data
        # 5. Normalize and yield
        
    def validate_quality(self, data: pd.DataFrame) -> float:
        """
        Score data quality using learned model.
        Checks: completeness, consistency, timeliness, anomalies
        """
        
    def detect_drift(self, recent: np.ndarray, historical: np.ndarray) -> float:
        """
        Adversarial validation to detect data drift.
        Trains classifier to distinguish old vs new data.
        High accuracy = drift detected.
        """
```

**Critical Requirements:**
1. Must handle 1M+ ticks per second
2. Must detect and handle bad data automatically
3. Must work with multiple data sources simultaneously
4. Implement proper error handling and reconnection
5. Use Polars for data processing (not Pandas)

**Data Sources to Implement:**
- **Polygon.io** (stocks) - REST + WebSocket
- **Coinbase Pro** (crypto) - WebSocket
- **Binance** (crypto) - WebSocket
- Use free tiers - no paid APIs required

---

### Phase 3: Neural Architecture Search Engine (Day 4-7)

**File: `src/nas_engine.py` (~400 lines)**

This is THE core innovation. This is what makes the system elite.

```python
class NASEngine:
    """
    Discovers optimal neural architectures using reinforcement learning.
    
    Architecture:
    - Controller: LSTM that generates architecture descriptions
    - Search space: {attention, conv1d, tcn, lstm, gru}
    - Training: REINFORCE algorithm
    - Reward: Sharpe ratio * accuracy / (latency + memory)
    - Population: Top 10 architectures evolve simultaneously
    """
    
    def __init__(self, config):
        self.controller = ArchitectureController(
            input_dim=128,
            hidden_dim=256,
            num_layers=2
        )
        self.search_space = self._define_search_space()
        self.population = []  # Top 10 models
        
    def search(self, data: np.ndarray, n_generations: int = 50):
        """
        Main NAS loop.
        
        For each generation:
        1. Controller samples N architectures
        2. Train each for K steps
        3. Evaluate on validation (Sharpe + accuracy + speed)
        4. Update controller with REINFORCE
        5. Keep top 10 in population
        6. Mutate and crossover population
        """
        
        for gen in range(n_generations):
            # Sample architectures
            architectures = self.controller.sample(n=20)
            
            # Train and evaluate in parallel (Ray)
            rewards = self._parallel_evaluate(architectures, data)
            
            # Update controller
            self.controller.update(architectures, rewards)
            
            # Evolve population
            self.population = self._evolve_population(
                self.population, 
                architectures, 
                rewards
            )
            
            # Display progress
            self._display_progress(gen, rewards)
    
    def _build_model(self, architecture: List[str]) -> nn.Module:
        """
        Build PyTorch model from architecture description.
        
        Example architecture:
        ['attention_8h', 'conv1d_64', 'tcn_dilation_8', 'attention_4h', 'dense']
        
        Translates to actual PyTorch layers.
        """
        
    def _evaluate_architecture(self, model: nn.Module, data) -> Dict:
        """
        Evaluate architecture on multiple criteria:
        - Sharpe ratio (most important)
        - Directional accuracy
        - Inference latency
        - Memory usage
        - Training time
        
        Returns weighted score.
        """
```

**Search Space Definition:**
```python
SEARCH_SPACE = {
    'attention': {
        'heads': [4, 8, 12],
        'dim': [128, 256, 512],
        'dropout': [0.1, 0.2]
    },
    'conv1d': {
        'filters': [32, 64, 128],
        'kernel': [3, 5, 7],
        'dilation': [1, 2, 4, 8]
    },
    'tcn': {
        'layers': [4, 6, 8],
        'channels': [64, 128, 256]
    },
    'lstm': {
        'units': [128, 256, 512],
        'bidirectional': [True, False]
    }
}
```

**Implementation Notes:**
1. Use Ray Tune for parallel architecture evaluation
2. Controller is a simple LSTM trained with REINFORCE
3. Must complete one generation in <30 minutes
4. Use early stopping for architecture evaluation (don't overtrain)
5. Save top architectures to disk

**This is complex. Take your time. Get it right.**

---

### Phase 4: Meta-Learning Controller (Day 8-10)

**File: `src/meta_controller.py` (~350 lines)**

```python
class MetaController:
    """
    Learns which models work best in which market regimes.
    Uses Model-Agnostic Meta-Learning (MAML).
    """
    
    def __init__(self, nas_engine: NASEngine):
        self.models = nas_engine.population  # Top 10 models
        self.regime_detector = self._init_regime_detector()
        self.meta_model = MAMLModel()
        
    def detect_regime(self, recent_data: np.ndarray) -> str:
        """
        Detect current market regime.
        
        Uses unsupervised learning (GMM or HDBSCAN) to discover regimes.
        Does NOT use pre-defined regimes.
        
        Returns:
            regime_id: str (e.g., "regime_4")
            confidence: float (0-1)
        """
        
    def select_model(self, regime: str) -> nn.Module:
        """
        Select best model for current regime.
        
        Based on:
        - Historical performance in this regime
        - Model specialization (some models are regime-specific)
        - Recent performance trend
        """
        
    def fast_adapt(self, model: nn.Module, support_set: Tuple) -> nn.Module:
        """
        MAML: Adapt model to new regime with few samples.
        
        Args:
            model: Pre-trained model
            support_set: (X, y) with 10-50 samples from new regime
            
        Returns:
            Adapted model that works well in new regime
        """
        # MAML algorithm:
        # 1. Clone model
        # 2. Take K gradient steps on support set
        # 3. Return adapted model
```

**Regime Detection Approach:**
```python
class RegimeDetector:
    """
    Discovers market regimes using unsupervised learning.
    """
    
    def __init__(self):
        self.gmm = GaussianMixture(n_components=12, covariance_type='full')
        self.feature_extractor = nn.Sequential(...)
        
    def fit(self, historical_data: np.ndarray):
        """
        Discover regimes from historical data.
        
        Features used:
        - Volatility (5 timeframes)
        - Trend strength
        - Volume patterns
        - Correlation structure
        - Order flow imbalance
        """
        
    def predict(self, recent_data: np.ndarray) -> Tuple[int, float]:
        """
        Predict current regime + confidence.
        """
```

**Key Implementation Details:**
1. Use Gaussian Mixture Model for regime clustering
2. Extract features using small neural network
3. MAML implementation: 2-3 inner loop steps, Adam optimizer
4. Cache regime predictions (update every 5 minutes, not every second)

---

### Phase 5: Feature Learner (Day 11-12)

**File: `src/feature_learner.py` (~300 lines)**

```python
class FeatureLearner:
    """
    Learns representations automatically.
    No manual feature engineering.
    """
    
    def __init__(self, config):
        # Temporal Fusion Transformer for time-series
        self.tft = TemporalFusionTransformer(
            input_size=8,  # raw features
            hidden_size=128,
            output_size=64,  # learned features
            attention_heads=4
        )
        
        # VAE for dimensionality reduction
        self.vae = VariationalAutoencoder(
            input_dim=64,
            latent_dim=32
        )
        
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """
        Transform raw data to learned features.
        
        Pipeline:
        Raw (8 features) → TFT (64 features) → VAE (32 features)
        
        Returns 32 learned features that capture:
        - Temporal patterns
        - Cross-asset relationships  
        - Market microstructure
        - Latent factors
        """
```

**Temporal Fusion Transformer Implementation:**
- Based on "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
- Multi-head attention over time
- Variable selection network
- Gated residual networks

**Don't implement from scratch. Use existing libraries:**
- `pytorch-forecasting` has TFT
- Just wrap it and integrate

---

### Phase 6: Online Learning (Day 13-14)

**File: `src/online_learner.py` (~300 lines)**

```python
class OnlineLearner:
    """
    Continuous learning without catastrophic forgetting.
    Uses Elastic Weight Consolidation (EWC).
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.fisher_matrix = None  # EWC Fisher information
        self.optimal_params = None  # Reference parameters
        self.replay_buffer = ReplayBuffer(max_size=10000)
        
    def update(self, new_data: Tuple[np.ndarray, np.ndarray]):
        """
        Update model with new data without forgetting old patterns.
        
        EWC Loss = Task Loss + λ * Σ F_i * (θ_i - θ*_i)^2
        where:
        - F_i = Fisher information (importance of parameter i)
        - θ*_i = optimal parameter from previous task
        - λ = forgetting prevention strength
        """
        
    def compute_fisher_matrix(self):
        """
        Compute Fisher information matrix.
        Measures importance of each parameter.
        """
        
    def streaming_update(self, x: np.ndarray, y: np.ndarray):
        """
        Single-sample update (for real-time learning).
        """
```

**Implementation Strategy:**
1. EWC prevents forgetting important patterns
2. Replay buffer maintains diverse examples
3. Adaptive learning rate per parameter
4. Stream updates every 100 samples (not every sample - too slow)

---

### Phase 7: Risk-Aware Predictor (Day 15-16)

**File: `src/risk_predictor.py` (~250 lines)**

```python
class RiskAwarePredictor:
    """
    Predicts full return distribution, not just point estimate.
    Optimizes for Sharpe ratio.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def predict(self, features: np.ndarray) -> PredictionOutput:
        """
        Predict return distribution.
        
        Returns:
            PredictionOutput with:
            - expected_return: E[R]
            - volatility: σ
            - quantiles: [0.05, 0.25, 0.5, 0.75, 0.95]
            - var_95: Value at Risk
            - cvar_95: Conditional VaR
            - sharpe_prediction: E[R] / σ
            - optimal_position: Kelly criterion
        """
        
    def compute_sharpe_loss(self, predictions: torch.Tensor, 
                           targets: torch.Tensor) -> torch.Tensor:
        """
        Custom loss that optimizes Sharpe ratio.
        
        Loss = -Sharpe = -(E[R] - R_f) / σ[R]
        """
```

**Quantile Regression Network:**
```python
class QuantileRegressionNetwork(nn.Module):
    """
    Predicts multiple quantiles simultaneously.
    Outputs full distribution, not just mean.
    """
    
    def forward(self, x):
        # Returns 5 quantiles: [0.05, 0.25, 0.5, 0.75, 0.95]
        # These represent the return distribution
```

---

### Phase 8: Experience Memory (Day 17)

**File: `src/experience_memory.py` (~200 lines)**

```python
class ExperienceMemory:
    """
    Intelligent memory system using vector database.
    Finds similar past situations.
    """
    
    def __init__(self):
        # FAISS for fast similarity search
        self.index = faiss.IndexFlatL2(128)  # 128-dim embeddings
        self.experiences = []
        
    def store(self, experience: Experience):
        """
        Store experience with embedding.
        
        Experience includes:
        - Market state
        - Prediction made
        - Actual outcome
        - Model used
        - Regime
        """
        
    def recall(self, current_state: np.ndarray, k: int = 5) -> List[Experience]:
        """
        Find K most similar past situations.
        Uses cosine similarity in embedding space.
        """
```

**Prioritized Experience Replay:**
- Store high-TD-error experiences (we learned a lot)
- Store rare regime experiences
- Forget experiences older than 1 year
- Maximum 1M experiences in memory

---

### Phase 9: Main System Integration (Day 18-19)

**File: `main.py` (~100 lines)**

```python
class ElitePredictionSystem:
    """
    Main system orchestrator.
    """
    
    def __init__(self, config):
        self.config = config
        self.data_engine = IntelligentDataEngine(config)
        self.nas_engine = NASEngine(config)
        self.meta_controller = MetaController(self.nas_engine)
        self.feature_learner = FeatureLearner(config)
        self.online_learner = None  # Initialized after NAS
        self.risk_predictor = None
        self.memory = ExperienceMemory()
        
    async def run(self):
        """
        Main system loop.
        """
        # Phase 1: Initial NAS (if no models exist)
        if not self._models_exist():
            await self._initial_nas()
        
        # Phase 2: Load best models
        self._load_models()
        
        # Phase 3: Main prediction loop
        async for data in self.data_engine.stream():
            # Extract features
            features = self.feature_learner.extract_features(data)
            
            # Detect regime
            regime = self.meta_controller.detect_regime(features)
            
            # Select best model
            model = self.meta_controller.select_model(regime)
            
            # Predict
            prediction = self.risk_predictor.predict(features)
            
            # Display
            self._display_prediction(prediction)
            
            # Learn online
            if data.has_label:
                self.online_learner.update(features, data.label)
                self.memory.store(Experience(...))
```

---

### Phase 10: Terminal Interface (Day 20)

**File: `src/utils.py` - Display Functions**

Use the `rich` library to create beautiful terminal output:

```python
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

def display_system_status(system: ElitePredictionSystem):
    """
    Real-time dashboard in terminal.
    """
    console = Console()
    
    with Live(console=console, refresh_per_second=2) as live:
        while True:
            # Create status panel
            table = create_status_table(system)
            panel = Panel(table, title="Elite ML System", border_style="green")
            live.update(panel)
            time.sleep(0.5)
```

Follow the terminal output format in ARCHITECTURE.md exactly. Make it look professional.

---

## Critical Implementation Requirements

### 1. Performance Benchmarks

Your implementation MUST meet these benchmarks:

```python
# Performance tests
def test_data_throughput():
    assert data_engine.throughput() > 1_000_000  # ticks/sec

def test_prediction_latency():
    assert predictor.latency() < 0.010  # 10ms

def test_nas_generation_time():
    assert nas_engine.generation_time() < 1800  # 30min

def test_memory_usage():
    assert system.memory_usage() < 4_000_000_000  # 4GB
```

### 2. Code Quality Standards

- **Type hints everywhere**: `def function(x: np.ndarray) -> Dict[str, float]:`
- **Docstrings**: Google style, comprehensive
- **Error handling**: Never crash, always recover
- **Logging**: Use Python logging module, structured logs
- **Tests**: >80% coverage

### 3. Mathematical Correctness

Implement these correctly:

**Sharpe Ratio:**
```
SR = (E[R] - R_f) / σ[R]
```

**Kelly Criterion:**
```
f* = (p * b - q) / b
where p = win probability, q = 1-p, b = win/loss ratio
```

**EWC Loss:**
```
L_EWC = L_task + (λ/2) * Σ F_i * (θ_i - θ*_i)^2
```

**Quantile Loss:**
```
L_q = Σ max[q * (y - ŷ), (q-1) * (y - ŷ)]
```

### 4. Production Readiness

- Graceful degradation (if one data source fails, use others)
- Automatic recovery from crashes
- Save state regularly (can resume after restart)
- Memory-efficient (stream processing, not loading everything)
- CPU-efficient (use torch.compile, ONNX for inference)

---

## Testing Strategy

**File: `tests/test_all.py`**

```python
import pytest

class TestDataEngine:
    def test_streaming(self):
        # Test data stream works
        
    def test_quality_scoring(self):
        # Test quality scores are reasonable
        
    def test_drift_detection(self):
        # Test drift detector works

class TestNASEngine:
    def test_architecture_building(self):
        # Test can build models from descriptions
        
    def test_search_loop(self):
        # Test NAS completes one generation
        
    def test_controller_learning(self):
        # Test controller improves over time

# ... more tests for each module
```

Run with: `pytest tests/ -v --cov=src`

---

## Validation Checklist

Before you consider it complete:

- [ ] Data engine handles 1M ticks/sec
- [ ] NAS discovers at least 5 unique architectures
- [ ] Meta-controller detects regime changes
- [ ] Predictions have confidence intervals
- [ ] Online learning updates without forgetting
- [ ] Risk metrics (Sharpe, VaR) are computed correctly
- [ ] Memory system recalls similar situations
- [ ] Terminal output is beautiful and informative
- [ ] All tests pass (>80% coverage)
- [ ] System runs for 24 hours without crashing
- [ ] Can resume after restart (saves state)
- [ ] Predictions are actually profitable on backtest

---

## What Success Looks Like

After 20 days of implementation:

1. **Run command:** `python main.py`

2. **System starts:**
   - Connects to data sources
   - Loads pre-trained models (or runs NAS if first time)
   - Begins streaming predictions

3. **Terminal shows:**
   - Live predictions with confidence intervals
   - Current market regime
   - Model performance metrics
   - Risk metrics (Sharpe, drawdown, VaR)
   - Learning progress (online updates)

4. **Performance:**
   - Predictions in <10ms
   - >70% directional accuracy
   - Sharpe ratio >2.0
   - Max drawdown <15%

5. **Adaptability:**
   - Detects regime changes within 30 minutes
   - Adapts to new regimes with <10 samples
   - Improves over time (online learning)

---

## Final Notes

### What Makes This Elite:

1. **NAS Engine** - Automatically discovers optimal models
2. **Meta-Learning** - Learns to learn quickly
3. **Online Learning** - Never stops improving
4. **Risk-First** - Optimizes Sharpe, not accuracy
5. **Production-Grade** - Fast, efficient, reliable

### Common Pitfalls to Avoid:

- ❌ Building toy models that work on paper only
- ❌ Ignoring performance optimization
- ❌ Not testing with real data
- ❌ Creating overly complex architecture
- ❌ Forgetting error handling
- ❌ Not validating mathematical correctness

### Your Goal:

Build something that could manage $10M. If you wouldn't trust it with real money, it's not done.

---

## You Got This 🚀

This is ambitious. This is hard. This is exactly what separates elite systems from mediocre ones.

Take your time. Build it right. Test everything. Make it bulletproof.

Remember: **Intelligence over complexity. Performance over features. Results over theory.**

Start with the NAS engine. That's the heart of the system. Get that right, and everything else follows.

Good luck.
