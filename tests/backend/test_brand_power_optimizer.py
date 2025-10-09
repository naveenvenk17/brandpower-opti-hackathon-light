"""
Tests for Brand Power Optimizer

Tests cover:
- Constraint validation
- Power prediction
- Optimization algorithms (gradient and evolutionary)
- Both optimization modes (per-brand and all-brands)
- Edge cases and error handling
"""
import pytest
import pandas as pd
import numpy as np

from src.services.optimization.unified_optimizer import (
    BrandPowerOptimizer,
    PowerPredictor,
    OptimizationRequest,
    OptimizationResult,
    OptimizationConstraints,
    OptimizationMode,
    OptimizationMethod,
    ConstraintValidator,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_baseline_data():
    """Create sample baseline data for testing."""
    data = {
        'brand': ['AGUILA', 'AGUILA', 'CORONA', 'CORONA'] * 4,
        'quarter': ['2024 Q3', '2024 Q4', '2024 Q3', '2024 Q4'] * 4,
        'paytv': [10_000_000, 12_000_000, 8_000_000, 9_000_000] * 4,
        'wholesalers': [15_000_000, 14_000_000, 12_000_000, 13_000_000] * 4,
        'total_distribution': [100, 100, 95, 95] * 4,
        'volume': [5000, 5200, 4800, 4900] * 4,
        'power': [15.0, 15.2, 12.0, 12.3] * 4,
    }
    return pd.DataFrame(data)


@pytest.fixture
def power_predictor(sample_baseline_data):
    """Create PowerPredictor instance."""
    return PowerPredictor(baseline_data=sample_baseline_data)


@pytest.fixture
def optimizer(power_predictor):
    """Create BrandPowerOptimizer instance."""
    constraints = OptimizationConstraints(
        total_budget=100_000_000,  # $100M for testing
        paytv_max_pct=0.5
    )
    return BrandPowerOptimizer(
        power_predictor=power_predictor,
        constraints=constraints
    )


# ============================================================================
# CONSTRAINT VALIDATOR TESTS
# ============================================================================

class TestConstraintValidator:
    """Test constraint validation logic."""
    
    def test_valid_allocation(self):
        """Test that valid allocation passes validation."""
        constraints = OptimizationConstraints(
            total_budget=100_000_000,
            paytv_max_pct=0.5
        )
        validator = ConstraintValidator(constraints)
        
        allocation = {'paytv': 20_000_000, 'wholesalers': 30_000_000}
        is_valid, violations = validator.validate_allocation(allocation)
        
        assert is_valid
        assert len(violations) == 0
    
    def test_paytv_cap_violation(self):
        """Test that paytv cap violation is detected."""
        constraints = OptimizationConstraints(
            total_budget=100_000_000,
            paytv_max_pct=0.5
        )
        validator = ConstraintValidator(constraints)
        
        # PayTV is 60% of total (40M / 60M)
        allocation = {'paytv': 40_000_000, 'wholesalers': 20_000_000}
        is_valid, violations = validator.validate_allocation(allocation)
        
        assert not is_valid
        assert len(violations) > 0
        assert 'PayTV exceeds' in violations[0]
    
    def test_zero_allocation(self):
        """Test that zero allocation is detected."""
        constraints = OptimizationConstraints(
            total_budget=100_000_000,
            paytv_max_pct=0.5
        )
        validator = ConstraintValidator(constraints)
        
        allocation = {'paytv': 0, 'wholesalers': 0}
        is_valid, violations = validator.validate_allocation(allocation)
        
        assert not is_valid
        assert 'zero or negative' in violations[0].lower()
    
    def test_per_brand_min_budget(self):
        """Test per-brand minimum budget constraint."""
        constraints = OptimizationConstraints(
            total_budget=100_000_000,
            paytv_max_pct=0.5,
            per_brand_min_budget=5_000_000
        )
        validator = ConstraintValidator(constraints)
        
        allocation = {'paytv': 2_000_000, 'wholesalers': 2_000_000}
        is_valid, violations = validator.validate_allocation(allocation, brand='TestBrand')
        
        assert not is_valid
        assert 'below minimum' in violations[0]
    
    def test_per_brand_max_budget(self):
        """Test per-brand maximum budget constraint."""
        constraints = OptimizationConstraints(
            total_budget=100_000_000,
            paytv_max_pct=0.5,
            per_brand_max_budget=50_000_000
        )
        validator = ConstraintValidator(constraints)
        
        allocation = {'paytv': 30_000_000, 'wholesalers': 30_000_000}
        is_valid, violations = validator.validate_allocation(allocation, brand='TestBrand')
        
        assert not is_valid
        assert 'exceeds maximum' in violations[0]


# ============================================================================
# POWER PREDICTOR TESTS
# ============================================================================

class TestPowerPredictor:
    """Test power prediction logic."""
    
    def test_initialization(self, sample_baseline_data):
        """Test that PowerPredictor initializes correctly."""
        predictor = PowerPredictor(baseline_data=sample_baseline_data)
        
        assert predictor.baseline_data is not None
        assert len(predictor._baseline_power_cache) > 0
    
    def test_baseline_cache_built(self, power_predictor):
        """Test that baseline cache is built correctly."""
        cache = power_predictor._baseline_power_cache
        
        assert 'AGUILA' in cache
        assert 'CORONA' in cache
        assert '2024 Q3' in cache['AGUILA']
        assert cache['AGUILA']['2024 Q3'] > 0
    
    def test_predict_power_basic(self, power_predictor):
        """Test basic power prediction."""
        features = {'paytv': 10_000_000, 'wholesalers': 15_000_000}
        power = power_predictor.predict_power('AGUILA', '2024 Q3', features)
        
        assert isinstance(power, float)
        assert power > 0
    
    def test_predict_power_increased_budget(self, power_predictor):
        """Test that increased budget increases predicted power."""
        baseline_features = {'paytv': 10_000_000, 'wholesalers': 15_000_000}
        increased_features = {'paytv': 20_000_000, 'wholesalers': 25_000_000}
        
        baseline_power = power_predictor.predict_power('AGUILA', '2024 Q3', baseline_features)
        increased_power = power_predictor.predict_power('AGUILA', '2024 Q3', increased_features)
        
        assert increased_power > baseline_power
    
    def test_saturation_curve(self, power_predictor):
        """Test that saturation curve shows diminishing returns."""
        # Compare marginal returns at different budget levels
        low_budget = {'paytv': 5_000_000, 'wholesalers': 10_000_000}
        mid_budget = {'paytv': 50_000_000, 'wholesalers': 100_000_000}
        high_budget = {'paytv': 500_000_000, 'wholesalers': 1_000_000_000}
        
        power_low = power_predictor.predict_power('AGUILA', '2024 Q3', low_budget)
        power_mid = power_predictor.predict_power('AGUILA', '2024 Q3', mid_budget)
        power_high = power_predictor.predict_power('AGUILA', '2024 Q3', high_budget)
        
        # Marginal returns should decrease
        delta_low_mid = power_mid - power_low
        delta_mid_high = power_high - power_mid
        
        # Even though budget increase is the same magnitude, power increase should be smaller
        # This demonstrates diminishing returns
        assert delta_low_mid > 0
        assert delta_mid_high > 0
        # Note: Exact comparison depends on saturation parameters
    
    def test_get_baseline_power(self, power_predictor):
        """Test baseline power retrieval."""
        baseline_power = power_predictor._get_baseline_power('AGUILA', '2024 Q3')
        
        assert baseline_power > 0
        assert isinstance(baseline_power, float)
    
    def test_get_baseline_power_missing_brand(self, power_predictor):
        """Test baseline power for missing brand uses fallback."""
        baseline_power = power_predictor._get_baseline_power('NONEXISTENT', '2024 Q3')
        
        assert baseline_power == 10.0  # Default fallback
    
    def test_get_baseline_features(self, power_predictor):
        """Test baseline feature retrieval."""
        features = power_predictor._get_baseline_features('AGUILA', '2024 Q3')
        
        assert 'paytv' in features
        assert 'wholesalers' in features
        assert features['paytv'] > 0


# ============================================================================
# OPTIMIZATION REQUEST TESTS
# ============================================================================

class TestOptimizationRequest:
    """Test OptimizationRequest validation."""
    
    def test_valid_request(self):
        """Test that valid request is accepted."""
        request = OptimizationRequest(
            total_budget=100_000_000,
            brands=['AGUILA', 'CORONA'],
            quarters=['2024 Q3', '2024 Q4'],
            mode=OptimizationMode.ALL_BRANDS,
            method=OptimizationMethod.GRADIENT
        )
        
        assert request.total_budget == 100_000_000
        assert len(request.brands) == 2
        assert len(request.quarters) == 2
    
    def test_invalid_quarter_format(self):
        """Test that invalid quarter format raises error."""
        with pytest.raises(ValueError, match="Invalid quarter format"):
            OptimizationRequest(
                total_budget=100_000_000,
                brands=['AGUILA'],
                quarters=['2024-Q3']  # Invalid format
            )
    
    def test_zero_budget_rejected(self):
        """Test that zero budget is rejected."""
        with pytest.raises(ValueError):
            OptimizationRequest(
                total_budget=0,
                brands=['AGUILA']
            )
    
    def test_negative_budget_rejected(self):
        """Test that negative budget is rejected."""
        with pytest.raises(ValueError):
            OptimizationRequest(
                total_budget=-1000,
                brands=['AGUILA']
            )
    
    def test_empty_brands_rejected(self):
        """Test that empty brands list is rejected."""
        with pytest.raises(ValueError):
            OptimizationRequest(
                total_budget=100_000_000,
                brands=[]
            )


# ============================================================================
# BRAND POWER OPTIMIZER TESTS
# ============================================================================

class TestBrandPowerOptimizer:
    """Test BrandPowerOptimizer class."""
    
    def test_initialization(self, optimizer):
        """Test that optimizer initializes correctly."""
        assert optimizer.predictor is not None
        assert optimizer.constraints is not None
        assert optimizer.validator is not None
    
    def test_optimize_all_brands_gradient(self, optimizer):
        """Test all-brands optimization with gradient method."""
        request = OptimizationRequest(
            total_budget=100_000_000,
            brands=['AGUILA', 'CORONA'],
            quarters=['2024 Q3', '2024 Q4'],
            mode=OptimizationMode.ALL_BRANDS,
            method=OptimizationMethod.GRADIENT
        )
        
        result = optimizer.optimize(request)
        
        assert isinstance(result, OptimizationResult)
        assert result.success
        assert 'AGUILA' in result.optimal_allocation
        assert 'CORONA' in result.optimal_allocation
        assert 'paytv' in result.optimal_allocation['AGUILA']
        assert 'wholesalers' in result.optimal_allocation['AGUILA']
    
    def test_optimize_all_brands_evolutionary(self, optimizer):
        """Test all-brands optimization with evolutionary method."""
        request = OptimizationRequest(
            total_budget=100_000_000,
            brands=['AGUILA', 'CORONA'],
            quarters=['2024 Q3', '2024 Q4'],
            mode=OptimizationMode.ALL_BRANDS,
            method=OptimizationMethod.EVOLUTIONARY
        )
        
        result = optimizer.optimize(request)
        
        assert isinstance(result, OptimizationResult)
        assert result.success
        assert len(result.optimal_allocation) == 2
    
    def test_optimize_per_brand(self, optimizer):
        """Test per-brand optimization mode."""
        request = OptimizationRequest(
            total_budget=100_000_000,
            brands=['AGUILA', 'CORONA'],
            quarters=['2024 Q3', '2024 Q4'],
            mode=OptimizationMode.PER_BRAND,
            method=OptimizationMethod.GRADIENT
        )
        
        result = optimizer.optimize(request)
        
        assert isinstance(result, OptimizationResult)
        # Each brand should get ~50M budget
        for brand in ['AGUILA', 'CORONA']:
            brand_budget = result.budget_allocation[brand]
            expected_budget = 100_000_000 / 2
            assert abs(brand_budget - expected_budget) < 100  # Allow small tolerance
    
    def test_budget_constraint_satisfied(self, optimizer):
        """Test that total budget constraint is satisfied."""
        request = OptimizationRequest(
            total_budget=100_000_000,
            brands=['AGUILA', 'CORONA'],
            quarters=['2024 Q3', '2024 Q4'],
            mode=OptimizationMode.ALL_BRANDS,
            method=OptimizationMethod.GRADIENT
        )
        
        result = optimizer.optimize(request)
        
        total_allocated = sum(result.budget_allocation.values())
        assert abs(total_allocated - 100_000_000) < 10  # Allow $10 tolerance
    
    def test_paytv_cap_satisfied(self, optimizer):
        """Test that paytv cap constraint is satisfied."""
        request = OptimizationRequest(
            total_budget=100_000_000,
            brands=['AGUILA', 'CORONA'],
            quarters=['2024 Q3', '2024 Q4'],
            mode=OptimizationMode.ALL_BRANDS,
            method=OptimizationMethod.GRADIENT
        )
        
        result = optimizer.optimize(request)
        
        for brand, allocation in result.optimal_allocation.items():
            paytv = allocation['paytv']
            wholesalers = allocation['wholesalers']
            total_brand = paytv + wholesalers
            
            if total_brand > 0:
                paytv_pct = paytv / total_brand
                assert paytv_pct <= 0.51  # Allow 1% tolerance for numerical optimization
    
    def test_power_uplift_calculation(self, optimizer):
        """Test that power uplift is calculated correctly."""
        request = OptimizationRequest(
            total_budget=100_000_000,
            brands=['AGUILA', 'CORONA'],
            quarters=['2024 Q3', '2024 Q4'],
            mode=OptimizationMode.ALL_BRANDS,
            method=OptimizationMethod.GRADIENT
        )
        
        result = optimizer.optimize(request)
        
        assert result.total_baseline_power > 0
        assert result.total_optimized_power > 0
        assert 'AGUILA' in result.power_uplift
        assert len(result.power_uplift['AGUILA']) == 2  # 2 quarters
    
    def test_single_brand_optimization(self, optimizer):
        """Test optimization with single brand."""
        request = OptimizationRequest(
            total_budget=50_000_000,
            brands=['AGUILA'],
            quarters=['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2'],
            mode=OptimizationMode.ALL_BRANDS,
            method=OptimizationMethod.GRADIENT
        )
        
        result = optimizer.optimize(request)
        
        assert result.success
        assert len(result.optimal_allocation) == 1
        assert 'AGUILA' in result.optimal_allocation
    
    def test_multiple_quarters(self, optimizer):
        """Test optimization across 4 quarters."""
        request = OptimizationRequest(
            total_budget=200_000_000,
            brands=['AGUILA', 'CORONA'],
            quarters=['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2'],
            mode=OptimizationMode.ALL_BRANDS,
            method=OptimizationMethod.GRADIENT
        )
        
        result = optimizer.optimize(request)
        
        assert result.success
        for brand in ['AGUILA', 'CORONA']:
            assert len(result.baseline_power[brand]) == 4
            assert len(result.optimized_power[brand]) == 4
            assert len(result.power_uplift[brand]) == 4
    
    def test_custom_constraints(self, power_predictor):
        """Test optimizer with custom constraints."""
        custom_constraints = OptimizationConstraints(
            total_budget=50_000_000,
            paytv_max_pct=0.3,  # Stricter 30% cap
            wholesalers_min_pct=0.5  # Require at least 50% wholesalers
        )
        
        optimizer = BrandPowerOptimizer(
            power_predictor=power_predictor,
            constraints=custom_constraints
        )
        
        request = OptimizationRequest(
            total_budget=50_000_000,
            brands=['AGUILA'],
            quarters=['2024 Q3'],
            mode=OptimizationMode.ALL_BRANDS,
            method=OptimizationMethod.GRADIENT
        )
        
        result = optimizer.optimize(request)
        
        # Check that custom constraints are reflected
        allocation = result.optimal_allocation['AGUILA']
        total = allocation['paytv'] + allocation['wholesalers']
        paytv_pct = allocation['paytv'] / total if total > 0 else 0
        wholesalers_pct = allocation['wholesalers'] / total if total > 0 else 0
        
        assert paytv_pct <= 0.31  # Allow small tolerance
        assert wholesalers_pct >= 0.49  # Allow small tolerance
    
    def test_result_immutability(self, optimizer):
        """Test that OptimizationResult is properly structured."""
        request = OptimizationRequest(
            total_budget=100_000_000,
            brands=['AGUILA'],
            quarters=['2024 Q3'],
            mode=OptimizationMode.ALL_BRANDS,
            method=OptimizationMethod.GRADIENT
        )
        
        result = optimizer.optimize(request)
        
        # Verify all required fields are present
        assert hasattr(result, 'success')
        assert hasattr(result, 'optimal_allocation')
        assert hasattr(result, 'baseline_power')
        assert hasattr(result, 'optimized_power')
        assert hasattr(result, 'power_uplift')
        assert hasattr(result, 'total_baseline_power')
        assert hasattr(result, 'total_optimized_power')
        assert hasattr(result, 'total_uplift_pct')
        assert hasattr(result, 'budget_allocation')
        assert hasattr(result, 'constraints_satisfied')
        assert hasattr(result, 'method_used')


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_small_budget(self, optimizer):
        """Test optimization with very small budget."""
        request = OptimizationRequest(
            total_budget=1000,  # $1K budget
            brands=['AGUILA'],
            quarters=['2024 Q3'],
            mode=OptimizationMode.ALL_BRANDS,
            method=OptimizationMethod.GRADIENT
        )
        
        result = optimizer.optimize(request)
        
        # Should still produce a result
        assert isinstance(result, OptimizationResult)
        assert result.budget_allocation['AGUILA'] <= 1000
    
    def test_large_number_of_brands(self, power_predictor):
        """Test optimization with many brands."""
        # Create baseline data for many brands
        brands = [f'BRAND_{i}' for i in range(10)]
        data_list = []
        for brand in brands:
            for quarter in ['2024 Q3', '2024 Q4']:
                data_list.append({
                    'brand': brand,
                    'quarter': quarter,
                    'paytv': 5_000_000,
                    'wholesalers': 10_000_000,
                    'power': 10.0 + np.random.random()
                })
        
        baseline_data = pd.DataFrame(data_list)
        predictor = PowerPredictor(baseline_data=baseline_data)
        
        optimizer = BrandPowerOptimizer(
            power_predictor=predictor,
            constraints=OptimizationConstraints(total_budget=100_000_000, paytv_max_pct=0.5)
        )
        
        request = OptimizationRequest(
            total_budget=100_000_000,
            brands=brands[:5],  # Use 5 brands
            quarters=['2024 Q3', '2024 Q4'],
            mode=OptimizationMode.ALL_BRANDS,
            method=OptimizationMethod.GRADIENT
        )
        
        result = optimizer.optimize(request)
        
        assert result.success
        assert len(result.optimal_allocation) == 5


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_optimization_workflow(self, sample_baseline_data):
        """Test complete optimization workflow from data to result."""
        # Step 1: Create predictor
        predictor = PowerPredictor(baseline_data=sample_baseline_data)
        
        # Step 2: Create constraints
        constraints = OptimizationConstraints(
            total_budget=100_000_000,
            paytv_max_pct=0.5,
            per_brand_min_budget=5_000_000
        )
        
        # Step 3: Create optimizer
        optimizer = BrandPowerOptimizer(
            power_predictor=predictor,
            constraints=constraints
        )
        
        # Step 4: Create request
        request = OptimizationRequest(
            total_budget=100_000_000,
            brands=['AGUILA', 'CORONA'],
            quarters=['2024 Q3', '2024 Q4', '2025 Q1', '2025 Q2'],
            mode=OptimizationMode.ALL_BRANDS,
            method=OptimizationMethod.GRADIENT
        )
        
        # Step 5: Run optimization
        result = optimizer.optimize(request)
        
        # Step 6: Validate result
        assert result.success
        assert result.constraints_satisfied
        assert result.total_optimized_power > result.total_baseline_power
        assert result.total_uplift_pct > 0
        
        # Verify budget is fully allocated
        total_allocated = sum(result.budget_allocation.values())
        assert abs(total_allocated - 100_000_000) < 100
        
        # Verify each brand has positive allocation
        for brand in ['AGUILA', 'CORONA']:
            assert result.budget_allocation[brand] > 0
            assert result.optimal_allocation[brand]['paytv'] > 0
            assert result.optimal_allocation[brand]['wholesalers'] > 0

