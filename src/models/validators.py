"""
Production-grade input validators using Pydantic
Ensures data integrity and provides clear error messages
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


class CalculateRequest(BaseModel):
    """Validate calculate endpoint request"""
    columns: List[str] = Field(..., min_length=1, description="Column names for the data")
    edited_rows: List[Dict[str, Any]] = Field(..., min_length=1, description="Edited data rows")
    
    @field_validator('columns')
    @classmethod
    def validate_columns(cls, v):
        """Ensure columns list is not empty and contains valid names"""
        if not v:
            raise ValueError("Columns list cannot be empty")
        
        # Check for required columns
        required_cols = {'brand'}
        if not any(col.lower() in required_cols for col in v):
            raise ValueError(f"Must include at least one of: {required_cols}")
        
        return v
    
    @field_validator('edited_rows')
    @classmethod
    def validate_edited_rows(cls, v):
        """Ensure edited_rows is not empty"""
        if not v:
            raise ValueError("Edited rows cannot be empty")
        
        if len(v) > 10000:
            raise ValueError("Too many rows. Maximum 10,000 rows allowed.")
        
        return v
    
    @model_validator(mode='after')
    def validate_rows_match_columns(self):
        """Ensure all rows have the expected structure"""
        if not self.edited_rows or not self.columns:
            return self
        
        # Check first row has reasonable keys
        first_row = self.edited_rows[0]
        if not isinstance(first_row, dict):
            raise ValueError("Each row must be a dictionary")
        
        # Warning: Not all column keys need to be in every row (some may be None/missing)
        # But at least check the structure is reasonable
        if len(first_row) > len(self.columns) * 2:
            raise ValueError("Row has too many keys compared to columns")
        
        return self


class MarketingChannelValue(BaseModel):
    """Validate marketing channel values"""
    value: float = Field(..., ge=0, description="Marketing spend must be non-negative")
    
    @field_validator('value')
    @classmethod
    def check_reasonable_range(cls, v):
        """Check value is in reasonable range"""
        if v > 1e12:  # 1 trillion
            raise ValueError("Marketing spend value is unreasonably large")
        return v


class BrandData(BaseModel):
    """Validate brand data"""
    brand: str = Field(..., min_length=1, max_length=200, description="Brand name")
    country: Optional[str] = Field(None, max_length=100, description="Country name")
    year: Optional[int] = Field(None, ge=2000, le=2100, description="Year")
    quarter: Optional[str] = Field(None, pattern="^Q[1-4]$", description="Quarter (Q1-Q4)")
    
    @field_validator('brand')
    @classmethod
    def validate_brand_name(cls, v):
        """Validate brand name doesn't contain malicious content"""
        if not v or v.strip() == '':
            raise ValueError("Brand name cannot be empty")
        
        # Basic sanitization
        dangerous_chars = ['<', '>', ';', '&', '|', '$', '`']
        if any(char in v for char in dangerous_chars):
            raise ValueError(f"Brand name contains invalid characters: {dangerous_chars}")
        
        return v.strip()
    
    @field_validator('country')
    @classmethod
    def validate_country(cls, v):
        """Validate country name"""
        if v and v.strip():
            valid_countries = {'brazil', 'colombia', 'us', 'united states', 'usa'}
            if v.lower() not in valid_countries:
                # Don't strictly reject, but normalize
                pass
            return v.strip()
        return v


class ForecastOutput(BaseModel):
    """Validate forecast output structure"""
    baseline: Dict[str, List[float]]
    simulated: Dict[str, List[float]]
    quarters: List[str]
    historical: Optional[Dict[str, Any]] = None
    
    @field_validator('baseline', 'simulated')
    @classmethod
    def validate_forecast_data(cls, v):
        """Ensure forecast data doesn't contain invalid values"""
        for brand, values in v.items():
            if not isinstance(values, list):
                raise ValueError(f"Values for {brand} must be a list")
            
            if len(values) != 4:
                raise ValueError(f"Expected 4 quarterly values for {brand}, got {len(values)}")
            
            for val in values:
                if not isinstance(val, (int, float)):
                    raise ValueError(f"All values must be numeric, got {type(val)}")
                
                if pd.isna(val) or pd.isnull(val):
                    raise ValueError(f"Values cannot be NaN for {brand}")
                
                if val < 0:
                    raise ValueError(f"Negative power values not allowed for {brand}")
                
                if val > 1000:
                    raise ValueError(f"Power value {val} is unreasonably high for {brand}")
        
        return v
    
    @field_validator('quarters')
    @classmethod
    def validate_quarters(cls, v):
        """Ensure quarters are in expected format"""
        if len(v) != 4:
            raise ValueError("Must have exactly 4 quarters")
        
        # Check format: "YYYY QN"
        import re
        pattern = r'^\d{4} Q[1-4]$'
        for q in v:
            if not re.match(pattern, q):
                raise ValueError(f"Invalid quarter format: {q}. Expected 'YYYY QN'")
        
        return v


def validate_dataframe_structure(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> None:
    """
    Validate DataFrame structure for production use
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
    
    Raises:
        ValueError: If validation fails
    """
    if df is None:
        raise ValueError("DataFrame cannot be None")
    
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    if len(df) > 100000:
        raise ValueError(f"DataFrame too large: {len(df)} rows. Maximum 100,000 rows.")
    
    if required_cols:
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for completely empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        raise ValueError(f"Columns are completely empty: {empty_cols}")
    
    # Check for duplicate column names
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        raise ValueError(f"Duplicate column names found: {duplicate_cols}")


def sanitize_numeric_data(data: Any) -> float:
    """
    Sanitize numeric data for JSON serialization
    
    Args:
        data: Numeric value to sanitize
    
    Returns:
        Clean float value (NaN, Inf -> 0.0)
    """
    if pd.isna(data) or pd.isnull(data):
        return 0.0
    
    if np.isinf(data):
        return 0.0
    
    try:
        val = float(data)
        # Clamp to reasonable range
        if val < -1e10 or val > 1e10:
            return 0.0
        return val
    except (ValueError, TypeError):
        return 0.0


def validate_marketing_features(df: pd.DataFrame) -> None:
    """
    Validate marketing feature values are reasonable
    
    Args:
        df: DataFrame with marketing features
    
    Raises:
        ValueError: If validation fails
    """
    marketing_cols = ['wholesalers', 'total_distribution', 'paytv', 'volume']
    
    for col in marketing_cols:
        if col in df.columns:
            # Check for negative values
            if (df[col] < 0).any():
                raise ValueError(f"Marketing feature '{col}' contains negative values")
            
            # Check for unreasonably large values
            if (df[col] > 1e12).any():
                raise ValueError(f"Marketing feature '{col}' contains unreasonably large values")
            
            # Check data type
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    raise ValueError(f"Marketing feature '{col}' contains non-numeric values")


__all__ = [
    'CalculateRequest',
    'MarketingChannelValue',
    'BrandData',
    'ForecastOutput',
    'validate_dataframe_structure',
    'sanitize_numeric_data',
    'validate_marketing_features',
]

