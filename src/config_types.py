from dataclasses import dataclass, field
from typing import Any, Dict, Type, get_type_hints

def safe_cast(value: Any, target_type: Type) -> Any:
    """Safely cast a value to a target type"""
    if target_type == int:
        return int(float(value))  # Handle both strings and floats
    elif target_type == float:
        return float(value)
    elif target_type == bool:
        return bool(value)
    elif target_type == str:
        return str(value)
    return value

@dataclass
class BaseConfig:
    """Base configuration class with type casting"""
    def __post_init__(self):
        # Get type hints for all fields
        hints = get_type_hints(self.__class__)
        
        # Cast each field to its proper type
        for field_name, field_type in hints.items():
            if hasattr(self, field_name):
                current_value = getattr(self, field_name)
                if current_value is not None:  # Skip None values
                    try:
                        casted_value = safe_cast(current_value, field_type)
                        setattr(self, field_name, casted_value)
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Could not cast {field_name} value {current_value} to {field_type}: {str(e)}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create a configuration instance from a dictionary"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}