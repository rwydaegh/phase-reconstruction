from dataclasses import dataclass, field
from typing import Any, Dict, Type, get_type_hints


def safe_cast(value: Any, target_type: Type) -> Any:
    if target_type is int:
        return int(float(value))  # Handle both strings and floats
    elif target_type is float:
        return float(value)
    if target_type is bool:
        return bool(value)
    elif target_type is str:
        return str(value)
    return value


@dataclass
class BaseConfig:

    def __post_init__(self):
        hints = get_type_hints(self.__class__)

        for field_name, field_type in hints.items():
            if hasattr(self, field_name):
                current_value = getattr(self, field_name)
                if current_value is not None:
                    try:
                        casted_value = safe_cast(current_value, field_type)
                        setattr(self, field_name, casted_value)
                    except (ValueError, TypeError) as e:
                        error_message = (
                            f"Could not cast {field_name} value {current_value} "
                            f"to {field_type}: {str(e)}"
                        )
                        raise ValueError(error_message) from e

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
