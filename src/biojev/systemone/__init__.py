__all__ = ["BioJevSystemOneAdapter", "validate_request"]

def __getattr__(name):
    if name == "BioJevSystemOneAdapter":
        from .adapter import BioJevSystemOneAdapter
        return BioJevSystemOneAdapter
    if name == "validate_request":
        from .schemas import validate_request
        return validate_request
    raise AttributeError(name)
