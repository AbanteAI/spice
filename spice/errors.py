class SpiceError(Exception):
    pass


class AuthenticationError(SpiceError):
    pass


class APIConnectionError(SpiceError):
    pass


class APIError(SpiceError):
    pass


class UnknownModelError(SpiceError):
    """Raised when an unknown model is passed in but no provider is given"""


class InvalidModelError(SpiceError):
    """Raised when a model is used in a situation it can't be used in; i.e., a text model passed into embeddings, or response format given to an Anthropic model."""


class InvalidProviderError(SpiceError):
    pass


class NoAPIKeyError(SpiceError):
    pass


class ImageError(SpiceError):
    pass
