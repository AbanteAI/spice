class SpiceError(Exception):
    pass


class AuthenticationError(SpiceError):
    pass


class APIConnectionError(SpiceError):
    pass


class InvalidModelError(SpiceError):
    pass


class InvalidProviderError(SpiceError):
    pass


class NoAPIKeyError(SpiceError):
    pass
