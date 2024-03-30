class SpiceError(Exception):
    pass


class AuthenticationError(SpiceError):
    pass


class APIConnectionError(SpiceError):
    pass
