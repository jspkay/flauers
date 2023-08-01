class InjectionError(Exception):

    def __init__(self, message):
        super().__init__(self, message)


class CastingError(Exception):
    def __int__(self, message):
        msg = f"Error during casting of value! "
        message = msg+message
        super().__int__(self, message)
