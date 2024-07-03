class InjectionError(Exception):

    def __init__(self, message):
        super().__init__(self, message)

class WrongDtypeError(Exception):

    def __init__(self, message):
        msg = f"The input do NOT have the correct dtype! "
        message = msg + message
        super().__init__(self, message)

class CastingError(Exception):
    def __int__(self, message):
        msg = f"Error during casting of value! "
        message = msg+message
        super().__int__(self, message)

class DimensionError(Exception):
    def __init__(self, message):
        msg = f"Dimensions are not compatible! {message}"
        super().__init__(self, msg)