class InvalidDataEntryError(Exception):
    """
    Exception when the data entered in invalid
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class InvalidUserCreationDate(Exception):
    """
    Raised when the date on which the user is created is invalid
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)