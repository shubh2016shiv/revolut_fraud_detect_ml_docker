from pycaret.classification import load_model


class InvalidModelError(Exception):
    """
    Exception when the model is loaded from the wrong path
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class LoadFraudDetectionModel:
    def __init__(self, model_path):
        self.__model_path = model_path

    def load(self):
        try:
            loaded_model = load_model(self.__model_path)
            print("Model Loaded Successfully!")

            if loaded_model:
                return loaded_model
            else:
                raise ValueError("Invalid model file")
        except FileNotFoundError:
            raise InvalidModelError(f"Model file not found: {self.__model_path}")
        except ValueError as ve:
            raise InvalidModelError(str(ve))


if __name__ == '__main__':
    model = LoadFraudDetectionModel("../resources/pycaret_models/saved_random_forest_model")
    model.load()
