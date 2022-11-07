from abc import ABC, abstractmethod

# Add other imports below


# implement this class for preprocessor
# or make edit in this class and remove @abstractmethod decorator
class Preprocessor(ABC):
    """
    Class with functions to preprocess images on the fly.

    Required Functions:
    - get:
        - Input:
            - image_path: str
                - path to image

            - transforms: list(str/int/other format)
                - list of transforms such as rotate/flip/resize/etc.

        - Returns:
            - image: np.ndarray (w, h, 3)
                - image matrix
    """

    @abstractmethod
    def get(self, image_path, transforms):
        pass
