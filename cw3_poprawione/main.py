import itertools
import os.path
from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL.Image
import skimage.feature
import sklearn.decomposition
import sklearn.metrics
import sklearn.model_selection
import sklearn.svm
from PIL.Image import Image as ImageType

RAW_TEXTURES_PATH = "textures/raw"
SAMPLE_TEXTURES_PATH = "textures/samples"
TEXTURE_FEATURES = (
    "dissimilarity",
    "contrast",
    "correlation",
    "energy",
    "homogeneity",
    "ASM",
)
PIXEL_DISTANCES = ("1px", "3px", "5px")
ANGLES = ("0deg", "45deg", "90deg", "135deg")
CATEGORIES = ("floor", "table", "wall")
COLUMNS = [
    "_".join(prop)
    for prop in itertools.product(TEXTURE_FEATURES, PIXEL_DISTANCES, ANGLES)
] + ["Category"]


def get_full_names():
    return [
        "_".join(f)
        for f in itertools.product(TEXTURE_FEATURES, PIXEL_DISTANCES, ANGLES)
    ]


def generate_glcm_feature_array_from_image(greyscale_image: ImageType):
    image_data = np.array(greyscale_image)
    image_data_64 = (greyscale_image / np.max(image_data) * 63).astype("uint8")
    angles = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
    glcm = skimage.feature.greycomatrix(
        image_data_64, (1, 3, 5), angles, 64, True, True
    )
    feature_vector = []
    for feature in TEXTURE_FEATURES:
        feature_vector.extend(
            list(skimage.feature.greycoprops(glcm, feature).flatten())
        )

    return feature_vector


def generate_random_samples_from_image(
    image: ImageType, n: int, size: tuple[int, int]
) -> Generator[ImageType, None, None]:
    sample_width, sample_height = size
    random_points = zip(
        np.random.randint(0, image.width - sample_width, n),
        np.random.randint(0, image.height - sample_height, n),
    )
    for x, y in random_points:
        box = (x, y, x + sample_width, y + sample_height)
        image_sample = image.crop(box)
        yield image_sample


def main() -> None:

    feature_vectors = []

    for category in CATEGORIES:
        image = PIL.Image.open(
            os.path.join(RAW_TEXTURES_PATH, category, f"{category}.jpg")
        )

        image_samples = list(
            generate_random_samples_from_image(image=image, n=10, size=(128, 128))
        )

        for i, image_sample in enumerate(image_samples):
            image_sample.save(
                os.path.join(SAMPLE_TEXTURES_PATH, category, f"{category}_{i:02d}.jpg")
            )
            greyscale_image_sample = image_sample.convert("L")
            feature_vector = [
                *generate_glcm_feature_array_from_image(greyscale_image_sample),
                category,
            ]
            feature_vectors.append(feature_vector)

    full_feature_names = get_full_names()
    full_feature_names.append("Category")

    df = pd.DataFrame(data=feature_vectors, columns=COLUMNS)
    df.to_csv("textures.csv", sep=",", index=False)

    features = pd.read_csv("textures.csv", sep=",")

    data = np.array(features)
    x = (data[:, :-1]).astype("float64")
    y = data[:, -1]

    x_transform = sklearn.decomposition.PCA(n_components=3)
    xt = x_transform.fit_transform(x)

    red = y == "floor"
    blue = y == "wall"
    green = y == "table"

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ax.scatter(xt[red, 0], xt[red, 1], xt[red, 2], c="r")
    ax.scatter(xt[blue, 0], xt[blue, 1], xt[blue, 2], c="b")
    ax.scatter(xt[green, 0], xt[green, 1], xt[green, 2], c="g")

    classifier = sklearn.svm.SVC(gamma="auto")

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.33
    )

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(f"{accuracy=}")

    confusion_matrix = sklearn.metrics.confusion_matrix(
        y_test, y_pred, normalize="true"
    )

    print(f"{confusion_matrix=}")

    sklearn.metrics.plot_confusion_matrix(classifier, x_test, y_test, cmap=plt.cm.Blues)
    plt.show()


if __name__ == "__main__":
    main()
