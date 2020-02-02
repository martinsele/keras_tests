from typing import Dict, List, Tuple

import utils
import os.path

from full_evaluator import FullEvaluator
from utils import AnimalType, BreedName

TEST_TOP_N = 3
cat = "cat"
dog = "dog"
# TODO: need non-cropped images for overall validation (same original files as the cropped ones)
valid_folders: Dict[AnimalType, str] = {cat: os.path.join(utils.DATA_DIRS[cat], "non-cropped-test"),
                                        dog: os.path.join(utils.DATA_DIRS[dog], "non-cropped-test")}


def eval_test_result(test_results: List[Tuple[AnimalType, AnimalType, BreedName, List[BreedName]]]):
    """
    Evaluate test results
    :param test_results: consist of a list of 4-tuples (animal_true, animal_pred, breed_tue, [top N breed_pred])
    :return:
    """
    correct_animals = 0
    correct_breeds = 0
    correct_dogs = 0
    correct_cats = 0
    samples = len(test_results)
    cat_samples = sum([1 for s in test_results if s[0] == "cat"])
    dog_samples = samples - cat_samples
    for res in test_results:
        true_animal = res[0]
        true_breed = res[2]
        if true_animal == res[1]:
            correct_animals += 1
        if true_breed in res[3]:
            correct_breeds += 1
            if true_animal == "cat":
                correct_cats += 1
            else:
                correct_dogs += 1

    animal_acc = correct_animals / samples * 100
    breed_acc = correct_breeds / samples * 100
    dog_acc = correct_dogs / dog_samples * 100
    cat_acc = correct_cats / cat_samples * 100
    print(f"Results: animal_acc: {animal_acc:.2f}, breed_acc: {breed_acc:.2f}, "
          f"dog_acc: {dog_acc:.2f}, cat_acc: {cat_acc:.2f}")


# results consist of a list of 4-tuples (animal_true, animal_pred, breed_tue, [top N breed_pred])
results: List[Tuple[AnimalType, AnimalType, BreedName, List[BreedName]]] = []

model_dirs = []
for data_dir in utils.DATA_DIRS.values():
    model_dirs.append(os.path.join(data_dir, "models"))
model_dirs.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models"))

evaluator = FullEvaluator(img_size=utils.IMG_SIZE)
evaluator.load_models(model_dirs)

# TODO: list all valid_folders, extract files and their folder names (-> breed labels)
for animal, folder in valid_folders.items():
    animal_true = animal
    print(f"Animal {animal}")
    for breed_folder in os.listdir(folder):
        print(f"Breed {breed_folder}")
        breed_true = breed_folder
        for sample in os.listdir(os.path.join(folder, breed_folder)):
            file_name = os.path.join(folder, breed_folder, sample)
            class_res = evaluator.classify(file_name, top_n=TEST_TOP_N)
            print(f"{sample}: {class_res}")
            test_res = (animal_true, class_res.animal, breed_true, [breed[0] for breed in class_res.breeds])
            results.append(test_res)


# file_name = 'e:\\data\\dogs_cats\\stanford_dogs\\non-cropped-test\\n02085782-Japanese_spaniel\\n02085782_2100.jpg'
# class_res = evaluator.classify(file_name, top_n=TEST_TOP_N)

eval_test_result(results)


# nejsou posunute classy< egyptmau rika british shorthair